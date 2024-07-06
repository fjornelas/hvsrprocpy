# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

"""Functions for signal processing."""

import numpy as np
from scipy.signal.windows import tukey

__all__ = ['detrend_ts', 'tukey_window', 'apply_filter', 'apply_normalization', 'sta_lta_calc', 'split_into_windows']


def detrend_ts(ts, **kwargs):
    """
    Detrends a time series based on the specified method.

    Parameters
    ----------

    ts: ndarray
        The input time series.

    **kwargs:

        detrend_type: int
            The detrending method.
                0: No detrending.
                1: Weighted mean detrending. (default)
                2: Linear detrending.
                6: 5th degree polynomial detrending.
        t_front: int
            Percentage of data to be tapered at the beginning for Tukey window. Default is 5.
        t_end: int
            Percentage of data to be tapered at the end for Tukey window. Default is 5.

    returns array of detrended time series.

    """

    defaults = {'detrend_type': 1, 't_front': 10, 't_end': 10}

    kwargs = {**defaults, **kwargs}

    detrend_type = kwargs['detrend_type']
    t_front = kwargs['t_front']
    t_end = kwargs['t_end']

    if detrend_type == 0:
        # Apply no detrend
        return ts
    elif detrend_type == 1:
        # Apply a weighted mean detrend
        win = tukey(len(ts), alpha=((t_front + t_end) / 200))
        ts_avg = np.average(ts, weights=win)
        return ts - ts_avg
    elif detrend_type == 2:
        # Apply linear detrending
        x = np.arange(1, len(ts) + 1)
        return ts - np.polyval(np.polyfit(x, ts, 1), x)
    elif detrend_type == 6:
        # Apply 5th order polynomial detrending
        x = np.arange(1, len(ts) + 1)
        return ts - np.polyval(np.polyfit(x, ts, 5), x)
    else:
        raise ValueError("Invalid detrend option. Choose from 0, 1, 2, or 6.")


def tukey_window(ts, **kwargs):
    """
    Applies Tukey window to the given time series.

    Parameters
    ---------

    ts: ndarray
        The input time series.
    **kwargs
        t_front: int
            Percentage of data to be tapered at the beginning for Tukey window. Default is 5.
        t_end: int
            Percentage of data to be tapered at the end for Tukey window. Default is 5.
        sym: boolean
            generates a symmetric window.


    returns an array of the tapered time series.

    """

    t_front = kwargs.get('t_front', 10)
    t_end = kwargs.get('t_end', 10)
    sym = kwargs.get('sym', True)

    win = tukey(len(ts), alpha=((t_front + t_end) / 200), sym=sym)

    return ts * win


def apply_filter(ts, **kwargs):
    """
    Calculate filter based on given parameters.

    Parameters
    ----------
    ts : ndarray
        An array representing the time series data.

    **kwargs
        ts_dt: float
            Time step of ts
        fc : float
            The filtering frequency to be applied.
        npole: int
            A value specifying low pass or high pass, a value less than 0
            applies to a high pass filter, default = -5.
        is_causal: Boolean
            Indicates whether to apply a causal or acausal filter to the time series.
        order_zero_padding: int
            Indicated what order of zero padding to be applied.

    Returns
    -------
    ts : ndarray
        Time series data after applying the filter.
    res : dict
        A dictionary of results from computing the fft of the time series, containing:
        - flt_ts: filtered time series
        - flt_amp: amplitude of filtered FAS
        - flt_resp: filter response
        - flt_fft: filtered fft containing imaginary values
    """
    # Default parameter values
    fc = kwargs.get('fc', 0.042)
    dt = kwargs.get('ts_dt', 0.005)
    npole = kwargs.get('npole', -5)
    is_causal = kwargs.get('is_causal', False)
    order_zero_padding = kwargs.get('order_zero_padding', 2)

    # Compute the order in which to apply filter
    order = np.abs(npole)
    # Compute the number of points of the time series
    npts = len(ts)
    # Compute the number of points after applying zero padding
    order_zero_padding = int(order_zero_padding)
    nfft = 2 ** int(np.floor(np.log2(npts)) + order_zero_padding) if order_zero_padding > 0 else npts
    n_nyq = nfft // 2 + 1
    df = 1 / (nfft * dt)
    # Compute the frequency
    freq = np.arange(1, n_nyq + 1) * df

    # Ensure non-negative frequencies
    ts_padded = np.pad(ts, (0, nfft - npts), 'constant')
    ts_fft = np.fft.rfft(ts_padded)

    # Calculate filter
    if is_causal:
        filt_caus = complex(real=1)  # hs = complex filter response at frequency f
        if order == 0:
            return ts, filt_caus
        if np.isnan(fc):
            return ts, filt_caus
        freq[0] = 10 ** -6
        as_ = np.array([complex(val) for val in abs(freq / fc)])
        if npole < 0:
            as_ = 1 / as_  # if high-pass
        filt_caus = as_ - np.exp(complex(imag=np.pi * (0.5 + (((2 * 1.) - 1.) / (2. * order)))))
        if order > 1:
            for i in range(2, order + 1):
                filt_caus *= (as_ - np.exp(complex(imag=np.pi * (0.5 + (((2. * i) - 1.) / (2. * order))))))
        filt_caus = 1 / filt_caus
        ts_fft[:n_nyq] *= filt_caus
    else:
        # Apply acausal filter
        if npole < 0:
            filt_acaus = 1.0 / np.sqrt(1.0 + (np.divide(fc, freq, where=freq > 0)) ** (2.0 * order))
            filt_acaus[0] = 0
            ts_fft[:n_nyq] *= filt_acaus
        if npole > 1:
            filt_acaus = 1 / np.sqrt(1 + (np.divide(freq, fc, where=freq > 0)) ** (2.0 * order))
            filt_acaus[0] = 0
            ts_fft[:n_nyq] *= filt_acaus

    if is_causal:
        filt = filt_caus
    else:
        filt = filt_acaus
    # Invert the fft to get the time series
    ts_padded = np.fft.irfft(ts_fft)
    ts = ts_padded[:npts]

    # Filtered FAS
    ts_flt_amp = np.abs(ts_fft)

    res = {'flt_ts': ts, 'flt_amp': ts_flt_amp, 'flt_resp': filt, 'flt_fft': ts_fft}

    return ts, res


def apply_normalization(ts):
    """
    A function to normalize time series

    Parameters
    ----------

    ts: ndarray
        Time series array.

    returns normalized time series.

    """

    factor = np.linalg.norm(ts)

    ts_new = ts / factor

    return ts_new


def sta_lta_calc(ts, **kwargs):
    """
    Function that computes the sta/lta ratio

    Parameters
    ----------
    ts: ndarray
        Time series array of microtremor/earthquake amplitudes

    **kwargs:
        short_term: int
            Short term average length
        long_term: int
            Long term average length
        moving_term: int
            Moving term of the sta/lta length

    returns the ratio of the sta and lta

    """

    # Extract parameters from kwargs or set defaults
    short_term = kwargs.get('short_term', 1)
    long_term = kwargs.get('long_term', 30)
    moving_term = kwargs.get('moving_term', 1)

    num_moving = int(np.floor((len(ts) - short_term) / moving_term) + 1)
    sta_lta_ratio = np.zeros(num_moving)

    for i in range(num_moving):
        short_temp = ts[i * moving_term: i * moving_term + short_term]
        idx1 = (i * moving_term + short_term - 1) - (long_term - 1)

        if idx1 < 0:
            long_temp = ts
        else:
            long_temp = ts[idx1: i * moving_term + short_term]

        sta_lta_ratio[i] = np.mean(np.abs(short_temp)) / np.mean(np.abs(long_temp))

    return sta_lta_ratio


def split_into_windows(ts, dt, win_width, overlapping):
    """
    Function to split time series into individual discrete windows

    Parameters
    ----------

    ts: ndarray
        Array of time series data.
    dt: float
        Time step of time series.
    win_width: int
        Length of each individual window.
    overlapping: int
        Length of overlapping between each window.

    returns a windowed time series

    """

    num_wins = int(np.floor(len(ts) * dt / win_width))
    npts_win = int(np.floor(win_width / dt))
    npts_over = int(np.floor(overlapping / dt))
    win_moving = npts_win - npts_over
    new_shape = (num_wins, win_moving)

    ts_wins = np.reshape(ts[0:(win_moving * num_wins)], new_shape)

    return ts_wins
