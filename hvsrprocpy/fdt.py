# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

"""Functions for computing HVSR."""

import numpy as np

from parzenpy.parzen_smooth import parzenpy

__all__ = ['compute_horizontal_combination', 'smoothed_fas_ko', 'smooth_fas', 'fas_cal', ]


def compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='geometric_mean'):
    """
    Function to combine horizontal components

    Parameters
    ----------

    h1_sub: ndarray
        Windowed array of h1 component
    h2_sub: ndarray
        Windowed array of h2 component
    dt: float
        Time step of time series
    horizontal_comb: string
        Specifies which horizontal combination to apply to time series.

    returns combined horizontal windowed array.

    """
    num_wins = len(h1_sub)
    if horizontal_comb == 'ps_RotD50':
        # Implementation for 'ps_RotD50'
        fas_h1 = np.fft.rfft(h1_sub)
        fas_h2 = np.fft.rfft(h2_sub)
        freq = np.fft.rfftfreq(len(h1_sub[0]), dt)
        angle_rad = np.linspace(np.zeros((len(freq), num_wins)), np.full((len(freq), num_wins), 2.0 * np.pi), 180).T
        frot_motions = fas_h1[:, :, np.newaxis] * np.cos(angle_rad) + fas_h2[:, :, np.newaxis] * np.sin(angle_rad)
        fas_h = np.median(np.abs(frot_motions), axis=2)
    elif horizontal_comb == 'squared_average':
        # Implementation for 'squared_average'
        fas_h1 = np.fft.rfft(h1_sub)
        freq = np.fft.rfftfreq(len(h1_sub[0]), dt)
        fas_h2 = np.fft.rfft(h2_sub)
        fas_h = np.sqrt((np.abs(fas_h1) ** 2 + np.abs(fas_h2) ** 2) / 2)
    elif horizontal_comb == 'geometric_mean':
        # Implementation for 'geometric_mean'
        fas_h1 = np.fft.rfft(h1_sub)
        freq = np.fft.rfftfreq(len(h1_sub[0]), dt)
        fas_h2 = np.fft.rfft(h2_sub)
        fas_h = np.sqrt(np.abs(fas_h1) * np.abs(fas_h2))
    else:
        raise KeyError('Horizontal combination does not exist. Choose ps_RotD50, squared_average, or geometric_mean')

    return fas_h, freq


def smoothed_fas_ko(fas, f, fc, b=40):
    """
    Function applies Konno and Ohmachi smoothing to a fourier amplitude spectra(FAS)

    Parameters
    ----------

    f: ndarray
        Frequency array of the fft.
    fas: ndarray
        Amplitude of the fft.
    fc: ndarray
        resampled frequency in which to apply smoothing
    b: int
        Smoothing parameter, default = 40.

    returns a smoothed FAS using KO smoothing.
    """
    df = f[1] - f[0]
    w = np.max([1.0 / 10 ** (3 / b), df / min(fc)])
    filter_ko = ((f > w * fc[:, np.newaxis]) & (f < fc[:, np.newaxis] / w) & (f > 0) & (f != fc[:, np.newaxis]))
    weights = np.zeros((len(fc), len(f)))
    weights[filter_ko] = (np.abs(np.sin(b * np.log10((f / fc[:, np.newaxis])[filter_ko])) / (
            b * np.log10((f / fc[:, np.newaxis])[filter_ko])))) ** 4.0
    weights[f == fc[:, np.newaxis]] = 1.0
    num = np.sum(weights * fas[:, np.newaxis], axis=2)
    den = np.sum(weights, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        smooth_fas_ko = np.divide(num, den)
    return smooth_fas_ko


def smooth_fas(fc, fas, freq, ko_smooth_flag=True, parzen_flag=False, ko_smooth_b=40, parzen_bwidth=1.5):
    """
    Function that applies either Konno and Ohmachi (KO) Smoothing or Parzen Smoothing to a windowed time series

    Parameters
    ----------

    fc: ndarray
        Resampled frequencies to which apply smoothing.
    fas: ndarray
        Amplitude of the fft.
    freq: ndarray
        Frequency of the fft.
    ko_smooth_flag: boolean
        Specifies whether to apply KO smoothing, default = True
    parzen_flag: boolean
        Specifies whether to apply parzen smoothing, default = False
    ko_smooth_b: int
        Smoothing parameter to apply KO smoothing, a smaller value will apply more smoothing, default = 40.
    parzen_bwidth: float
        Smoothing parameter to apply parzen smoothing, a larger value to apply more smoothing, default = 1.5.

    returns the smoothed FAS.

    """

    # Apply Konno Ohmachi Smoothing
    if ko_smooth_flag:
        smoothed_fas = smoothed_fas_ko(np.abs(fas), freq, fc, ko_smooth_b)

    # Apply Parzen Smoothing
    if parzen_flag:
        smooth = parzenpy(freq=freq, fft=fas)
        smoothed_fas = smooth.apply_smooth(fc=fc, b=parzen_bwidth, windowed_flag=True)

    return smoothed_fas


def fas_cal(ts, dt=0.005, max_order=0):
    """
    Function that computes the FAS for a time series.

    Parameters
    ----------

    ts: ndarray
        Array of  time series.
    dt: float
        time step of time series.
    max_order: int
        order in which to compute the number of points for the fft.

    returns a dictionary of outputs such as the time series, fas amp, fas freq, and phase.

    """
    # Compute the number of points for the time series
    npts = len(ts)
    # Compute the number of points for the fft
    nfft = 2 ** (int(np.log2(npts)) + max_order) if max_order > 0 else npts
    # Compute the nyquist frequency
    n_nyq = nfft // 2 + 1
    # Compute the step of the fft
    df = 1 / (nfft * dt)
    # Compute the padded ts
    ts = np.concatenate((ts, np.zeros(nfft - npts)))
    # Compute the fft of the ts
    fft_ts = np.fft.rfft(ts)
    # Extract the amplitude part of the fft
    amp = np.abs(fft_ts[:n_nyq]) * dt
    # Compute the phase angle of the fft
    phase = np.angle(fft_ts[:n_nyq])
    # Create an array of frequencies
    freq = np.arange(n_nyq) * df

    output = {'fft_ts': fft_ts, 'amp': amp, 'freq': freq, 'phase': phase}

    return output
