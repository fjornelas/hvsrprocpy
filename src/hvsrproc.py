# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

"""Class definition for HvsrProc object."""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import obspy

import mplcursors

import warnings

warnings.filterwarnings("ignore")


class HvsrProc:
    """
    This class is developed for processing and
    calculating HVSR from noise-based
    and earthquake-based time series data.

    """

    @staticmethod
    def _check_type(name, input_type):
        if not isinstance(input_type, str):
            msg = f"{name} must be a string indicating the file type, not {type(input_type)}."
            raise TypeError(msg)

        return input_type

    @staticmethod
    def add_mseed_tool(st):
        while True:
            # Plot the current stream
            st.plot()

            in_start_time = st[0].stats.starttime
            in_end_time = st[0].stats.endtime

            print("Current start time:", in_start_time)
            print("Current end time:", in_end_time)

            new_start_time = input("Enter the new start time (YYYY-MM-DDTHH:MM:SS): ")
            new_end_time = input("Enter the new end time (YYYY-MM-DDTHH:MM:SS): ")

            # Convert input times to UTCDateTime objects
            start_time = obspy.UTCDateTime(new_start_time)
            end_time = obspy.UTCDateTime(new_end_time)

            # Trim the stream
            st.trim(starttime=start_time, endtime=end_time)

            # Replot the trimmed stream
            st.plot()

            continue_trim = input("Do you want to continue trimming? (y/n): ")
            if continue_trim.lower() != 'y':
                break

        return st

    @staticmethod
    def proc_mseed_data(file_direc, h1_fn, h2_fn, v_fn, trim_flag=False, time_cut=300):

        h1 = obspy.read(os.path.join(file_direc, h1_fn))
        h2 = obspy.read(os.path.join(file_direc, h2_fn))
        v = obspy.read(os.path.join(file_direc, v_fn))

        h1.merge(method=1, interpolation_samples=-1)
        h2.merge(method=1, interpolation_samples=-1)
        v.merge(method=1, interpolation_samples=-1)

        if trim_flag:
            h1 = HvsrProc.add_mseed_tool(h1)
            h2 = HvsrProc.add_mseed_tool(h2)
            v = HvsrProc.add_mseed_tool(v)
        print(h1)

        dt = float(h1[0].stats.delta)

        h1 = h1[0].data
        h2 = h2[0].data
        v = v[0].data

        npts_min = min(len(h1), len(h2), len(v))

        time = np.dot(range(npts_min), dt)

        h1 = h1[:npts_min]
        h2 = h2[:npts_min]
        v = v[:npts_min]

        tcut_val = time_cut

        npts_cut = int(tcut_val / dt)

        h1 = h1[(npts_cut + 1):(npts_min - npts_cut)]
        h2 = h2[(npts_cut + 1):(npts_min - npts_cut)]
        v = v[(npts_cut + 1):(npts_min - npts_cut)]

        return h1, h2, v, dt, time

    @staticmethod
    def proc_txt_data(file_direc, h1_fn, h2_fn, v_fn, time_cut=300):

        h1 = pd.read_csv(os.path.join(file_direc, h1_fn), sep=' ', header=None, names=['time', 'vel'])
        h2 = pd.read_csv(os.path.join(file_direc, h2_fn), sep=' ', header=None, names=['time', 'vel'])
        v = pd.read_csv(os.path.join(file_direc, v_fn), sep=' ', header=None, names=['time', 'vel'])

        # truncate records to the smallest length
        npts_min = min(len(v), len(h1), len(h2))

        # Extract columns from dataframe
        time = h1['time'].values
        h1 = h1['vel'].values
        h2 = h2['vel'].values
        v = v['vel'].values

        time = time[:npts_min]
        h1 = h1[:npts_min]
        h2 = h2[:npts_min]
        v = v[:npts_min]

        # Specify time step
        dt = time[1] - time[0]

        tcut_val = time_cut

        npts_cut = int(tcut_val / dt)

        h1 = h1[(npts_cut + 1):npts_min - npts_cut]
        h2 = h2[(npts_cut + 1):npts_min - npts_cut]
        v = v[(npts_cut + 1):npts_min - npts_cut]

        return h1, h2, v, dt, time

    def __init__(self, h1, h2, v, directory='', output_dir=''):

        """

        Parameters
        ----------
        h1 : string
            string of h1 time series. Each row represents timeseries data
            from the first horizontal component.
        h2 :  string
            string of h2 time series. Each row represents timeseries data
            from the second horizontal component.
        v : string
            string of v time series. Each row represents timeseries data
            from the vertical component.
        directory: string
            directory where data is stored.
        output_dir : string
            directory where to save data.

        """

        self.h1 = HvsrProc._check_type("horizontal_1", h1)
        self.h2 = HvsrProc._check_type("horizontal_2", h2)
        self.v = HvsrProc._check_type("vertical", v)
        self.directory = HvsrProc._check_type("directory", directory)
        self.output_dir = HvsrProc._check_type("output directory", output_dir)

    def process_time_series(self, **kwargs):

        """

        Parameters
        ----------

        **kwargs
            time_cut : int
                Integer representing how much time to be cut from the beginning
                and end of the amplitude component of a  time series array.
            file_type :  int
                Indicates whether file is of mseed or txt file type. default = 1 (mseed)

        returns arrays of time series components

        """

        time_cut = kwargs.get('time_cut', 300)
        file_type = kwargs.get('file_type', 1)
        trim_flag = kwargs.get('trim_flag', False)

        if file_type == 1:
            h1, h2, v, self.dt, self.time = HvsrProc.proc_mseed_data(file_direc=self.directory, h1_fn=self.h1,
                                                                     h2_fn=self.h2, v_fn=self.v, trim_flag=trim_flag,
                                                                     time_cut=time_cut)

        elif file_type == 2:
            h1, h2, v, self.dt, self.time = HvsrProc.proc_txt_data(file_direc=self.directory, h1_fn=self.h1,
                                                                   h2_fn=self.h2, v_fn=self.v, time_cut=time_cut)
        else:
            raise ValueError("Invalid file type used. Use either 1 for mseed or 2 for text file")

        return h1, h2, v

    @staticmethod
    def create_directory(directory):
        """

        Parameters
        ----------
        directory : string
            A string representing the directory where needs to be created.

        returns directory

        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    @staticmethod
    def _len_guards(m):
        # This function is from scipy
        """Handle small or incorrect window lengths"""
        if int(m) != m or m < 0:
            raise ValueError('Window length M must be a non-negative integer')
        return m <= 1

    @staticmethod
    def _extend(m, sym):
        # This function is from scipy
        """Extend window by 1 sample if needed for DFT-even symmetry"""
        if not sym:
            return m + 1, True
        else:
            return m, False

    @staticmethod
    def _truncate(w, needed):
        # This function is from scipy
        """Truncate window by 1 sample if needed for DFT-even symmetry"""
        if needed:
            return w[:-1]
        else:
            return w

    @staticmethod
    def general_cosine(m, a, sym=True):
        # This function is from scipy
        if HvsrProc._len_guards(m):
            return np.ones(m)
        m, needs_trunc = HvsrProc._extend(m, sym)

        fac = np.linspace(-np.pi, np.pi, m)
        w = np.zeros(m)
        for k in range(len(a)):
            w += a[k] * np.cos(k * fac)
        return HvsrProc._truncate(w, needs_trunc)

    @staticmethod
    def general_hamming(m, alpha, sym=True):
        # This function is from scipy
        return HvsrProc.general_cosine(m, [alpha, 1. - alpha], sym)

    @staticmethod
    def hann(m, sym=True):
        # This function is from scipy
        return HvsrProc.general_hamming(m, 0.5, sym)

    @staticmethod
    def tukey(m, alpha=0.5, sym=True):
        # This function is from scipy

        if HvsrProc._len_guards(m):
            return np.ones(m)

        if alpha <= 0:
            return np.ones(m, 'd')

        elif alpha >= 1.0:
            return HvsrProc.hann(m, sym=sym)

        m, needs_trunc = HvsrProc._extend(m, sym)

        n = np.arange(0, m)
        width = int(np.floor(alpha * (m - 1) / 2.0))
        n1 = n[0:width + 1]
        n2 = n[width + 1:m - width - 1]
        n3 = n[m - width - 1:]

        w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (m - 1))))
        w2 = np.ones(n2.shape)
        w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (m - 1))))

        w = np.concatenate((w1, w2, w3))

        return HvsrProc._truncate(w, needs_trunc)

    @staticmethod
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

        defaults = {'detrend_type': 1, 't_front': 5, 't_end': 5}

        kwargs = {**defaults, **kwargs}

        detrend_type = kwargs['detrend_type']
        t_front = kwargs['t_front']
        t_end = kwargs['t_end']

        if detrend_type == 0:
            # Apply no detrend
            return ts
        elif detrend_type == 1:
            # Apply a weighted mean detrend
            win = HvsrProc.tukey(len(ts), alpha=((t_front + t_end) / 200))
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

    @staticmethod
    def tukey_window(ts, t_front=5, t_end=5, sym=True):

        """
        Applies Tukey window to the given time series.

        Parameters
        ---------

        ts: ndarray
            The input time series.
        t_front: int
            Percentage of data to be tapered at the beginning for Tukey window. Default is 5.
        t_end: int
            Percentage of data to be tapered at the end for Tukey window. Default is 5.
        sym: boolean
            generates a symmetric window.


        returns an array of the tapered time series.

        """

        win = HvsrProc.tukey(len(ts), alpha=((t_front + t_end) / 200), sym=sym)

        return ts * win

    @staticmethod
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
        order_zero_padding = kwargs.get('order_zero_padding', 4)

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
            hs = complex(real=1)  # hs = complex filter response at frequency f
            if order == 0:
                return ts, hs
            if np.isnan(fc):
                return ts, hs
            # if freq = 0 Hz
            ii = np.where(freq == 0)
            freq[ii] = 10 ** -6
            as_ = np.array([complex(val) for val in abs(freq / fc)])
            if npole < 0:
                as_ = 1 / as_  # if high-pass
            hs = as_ - np.exp(complex(imag=np.pi * (0.5 + (((2 * 1.) - 1.) / (2. * order)))))
            if order > 1:
                for i in range(2, order + 1):
                    hs *= (as_ - np.exp(complex(imag=np.pi * (0.5 + (((2. * i) - 1.) / (2. * order))))))
            hs = 1 / hs
            ts_fft[:n_nyq] *= hs
        else:
            # Apply acausal filter
            if npole < 0:
                filt = 1.0 / np.sqrt(1.0 + (np.divide(fc, freq, where=freq > 0)) ** (2.0 * order))
                filt[0] = 0
                ts_fft[:n_nyq] *= filt
            if npole > 1:
                filt = 1 / np.sqrt(1 + (np.divide(freq, fc, where=freq > 0)) ** (2.0 * order))
                filt[0] = 0
                ts_fft[:n_nyq] *= filt

        # Specify the response of the filter
        if is_causal:
            flt_resp = hs
        else:
            flt_resp = filt

        # Invert the fft to get the time series
        ts_padded = np.fft.irfft(ts_fft)
        ts = ts_padded[:npts]

        # Filtered FAS
        ts_flt_amp = np.abs(ts_fft)

        res = {'flt_ts': ts, 'flt_amp': ts_flt_amp, 'flt_resp': flt_resp, 'flt_fft': ts_fft}

        return ts, res

    @staticmethod
    def pre_proc(ts, **kwargs):

        """
        A function to apply pre-processing to time series data prior to processing.

        Parameters
        ----------

        ts : ndarray
             An array representing the time series data.

        **kwargs
            ts_dt: float
                Time step of ts
            detrend_type: int
                An integer indicating the type of detrending to be applied, default = 1.
                1 = Mean_Removal, 2 = Linear Removal, 6 = 5th Order Polynomial.
            taper_flag: Boolean
                 Indicates whether to apply tapering to the data.
            t_front:int
                Integer representing the percentage to taper on start of the time series
                data. Default = 5
            t_end: int
                Integer representing the percentage to taper on end of the time series
                data. Default = 5
            filter_flag: boolean
                Indicates whether to apply filtering to the data.
            fc : float
                 The filtering frequency to be applied.
            npole: int
               A value specifying low pass or high pass, a value less than 0
               applies to a high pass filter, default = -5.
            is_causal: boolean
               Indicates whether to apply a causal or acausal filter to the time series.
            order_zero_padding: int
                Indicates what order of zero padding to be applied.


        return processed time series data array

        """

        # Default parameter values
        ts_dt = kwargs.get('ts_dt', 0.005)
        detrend_type = kwargs.get('detrend_type', 1)
        taper_flag = kwargs.get('taper_flag', True)
        t_front = kwargs.get('t_front', 5)
        t_end = kwargs.get('t_end', 5)
        filter_flag = kwargs.get('filter_flag', False)
        fc = kwargs.get('fc', 0.01)
        npole = kwargs.get('npole', -5)
        is_causal = kwargs.get('is_causal', False)
        order_zero_padding = kwargs.get('order_zero_padding', 0)

        for i in range(len(ts)):
            # Detrend
            ts[i] = HvsrProc.detrend_ts(ts=ts[i], detrend_type=detrend_type, t_front=t_front, t_end=t_end)

            # Taper
            if taper_flag:
                ts[i] = HvsrProc.tukey_window(ts=ts[i], t_front=t_front, t_end=t_end)

            # Filter
            if filter_flag:
                ts[i], res = HvsrProc.apply_filter(ts=ts[i], fc=fc, dt=ts_dt, npole=npole, is_causal=is_causal,
                                                   order_zero_padding=order_zero_padding)

        return ts

    @staticmethod
    def sta_lta_calc(ts, **kwargs):

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

    def pre_process_noise_data(self, **kwargs):

        """
        Processes the time series data, by pre-processing of the raw time series data,
        then processes the data within windows.

        Parameters
        ----------

        **kwargs
            detrend_type: int
                An integer indicating the type of detrending to be applied, default = 1.
                0 = No detrend, 1 = Mean_Removal, 2 = Linear Removal, 6 = 5th Order Polynomial.
            taper_flag: Boolean
                 Indicates whether to apply tapering to the data.
            t_front: int
                Integer representing the percentage to taper on start of the time series
                data. Default = 5
            t_end: int
                Integer representing the percentage to taper on end of the time series
                data. Default = 5
            pre_filter_flag: Boolean
                Indicates whether to apply filtering to the data.
            pre_filter_t_front: int
            `   Integer representing the percentage to taper on start of the time series
                data. Default = 5
            pre_filter_t_end: int
                Integer representing the percentage to taper on end of the time series
                data. Default = 5
            pre_filter_hpass_fc : float
                 The high pass filtering frequency to be applied.
            pre_filter_npole_hp: int
               A value specifying high pass filter, a value less than 0
               applies to a high pass filter, default = -5.
            pre_filter_is_causal: Boolean
               Indicates whether to apply a causal or acausal filter to the time series.
            pre_filter_order_zero_padding: int
                Indicates what order of zero padding to be applied.
            pre_filter_lpass_fc : float
                 The low pass filtering frequency to be applied. default = None
            pre_filter_npole_lp: int
               A value specifying low pass, a value greater than 0
               applies to a low pass filter, default = None.
            is_noise: Boolean
               Indicates whether the data being processed is microtremor data, default = True.
               False indicates earthquake data.
            win_width: int
                Integer representing the size of the window that the time series data will be
                partitioned into discrete windows.
            overlapping: int
                Integer indicating the amount of overlap within each window.
            sta_lta_flag: Boolean
                Indicates whether a user wants to compute the short term (STA) or long term average (LTA) of each
                window of the time series.
            short_term_len: int
                Integer representing the short term length
            long_term_len: int
                Integer representing the long term length.
            sta_lta_moving_term: int
                Integer representing the sta/lta ratio moving term.
            filter_flag: Boolean
                Indicates whether to apply filtering to the data.
            hpass_fc: float
                The high pass filtering frequency to be applied.
            npole_hp: int
                A value specifying high pass filter, a value less than 0
               applies to a high pass filter, default = -5.
            lpass_fc: float
                The low pass filtering frequency to be applied
            npole_lp: int
                A value specifying low pass filter, a value greater than 0
               applies to a low pass filter, default = None.
            eqk_filepath: string
                The directory from which to extract earthquake data.
            is_causal: Boolean
                Indicates whether to apply acausal to causal filtering, default = False.
            order_zero_padding: int
                Indicates the order to which to apply zero padding to the data.

        returns windowed time series and the sta_lta for each component.

        """

        # Set default values for all parameters
        defaults = {'time_cut': 5, 'file_type': 1, 'trim_flag': False, 'detrend_type': 1, 'taper_flag': True,
                    't_front': 5, 't_end': 5, 'pre_filter_flag': True, 'pre_filter_t_front': 5, 'pre_filter_t_end': 5,
                    'pre_filter_hpass_fc': 0.042, 'pre_filter_npole_hp': -5, 'pre_filter_is_causal': False,
                    'pre_filter_order_zero_padding': 0, 'pre_filter_lpass_fc': None, 'pre_filter_npole_lp': None,
                    'is_noise': True, 'win_width': 300, 'overlapping': 0, 'sta_lta_flag': True, 'short_term_len': 1,
                    'long_term_len': 30, 'sta_lta_moving_term': 1, 'filter_flag': True, 'hpass_fc': 0.01,
                    'npole_hp': -5,
                    'lpass_fc': None, 'npole_lp': None, 'eqk_filepath': None, 'is_causal': False,
                    'order_zero_padding': 0}

        # Update default values with user-provided values
        kwargs = {**defaults, **kwargs}

        h1, h2, v = self.process_time_series(time_cut=kwargs['time_cut'], file_type=kwargs['file_type'],
                                             trim_flag=kwargs['trim_flag'])

        # Specify the variables to be used
        time = self.time
        dt = self.dt

        # Pre-process noise data
        if kwargs['pre_filter_flag']:
            # Apply detrend
            det_h1 = HvsrProc.detrend_ts(ts=h1, detrend_type=kwargs['detrend_type'],
                                         t_front=kwargs['pre_filter_t_front'],
                                         t_end=kwargs['pre_filter_t_end'])
            det_h2 = HvsrProc.detrend_ts(ts=h2, detrend_type=kwargs['detrend_type'],
                                         t_front=kwargs['pre_filter_t_front'],
                                         t_end=kwargs['pre_filter_t_end'])
            det_v = HvsrProc.detrend_ts(ts=v, detrend_type=kwargs['detrend_type'], t_front=kwargs['pre_filter_t_front'],
                                        t_end=kwargs['pre_filter_t_end'])

            # Apply tukey window
            h1 = HvsrProc.tukey_window(ts=det_h1, t_front=kwargs['pre_filter_t_front'],
                                       t_end=kwargs['pre_filter_t_end'])

            h2 = HvsrProc.tukey_window(ts=det_h2, t_front=kwargs['pre_filter_t_front'],
                                       t_end=kwargs['pre_filter_t_end'])

            v = HvsrProc.tukey_window(ts=det_v, t_front=kwargs['pre_filter_t_front'], t_end=kwargs['pre_filter_t_end'])

            # Apply filtering of the time series
            if kwargs['pre_filter_hpass_fc'] is not None:
                h1, res = HvsrProc.apply_filter(ts=h1, fc=kwargs['pre_filter_hpass_fc'], dt=dt,
                                                npole=kwargs['pre_filter_npole_hp'],
                                                is_causal=kwargs['pre_filter_is_causal'],
                                                order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                h2, res = HvsrProc.apply_filter(ts=h2, fc=kwargs['pre_filter_hpass_fc'], dt=dt,
                                                npole=kwargs['pre_filter_npole_hp'],
                                                is_causal=kwargs['pre_filter_is_causal'],
                                                order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                v, res = HvsrProc.apply_filter(ts=v, fc=kwargs['pre_filter_hpass_fc'], dt=dt,
                                               npole=kwargs['pre_filter_npole_hp'],
                                               is_causal=kwargs['pre_filter_is_causal'],
                                               order_zero_padding=kwargs['pre_filter_order_zero_padding'])

            if kwargs['pre_filter_lpass_fc'] is not None:
                h1, res = HvsrProc.apply_filter(ts=h1, fc=kwargs['pre_filter_lpass_fc'], dt=dt,
                                                npole=kwargs['pre_filter_npole_lp'],
                                                is_causal=kwargs['pre_filter_is_causal'],
                                                order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                h2, res = HvsrProc.apply_filter(ts=h2, fc=kwargs['pre_filter_lpass_fc'], dt=dt,
                                                npole=kwargs['pre_filter_npole_lp'],
                                                is_causal=kwargs['pre_filter_is_causal'],
                                                order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                v, res = HvsrProc.apply_filter(ts=v, fc=kwargs['pre_filter_lpass_fc'], dt=dt,
                                               npole=kwargs['pre_filter_npole_lp'],
                                               is_causal=kwargs['pre_filter_is_causal'],
                                               order_zero_padding=kwargs['pre_filter_order_zero_padding'])

        # split data into num_wins windows
        num_wins = int(np.floor(len(h1) * dt / kwargs['win_width']))
        npts_win = int(np.floor(kwargs['win_width'] / dt))
        npts_over = int(np.floor(kwargs['overlapping'] / dt))
        win_moving = npts_win - npts_over
        new_shape = (num_wins, win_moving)
        ts_wins = np.reshape(time[0:(win_moving * num_wins)], new_shape)
        h1_wins = np.reshape(h1[0:(win_moving * num_wins)], new_shape)
        h2_wins = np.reshape(h2[0:(win_moving * num_wins)], new_shape)
        v_wins = np.reshape(v[0:(win_moving * num_wins)], new_shape)

        h1_stalta, h2_stalta, v_stalta = [], [], []

        # Specify if the data is microtremor noise or earthquake noise
        if kwargs['is_noise']:
            # compute sta/lta
            if kwargs['sta_lta_flag']:
                short_term = int(np.floor(kwargs['short_term_len'] / dt))
                long_term = int(np.floor(kwargs['long_term_len'] / dt))
                sta_lta_moving = int(np.floor(kwargs['sta_lta_moving_term'] / dt))

                h1_stalta, h2_stalta, v_stalta = [], [], []
                for i in range(1, len(h1_wins)):
                    h1_stalta.append(HvsrProc.sta_lta_calc(ts=h1_wins[i], short_term=short_term, long_term=long_term,
                                                           moving_term=sta_lta_moving))
                    h2_stalta.append(HvsrProc.sta_lta_calc(ts=h2_wins[i], short_term=short_term, long_term=long_term,
                                                           moving_term=sta_lta_moving))
                    v_stalta.append(HvsrProc.sta_lta_calc(ts=v_wins[i], short_term=short_term, long_term=long_term,
                                                          moving_term=sta_lta_moving))

            # Post-processing
            if kwargs['filter_flag']:  # Apply filter
                if kwargs['hpass_fc'] is not None or kwargs['lpass_fc'] is not None:
                    if kwargs['hpass_fc'] is not None:  # Apply hpass
                        h1_wins = HvsrProc.pre_proc(ts=h1_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                    taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                    t_end=kwargs['t_end'],
                                                    filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                    npole=kwargs['npole_hp'],
                                                    is_causal=kwargs['is_causal'],
                                                    order_zero_padding=kwargs['order_zero_padding'])

                        h2_wins = HvsrProc.pre_proc(ts=h2_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                    taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                    t_end=kwargs['t_end'],
                                                    filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                    npole=kwargs['npole_hp'],
                                                    is_causal=kwargs['is_causal'],
                                                    order_zero_padding=kwargs['order_zero_padding'])

                        v_wins = HvsrProc.pre_proc(ts=v_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                   taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                   t_end=kwargs['t_end'],
                                                   filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                   npole=kwargs['npole_hp'],
                                                   is_causal=kwargs['is_causal'],
                                                   order_zero_padding=kwargs['order_zero_padding'])

                    if kwargs['lpass_fc'] is not None:  # Apply lpass
                        h1_wins = HvsrProc.pre_proc(ts=h1_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                    taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                    t_end=kwargs['t_end'],
                                                    filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                                    npole=kwargs['npole_lp'],
                                                    is_causal=kwargs['is_causal'],
                                                    order_zero_padding=kwargs['order_zero_padding'])

                        h2_wins = HvsrProc.pre_proc(ts=h2_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                    taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                    t_end=kwargs['t_end'],
                                                    filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                                    npole=kwargs['npole_lp'],
                                                    is_causal=kwargs['is_causal'],
                                                    order_zero_padding=kwargs['order_zero_padding'])

                        v_wins = HvsrProc.pre_proc(ts=v_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                   taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                   t_end=kwargs['t_end'],
                                                   filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                                   npole=kwargs['npole_lp'],
                                                   is_causal=kwargs['is_causal'],
                                                   order_zero_padding=kwargs['order_zero_padding'])

                else:
                    pass
            else:  # Do not apply filter
                pass
        print("Pre-processing noise data is DONE!")

        # Assemble corrected earthquake strong motions
        if not kwargs['is_noise']:
            eqk_data = np.genfromtxt(kwargs['eqk_filepath'], delimiter=',')
            num_wins = int(eqk_data.shape[1] / 4)
            h1_wins = [None] * num_wins
            h2_wins = [None] * num_wins
            v_wins = [None] * num_wins
            for i in range(num_wins):
                h1_wins[i] = eqk_data[:, 1 + (i - 1) * 4][~np.isnan(eqk_data[:, 1 + (i - 1) * 4])]
                h2_wins[i] = eqk_data[:, 2 + (i - 1) * 4][~np.isnan(eqk_data[:, 2 + (i - 1) * 4])]
                v_wins[i] = eqk_data[:, i * 4][~np.isnan(eqk_data[:, i * 4])]
            print("Assembling earthquake strong motion is DONE!")

        return time, h1, h2, v, ts_wins, h1_wins, h2_wins, v_wins, h1_stalta, h2_stalta, v_stalta

    @staticmethod
    def shift_update(h2_wins, v_wins, shift_1, shift_2):

        """
        function which performs a shift after a time window is selected.

        Parameters
        -----------

        h2_wins: ndarray
            An array that is separated based on the window length.
        v_wins: ndarray
            An array that is separated based on the window length.
        shift_1: Int
            Indicates the amount to be shifted.
        shift_2: Int
            Indicates the amount to be shifted.

        returns the shifted windows of the horizontal and vertical components.

        """

        h2_wins_sh = np.array([x - shift_1 for x in h2_wins])
        v_wins_sh = np.array([x - shift_2 for x in v_wins])

        return h2_wins_sh, v_wins_sh

    def plot_selected(self, h1_wins, h2_wins_sh, v_wins_sh, idx_select, cols, sta_lta_flag=False,
                      h1_stalta=None, h2_stalta=None, v_stalta=None):

        """

        function which plots the selected time series windows.

        Parameters
        ----------
        h1_wins: ndarray
            Array of windowed time series data for the first horizontal component.
        h2_wins_sh: ndarray
            Array of shifted windowed time series data for the second horizontal component.
        v_wins_sh: ndarray
            Array of the shifted windowed time series data for the vertical component.
        idx_select: list
             List of the indices that were selected to plot the data. The indices represent
             the windows of the time series.
        cols: list
            List of the colors to be used for plotting the time series data.
        sta_lta_flag: boolean
            Indicates whether the sta_lta are plotted over the time series windows.
        h1_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows for h1 component.
        h2_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows for h2 component.
        v_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows for v component.

        returns plotted time series data

        """
        plt.figure(figsize=(8, 6))

        for i, (i_plot, h1_win, h2_win, v_win) in enumerate(zip(idx_select, h1_wins, h2_wins_sh, v_wins_sh)):
            t_seq = np.abs(np.arange(0, len(h1_wins[i_plot])) * self.dt + i_plot * len(h1_wins[i_plot]) * self.dt)
            idx = np.arange(0, len(t_seq))
            plt.subplot(311)
            plt.title('Horizontal - 1')
            plt.plot(t_seq[idx], h1_wins[i_plot][idx], color=cols[i])
            plt.ylabel('Counts')
            plt.legend(['Selected windows'], loc='upper center')
            range_h1 = max(h1_wins[i_plot]) - min(h1_wins[i_plot])
            plt.text(np.mean(t_seq[idx]), min(h1_wins[i_plot][idx]) + range_h1 * 0.5,
                     i_plot, color='red')
            plt.subplot(312)
            plt.title('Horizontal - 2')
            plt.plot(t_seq[idx], h2_wins_sh[i_plot][idx], color=cols[i])
            plt.ylabel('Counts')
            range_h2 = max(h2_wins_sh[i_plot]) - min(h2_wins_sh[i_plot])
            plt.text(np.mean(t_seq[idx]), min(h2_wins_sh[i_plot][idx]) + range_h2 * 0.5,
                     i_plot, color='red')
            plt.subplot(313)
            plt.title('Vertical')
            plt.plot(t_seq[idx], v_wins_sh[i_plot][idx], color=cols[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Counts')
            range_v = max(v_wins_sh[i_plot][idx]) - min(v_wins_sh[i_plot][idx])
            plt.text(np.mean(t_seq[idx]), min(v_wins_sh[i_plot][idx]) + range_v * 0.5,
                     i_plot, color='red')

            if sta_lta_flag:
                if i_plot < len(h1_stalta):
                    plt.subplot(311)
                    range_h1 = max(h1_win) - min(h1_win)
                    plt.text(np.mean(t_seq[idx]), min(h1_wins[i_plot][idx]) + range_h1 * 0.2,
                             round(min(h1_stalta[i_plot]), 1), color='black')
                    plt.text(np.mean(t_seq[idx]), max(h1_wins[i_plot][idx]) - range_h1 * 0.2,
                             round(max(h1_stalta[i_plot]), 1), color='black')

                if i_plot < len(h2_stalta):
                    plt.subplot(312)
                    range_h2 = max(h2_win) - min(h2_win)
                    plt.text(np.mean(t_seq[idx]), min(h2_wins_sh[i_plot][idx]) + range_h2 * 0.2,
                             round(min(h2_stalta[i_plot]), 1), color='black')
                    plt.text(np.mean(t_seq[idx]), max(h2_wins_sh[i_plot][idx]) - range_h2 * 0.2,
                             round(max(h2_stalta[i_plot]), 1), color='black')

                if i_plot < len(v_stalta):
                    plt.subplot(313)
                    range_v = max(v_win) - min(v_win)
                    plt.text(np.mean(t_seq[idx]), min(v_wins_sh[i_plot][idx]) + range_v * 0.2,
                             round(min(v_stalta[i_plot]), 1), color='black')
                    plt.text(np.mean(t_seq[idx]), max(v_wins_sh[i_plot][idx]) - range_v * 0.2,
                             round(max(v_stalta[i_plot]), 1), color='black')
        plt.tight_layout()

    def select_windows(self, ts_wins, h1_wins, h2_wins, v_wins, idx_select, cols=None,
                       sta_lta_flag=False, h1_stalta=None, h2_stalta=None, v_stalta=None):

        """

        function which plots the  time series windows.

        Parameters
        ----------
        ts_wins: ndarray
            Array of windowed time series
        h1_wins: ndarray
            Array of windowed time series data for the first horizontal component.
        h2_wins: ndarray
            Array of windowed time series data for the second horizontal component.
        v_wins: ndarray
            Array of the windowed time series data for the vertical component.
        idx_select: list
             List of the indices that were selected to plot the data. The indices represent
             the windows of the time series.
        cols: list
            List of the colors to be used for plotting the time series data.
        sta_lta_flag: boolean
            Indicates whether the sta_lta are plotted over the time series windows.
        h1_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows of h1 component.
        h2_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows of h2 component.
        v_stalta: ndarray
            Array containing the sta/lta ratios for the time series windows of v component.

        returns plotted time series data

        """

        plt.figure(figsize=(8, 6))

        while True:
            plt.figure(figsize=(8, 6))

            for i, (i_plot, tt, h1_win, h2_win, v_win) in enumerate(zip(idx_select, ts_wins, h1_wins, h2_wins, v_wins)):
                t_seq = np.arange(1, len(h1_wins[i_plot]) + 1) * self.dt + (i_plot * len(h1_wins[i_plot])) * self.dt
                idx = np.arange(0, len(t_seq))
                plt.subplot(311)
                plt.title('Horizontal - 1')
                plt.plot(t_seq[idx], h1_wins[i_plot][idx], color=cols[i])
                plt.legend(['Please select time series for removal.'], loc='upper center')
                plt.ylabel('Counts')
                range_h1 = max(h1_wins[i_plot]) - min(h1_wins[i_plot])
                plt.text(np.mean(t_seq[idx]), min(h1_wins[i_plot][idx]) + range_h1 * 0.5,
                         i_plot, color='red')
                plt.subplot(312)
                plt.title('Horizontal - 2')
                plt.plot(t_seq[idx], h2_wins[i_plot][idx], color=cols[i])
                plt.ylabel('Counts')
                range_h2 = max(h2_wins[i_plot]) - min(h2_wins[i_plot])
                plt.text(np.mean(t_seq[idx]), min(h2_wins[i_plot][idx]) + range_h2 * 0.5,
                         i_plot, color='red')
                plt.subplot(313)
                plt.title('Vertical')
                plt.plot(t_seq[idx], v_wins[i_plot][idx], color=cols[i])
                plt.xlabel('Time (s)')
                plt.ylabel('Counts')
                range_v = max(v_wins[i_plot]) - min(v_wins[i_plot])
                plt.text(np.mean(t_seq[idx]), min(v_wins[i_plot][idx]) + range_v * 0.5,
                         i_plot, color='red')

                if sta_lta_flag:
                    if i_plot < len(h1_stalta):
                        plt.subplot(311)
                        range_h1 = max(h1_win) - min(h1_win)
                        plt.text(np.mean(t_seq[idx]), min(h1_wins[i_plot][idx]) + range_h1 * 0.2,
                                 round(min(h1_stalta[i]), 1))
                        plt.text(np.mean(t_seq[idx]), max(h1_wins[i_plot][idx]) - range_h1 * 0.2,
                                 round(max(h1_stalta[i_plot]), 1), color='black')

                    if i_plot < len(h2_stalta):
                        plt.subplot(312)
                        range_h2 = max(h2_win) - min(h2_win)
                        plt.text(np.mean(t_seq[idx]), min(h2_wins[i_plot][idx]) + range_h2 * 0.2,
                                 round(min(h2_stalta[i]), 1), color='black')
                        plt.text(np.mean(t_seq[idx]), max(h2_wins[i_plot][idx]) - range_h2 * 0.2,
                                 round(max(h2_stalta[i_plot]), 1), color='black')

                    if i_plot < len(v_stalta):
                        plt.subplot(313)
                        range_v = max(v_win) - min(v_win)
                        plt.text(np.mean(t_seq[idx]), min(v_wins[i_plot][idx]) + range_v * 0.2,
                                 round(min(v_stalta[i]), 1), color='black')
                        plt.text(np.mean(t_seq[idx]), max(v_wins[i_plot][idx]) - range_v * 0.2,
                                 round(max(v_stalta[i_plot]), 1), color='black')

            plt.tight_layout()

            plt.show()

            p_index = input("Enter the index of the window to remove (Press q to quit): ")
            if p_index == 'q':
                break

            p_index = int(p_index)

            # Remove the selected window index
            idx_select.remove(p_index)

            # Plot the updated selected windows
            plt.clf()
            h2_wins_sh, v_wins_sh = HvsrProc.shift_update(h2_wins, v_wins, 0, 0)  # Assuming no shift
            self.plot_selected(h1_wins, h2_wins_sh, v_wins_sh, idx_select=idx_select, cols=cols,
                               sta_lta_flag=sta_lta_flag, h1_stalta=h1_stalta, h2_stalta=h2_stalta, v_stalta=v_stalta)

            plt.show()

        plt.close()

        return idx_select

    def td_plt_select(self, ts_wins, h1_wins, h2_wins, v_wins, sta_lta_flag=False, h1_stalta=None,
                      h2_stalta=None, v_stalta=None):

        """

        function which plots the time series windows for the main function hv_proc.

        Parameters
        ----------
        ts_wins: ndarray
            Array of time series data.
        h1_wins: ndarray
            Array of windowed time series data for the first horizontal component.
        h2_wins: ndarray
            Array of windowed time series data for the second horizontal component.
        v_wins: ndarray
            Array of the windowed time series data for the vertical component.
        sta_lta_flag: boolean
            Indicates whether the sta_lta are plotted over the time series windows.
        h1_stalta: ndarray
            Array containing the sta/lta ratios for the h1 time series windows.
        h2_stalta: ndarray
            Array containing the sta/lta ratios for the h2 time series windows.
        v_stalta: ndarray
            Array containing the sta/lta ratios for the v time series windows.

        returns plotted time series data

        """

        num_wins = len(h1_wins)

        cols = ['yellow', 'lightblue'] * num_wins

        idx_select = list(range(num_wins))

        idx_select = self.select_windows(ts_wins=ts_wins, h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins,
                                         idx_select=idx_select, cols=cols, sta_lta_flag=sta_lta_flag,
                                         h1_stalta=h1_stalta, h2_stalta=h2_stalta, v_stalta=v_stalta)

        return idx_select

    @staticmethod
    def ang_pga_rotd50_calc(fft1, fft2, freq, num_wins=12):

        """

        function computes the rotD50 between two components.

        Parameters
        ----------

        fft1: ndarray
            Array of frequency data for the first horizontal component.
        fft2: ndarray
            Array of frequency data for the second horizontal component.
        freq:ndarray
            Array of frequency values
        num_wins:int
            integer representing the number of windows

        returns RotD50 of the two components.

        """
        angle_rad = np.linspace(np.zeros((len(freq), num_wins)), np.full((len(freq), num_wins), 2.0 * np.pi), 180).T
        frot_motions = fft1[:, :, np.newaxis] * np.cos(angle_rad) + fft2[:, :, np.newaxis] * np.sin(angle_rad)
        rot_d50 = np.median(np.abs(frot_motions), axis=2)

        return rot_d50

    @staticmethod
    def smoothed_fas_numpy(fas, f, fc, b=40):
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
        b: float
            Smoothing parameter, default = 40.

        returns a smoothed FAS using KO smoothing.
        """
        w = 1.0 / (10.0 ** (3.0 / b))
        filter_ko = ((f > w * fc[:, np.newaxis]) & (f < fc[:, np.newaxis] / w) & (f > 0) & (f != fc[:, np.newaxis]))
        weights = np.zeros((len(fc), len(f)))
        weights[filter_ko] = (np.abs(np.sin(b * np.log10((f / fc[:, np.newaxis])[filter_ko])) / (
                    b * np.log10((f / fc[:, np.newaxis])[filter_ko])))) ** 4.0
        weights[f == fc[:, np.newaxis]] = 1.0
        smooth_fas = np.sum(weights * fas[:, np.newaxis], axis=2) / np.sum(weights, axis=1)
        return smooth_fas

    @staticmethod
    def parzen_smooth(f, amp, fc, b=1.5):
        """
        Function applies parzen smoothing to a fourier amplitude spectra(FAS)

        Parameters
        ----------

        f: ndarray
            Frequency array of the fft.
        amp: ndarray
            Amplitude of the fft.
        fc: ndarray
            resampled frequency in which to apply smoothing
        b: float
            Smoothing parameter, default = 1.5.

        returns a smoothed FAS using parzen smoothing
        """

        u = 151 * b / 280
        temp = np.pi * u * (f - fc[:, np.newaxis]) / 2
        filter_parzen = ((f > fc[:, np.newaxis] / b) & (f < fc[:, np.newaxis] * b) & (f != fc[:, np.newaxis]))
        weights = np.zeros((len(fc), len(f)))
        weights[filter_parzen] = np.power(np.sin(temp[filter_parzen]) / temp[filter_parzen], 4) * 3 / 4 * u
        weights[f == fc[:, np.newaxis]] = 1.0
        num = weights * amp[:, np.newaxis]
        smoothed = np.sum(num, axis=2) / np.sum(weights, axis=1)

        return smoothed

    @staticmethod
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
            smooth_fas = HvsrProc.smoothed_fas_numpy(np.abs(fas), freq, fc, ko_smooth_b)

        # Apply Parzen Smoothing
        if parzen_flag:
            smooth_fas = HvsrProc.parzen_smooth(freq, np.abs(fas), fc, parzen_bwidth)

        return smooth_fas

    @staticmethod
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

    @staticmethod
    def hvsr_win_calc(h1_wins, h2_wins, v_wins, **kwargs):

        """

        function which computes the Horizontal-to-Vertical Spectral Ratio (HVSR)
        from each individual windowed time series.

        Parameters
        ----------

        h1_wins: ndarray
            Array of windowed time series data for the first horizontal component.
        h2_wins: ndarray
            Array of windowed time series data for the second horizontal component.
        v_wins: ndarray
            Array of windowed time series data for the vertical component.

        **kwargs:

            ts_dt: float
                Time step of ts
            ko_smooth_b: int
                bandwidth value to which apply Konno and Ohmachi Smoothing, default = 40.
            ko_smooth_flag: boolean
                Indicates whether to apply Konno and Ohmachi Smoothing.
            parzen_flag: boolean
                Indicates whether to apply to parzen smoothing.
            parzen_bwidth: float
                bandwidth value to which apply parzen smoothing.
            horizontal_comb: string
                Indicates the type of combination to be done to the horizontal components, default = 'ps_RotD50'.There
                are three type of combinations, ps_RotD50 which computes the median of the azimuth of the two
                components, squared_average which computes the squared average between the two components, and
                geometric_mean which computes the geometric mean of the two components.
            freq_hv_mean: ndarray
                Array to resample the frequencies.
            polar_curves_flag: boolean
                Indicates whether to compute the polar curve of the mean HSVR curve
            freq_polar: ndarray
                Array to resample the polar curve frequencies
            deg_increment: int
                Integer indicating the increment in which the polar curve is to be computed.



        returns a list of the FAS values for each window, the HVSR curve for each window,
        and the Polar Ratio for each azimuth.

        """

        defaults = {'ts_dt': 0.005, 'ko_smooth_b': 40, 'ko_smooth_flag': True,
                    'parzen_flag': False, 'parzen_bwidth': 1.5, 'horizontal_comb': 'ps_RotD50',
                    'freq_hv_mean': None, 'polar_curves_flag': False, 'freq_polar': None,
                    'deg_increment': 10, 'sjb_avg': False}

        kwargs = {**defaults, **kwargs}

        res = {}

        h1_sub = h1_wins
        h2_sub = h2_wins
        v_sub = v_wins

        if kwargs['horizontal_comb'] == 'ps_RotD50':
            # Implementation for 'ps_RotD50'
            fas_h1 = np.fft.rfft(h1_sub)
            fas_h2 = np.fft.rfft(h2_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), kwargs['ts_dt'])
            fas_h = HvsrProc.ang_pga_rotd50_calc(fft1=fas_h1, fft2=fas_h2, freq=freq, num_wins=len(h1_wins))
        elif kwargs['horizontal_comb'] == 'squared_average':
            # Implementation for 'squared_average'
            fas_h1 = np.fft.rfft(h1_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), kwargs['ts_dt'])
            fas_h2 = np.fft.rfft(h2_sub)
            fas_h = np.sqrt((np.abs(fas_h1) ** 2 + np.abs(fas_h2) ** 2) / 2)
        elif kwargs['horizontal_comb'] == 'geometric_mean':
            # Implementation for 'geometric_mean'
            fas_h1 = np.fft.rfft(h1_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), kwargs['ts_dt'])
            fas_h2 = np.fft.rfft(h2_sub)
            fas_h = np.sqrt(np.abs(fas_h1) * np.abs(fas_h2))
        else:
            raise KeyError(
                'Horizontal combination does not exist. Choose ps_RotD50, squared_average, or geometric_mean')

        # Compute the vertical FAS
        fas_v = np.fft.rfft(v_sub)

        # Smooth both the Horizontal and the Vertical FAS
        h_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_hv_mean'], fas=fas_h,
                                       freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                       parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                       parzen_bwidth=kwargs['parzen_bwidth'])

        v_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_hv_mean'], fas=fas_v,
                                       freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                       parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                       parzen_bwidth=kwargs['parzen_bwidth'])

        # Compute HVSR
        hv_ratio = np.abs(np.divide(h_smooth, v_smooth, out=np.zeros_like(h_smooth), where=v_smooth != 0))

        # Compute the smoothed horizontal components
        fas_h1 = np.fft.rfft(h1_sub)
        fas_h2 = np.fft.rfft(h2_sub)

        h1_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_hv_mean'], fas=fas_h1,
                                        freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                        parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                        parzen_bwidth=kwargs['parzen_bwidth'])

        h2_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_hv_mean'], fas=fas_h2,
                                        freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                        parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                        parzen_bwidth=kwargs['parzen_bwidth'])

        res['hv_ratio'] = hv_ratio
        res['h1_smooth'] = h1_smooth
        res['h2_smooth'] = h2_smooth
        res['v_smooth'] = v_smooth

        if kwargs['sjb_avg']:
            avg_hsmooth = np.median(h_smooth, axis=0)
            avg_vsmooth = np.median(v_smooth, axis=0)
            sjb_hv_ratio = np.divide(avg_hsmooth, avg_vsmooth)

            res['sjb_hv_ratio'] = sjb_hv_ratio

        # Polar curves
        if kwargs['polar_curves_flag']:

            polar_degs = np.arange(0, 180, kwargs['deg_increment'])
            polar_hv_ratio = np.zeros((len(kwargs['freq_polar']), len(polar_degs)))
            h1_fft = np.fft.rfft(h1_sub)
            h2_fft = np.fft.rfft(h2_sub)

            fas_v = np.fft.rfft(v_sub)
            freq = np.fft.rfftfreq(len(v_sub[0]), kwargs['ts_dt'])
            v_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_polar'], fas=fas_v,
                                           freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                           parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                           parzen_bwidth=kwargs['parzen_bwidth'])

            for i in range(len(polar_degs)):
                angle_idx = polar_degs[i]
                fas_h = h1_fft * np.cos(np.radians(angle_idx)) + h2_fft * np.sin(np.radians(angle_idx))

                h_smooth = HvsrProc.smooth_fas(fc=kwargs['freq_polar'], fas=fas_h,
                                               freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                                               parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                                               parzen_bwidth=kwargs['parzen_bwidth'])

                hv_ratio = np.abs(np.divide(h_smooth, v_smooth, out=np.zeros_like(h_smooth), where=v_smooth != 0))

                hv_ratio_reshaped = np.resize(hv_ratio, len(kwargs['freq_polar']))

                polar_hv_ratio[:, i] = hv_ratio_reshaped

            res['polar_hv_ratio'] = polar_hv_ratio

        return res

    @staticmethod
    def plot_hvsr(freq, hvsr_mat, idx_select, hvsr_mean, hvsr_sd, hvsr_sd1,
                  h1_mat, h2_mat, v_mat, h1_mat_mean, h2_mat_mean, v_mat_mean,
                  robust_est=False, sjb_avg=True, distribution='normal'):

        """

        function which plots the selected Horizontal-to-Vertical Spectral Ratio (HVSR)
        from each individual windowed time series.

        Parameters
        ----------

        freq: ndarray
            Frequency to plot the HVSR curves.
        hvsr_mat: ndarray
            Array of HVSR curves.
        idx_select: list
            List of the indices selected.
        hvsr_mean: ndarray
            Array of the mean HVSR curve.
        hvsr_sd: ndarray
            Array of the standard deviation of the HVSR curve.
        hvsr_sd1: ndarray
            Array of the standard deviation of the HVSR curve using robust_est.
        h1_mat: ndarray
            Array of the first horizontal component FAS.
        h2_mat: ndarray
            Array of the second horizontal component FAS.
        v_mat: ndarray
            Array of the vertical component FAS.
        h1_mat_mean: ndarray
            Array of the mean first horizontal component FAS.
        h2_mat_mean: ndarray
            Array of the mean second horizontal component FAS.
        v_mat_mean: ndarray
            Array of the mean vertical component FAS.
        robust_est: boolean
            Indicates whether to apply the robust estimate.
        sjb_avg: boolean
            Specifies whether to use a different type of averaging. This method takes the average of the
            smoothed FAS curves and takes the ratio of the horizontal and vertical components to get HVSR.
        distribution: string
            Indicates the type of distribution to apply.



        returns The plotted selected FAS and HVSR curves as well as their means.

        """

        while True:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            for i_plot in idx_select:
                plt.loglog(freq, h1_mat[i_plot], '-', color='pink')
                plt.loglog(freq, h2_mat[i_plot], '-', color='lightgreen')
                plt.loglog(freq, v_mat[i_plot], '-', color='lightblue')

            plt.xlabel('Frequency (Hz)')
            plt.ylabel('FAS')
            plt.title('Selected FAS Curves')

            colors = ['red', 'green', 'blue']
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
            labels = ['H1', 'H2', 'V']

            if robust_est:
                plt.loglog(freq, h1_mat_mean, color='red', linewidth=2)
                plt.loglog(freq, h2_mat_mean, color='green', linewidth=2)
                plt.loglog(freq, v_mat_mean, color='blue', linewidth=2)
            else:
                plt.loglog(freq, h1_mat_mean, color='red', linewidth=2)
                plt.loglog(freq, h2_mat_mean, color='green', linewidth=2)
                plt.loglog(freq, v_mat_mean, color='blue', linewidth=2)

            plt.xlim(0.01, 50)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('FAS')
            plt.legend(lines, labels, loc='upper right')

            plt.subplot(122)
            for i, i_plot in enumerate(idx_select):
                if i < 12:
                    plt.plot(freq, hvsr_mat[i_plot], '-', alpha=0.75, label=f'Index: {i_plot}')
                else:
                    plt.plot(freq, hvsr_mat[i_plot], '--', alpha=0.75, label=f'Index: {i_plot}')
            plt.semilogx(freq, hvsr_mean, color='black', linewidth=4)
            plt.semilogx(freq, hvsr_mean - hvsr_sd, color='red', linewidth=2, linestyle='--')
            plt.semilogx(freq, hvsr_mean + hvsr_sd, color='red', linewidth=2, linestyle='--')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('HVSR')
            if np.any(np.isnan(hvsr_mean)) or np.any(np.isinf(hvsr_mean)):
                hvsr_mean = np.nan_to_num(hvsr_mean)
            plt.ylim(0, np.abs(np.max(hvsr_mean)) * 3)
            plt.xlim(0.01, 50)
            plt.title('Selected HVSR Curves')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), borderaxespad=0., fontsize='x-small')

            plt.tight_layout()
            mplcursors.cursor(hover=True)
            plt.show()

            option = input("Enter the index of the curve to remove (press 'q' to quit): ")
            if option.lower() == 'q':
                break
            else:
                remove_index = int(option)
                if remove_index in idx_select:
                    idx_select.remove(remove_index)

                    # Recalculate mean and standard deviation after removal
                    if sjb_avg:
                        h1_mat_mean = np.mean(h1_mat[idx_select], axis=0)
                        h2_mat_mean = np.mean(h2_mat[idx_select], axis=0)
                        v_mat_mean = np.mean(v_mat[idx_select], axis=0)
                        hvsr_sd1 = np.percentile(hvsr_mat[idx_select], 75, axis=0) - np.percentile(hvsr_mat[idx_select],
                                                                                                   25, axis=0)
                        if distribution == 'normal':
                            hvsr_mean = hvsr_mat['sjb_hv_ratio']
                            hvsr_sd = np.std(hvsr_mat[idx_select], axis=0)
                        elif distribution == 'log_normal':
                            hvsr_mat[hvsr_mat <= 0] = 1e-5
                            hvsr_mean = np.exp(np.log(hvsr_mat['sjb_hv_ratio']))
                            hvsr_sd = np.std(np.log(hvsr_mat[idx_select]), axis=0)
                    elif robust_est:
                        h1_mat_mean = np.median(h1_mat[idx_select], axis=0)
                        h2_mat_mean = np.median(h2_mat[idx_select], axis=0)
                        v_mat_mean = np.median(v_mat[idx_select], axis=0)
                        hvsr_mean = np.median(hvsr_mat[idx_select], axis=0)
                        hvsr_sd = np.std(hvsr_mat[idx_select], axis=0)
                        hvsr_sd1 = np.percentile(hvsr_mat[idx_select], 75, axis=0) - np.percentile(
                            hvsr_mat[idx_select], 25, axis=0)
                    else:
                        h1_mat_mean = np.median(np.abs(h1_mat[idx_select]), axis=0)
                        h2_mat_mean = np.median(np.abs(h2_mat[idx_select]), axis=0)
                        v_mat_mean = np.median(np.abs(v_mat[idx_select]), axis=0)
                        if distribution == 'normal':
                            hvsr_mean = np.median(hvsr_mat[idx_select], axis=0)
                            hvsr_sd = np.std(hvsr_mat[idx_select], axis=0)
                        elif distribution == 'log_normal':
                            hvsr_mat[hvsr_mat <= 0] = 1e-5
                            hvsr_mean = np.exp(np.median(np.log(hvsr_mat[idx_select]), axis=0))
                            hvsr_sd = np.std(np.log(hvsr_mat[idx_select]), axis=0)

        return idx_select, hvsr_mean, hvsr_sd, hvsr_sd1, h1_mat, h2_mat, v_mat, h1_mat_mean, h2_mat_mean, v_mat_mean

    @staticmethod
    def fd_plt_select(hvsr_list, robust_est=False, freq_hv_mean=None,
                      distribution='normal', plot_hvsr=True, sjb_avg=True):

        """

        function which plots the Horizontal-to-Vertical Spectral Ratio (HVSR)
        from each individual windowed time series.

        Parameters
        ----------

        hvsr_list: list
            Containing the hvsr and FAS for each window from the function hvsr_win_calc.
        robust_est: boolean
            Indicates whether to apply the robust estimate.
        freq_hv_mean:ndarray
            Array to plot the HVSR curves.
        distribution: string
            Indicates the type of distribution to apply.
        plot_hvsr: boolean
            Specifies whether to plot the FAS and HVSR for window selection, default = True.
        sjb_avg: boolean
            Specifies whether to use a different type of averaging. This method takes the average of the
            smoothed FAS curves and takes the ratio of the horizontal and vertical components to get HVSR.

        returns The indices of HVSR curves selected in the frequency-domain and their
        associated mean, standard deviation, and mean FAS.

        """

        hvsr_mat = hvsr_list['hv_ratio']
        h1_mat = hvsr_list['h1_smooth']
        h2_mat = hvsr_list['h2_smooth']
        v_mat = hvsr_list['v_smooth']

        idx_select = list(range(hvsr_mat.shape[0]))

        # Calculate mean and standard deviation of HVSR and FAS
        if sjb_avg:
            h1_mat_mean = np.mean(h1_mat, axis=0)
            h2_mat_mean = np.mean(h2_mat, axis=0)
            v_mat_mean = np.mean(v_mat, axis=0)
            hvsr_sd1 = np.percentile(hvsr_mat, 75, axis=0) - np.percentile(hvsr_mat, 25, axis=0)
            if distribution == 'normal':
                hvsr_mean = hvsr_list['sjb_hv_ratio']
                hvsr_sd = np.std(hvsr_mat, axis=0)
            elif distribution == 'log_normal':
                hvsr_mat[hvsr_mat <= 0] = 1e-5
                hvsr_mean = np.exp(np.log(hvsr_list['sjb_hv_ratio']))
                hvsr_sd = np.std(np.log(hvsr_mat), axis=0)
        elif robust_est:
            hvsr_mean = np.median(hvsr_mat, axis=0)
            hvsr_sd = np.std(hvsr_mat, axis=0)
            hvsr_sd1 = np.percentile(hvsr_mat, 75, axis=0) - np.percentile(hvsr_mat, 25, axis=0)
            h1_mat_mean = np.median(h1_mat, axis=0)
            h2_mat_mean = np.median(h2_mat, axis=0)
            v_mat_mean = np.median(v_mat, axis=0)
        else:
            h1_mat_mean = np.median(h1_mat, axis=0)
            h2_mat_mean = np.median(h2_mat, axis=0)
            v_mat_mean = np.median(v_mat, axis=0)
            hvsr_sd1 = np.percentile(hvsr_mat, 75, axis=0) - np.percentile(hvsr_mat, 25, axis=0)
            if distribution == 'normal':
                hvsr_mean = np.median(hvsr_mat, axis=0)
                hvsr_sd = np.std(hvsr_mat, axis=0)
            elif distribution == 'log_normal':
                hvsr_mat[hvsr_mat <= 0] = 1e-5
                hvsr_mean = np.exp(np.median(np.log(hvsr_mat), axis=0))
                hvsr_sd = np.std(np.log(hvsr_mat), axis=0)

        # Call plot_hvsr function here to allow continuous selection
        if plot_hvsr:
            idx_select, hvsr_mean, hvsr_sd, hvsr_sd1, h1_mat, h2_mat, v_mat, h1_mat_mean, h2_mat_mean, v_mat_mean = \
                HvsrProc.plot_hvsr(freq_hv_mean, hvsr_mat, idx_select, hvsr_mean, hvsr_sd, hvsr_sd1, h1_mat,
                                   h2_mat, v_mat, h1_mat_mean, h2_mat_mean, v_mat_mean, robust_est=robust_est,
                                   sjb_avg=sjb_avg,
                                   distribution=distribution)
        else:
            pass

        plt.show()

        res = {'idx_select': idx_select, 'hvsr_mean': hvsr_mean, 'hvsr_sd': hvsr_sd, 'FAS_h1_mean': h1_mat_mean,
               'FAS_h2_mean': h2_mat_mean, 'FAS_v_mean': v_mat_mean}

        if robust_est:
            res['hvsr_sd1'] = hvsr_sd1
        return res

    @staticmethod
    def process_polar_curve(polar_data, azimuths, standard_freqs):

        """

        function which processes polar curves to be in a specific format to plot using plot_polar_ratio

        Parameters
        ----------

        polar_data: dataframe
            Dataframe containing polar data.
        azimuths: ndarray
            Array containing polar azimuths.
        standard_freqs: ndarray
            Array containing the freqs to plot the polar data.

        returns dataframe with an updated format

        """

        unique_polar_freqs = standard_freqs['unique_polar_freqs']

        polar = polar_data.copy()

        # Filter polar data by frequency range
        polar = polar.loc[(polar['Freq_0'] >= 0.1) & (polar['Freq_0'] <= 50)].reset_index(drop=True)

        # Interpolate missing frequencies for each azimuth
        polar2 = polar.copy()
        freq_fields = ['Freq_' + str(azimuth) for azimuth in azimuths]
        for f in unique_polar_freqs:
            if f not in np.array(polar2['Freq_0']):
                polar2.loc[len(polar2), freq_fields] = [f] * len(freq_fields)
        polar2.sort_values(by='Freq_0')
        polar2 = polar2.set_index('Freq_0').interpolate(method='index').reset_index()
        polar3 = polar2[polar2['Freq_0'].isin(unique_polar_freqs)].reset_index(drop=True)
        polar = polar3.copy()

        # Process each azimuth
        result_data = pd.DataFrame()
        for azimuth in azimuths:
            new_polar = polar[['Freq_%i' % azimuth, 'HVSR_%i' % azimuth, 'HVSR_sd_%i' % azimuth]]
            new_polar = new_polar.rename(columns={'Freq_%i' % azimuth: 'frequency',
                                                  'HVSR_%i' % azimuth: 'ratio',
                                                  'HVSR_sd_%i' % azimuth: 'standard_deviation'})

            new_polar['azimuth_values'] = azimuth  # Add azimuth column

            result_data = pd.concat([result_data, new_polar], axis=0)

        return result_data

    @staticmethod
    def plot_polar_ratio(result_data):

        """

       function which plots the polar data.

       Parameters
       ----------

       result_data: dataframe
           Dataframe containing polar ratio, azimuths, and frequencies.

       returns plotted polar data

       """
        polar_freq = result_data['frequency']
        azimuth_v = result_data['azimuth_values']
        polar_ratio = result_data['ratio']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Azimuth (Degrees)')
        sc = ax.scatter(polar_freq, azimuth_v, c=polar_ratio, cmap='viridis')
        ax.set_xscale('log')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Polar Ratio')

        return fig

    @staticmethod
    def polar_plot(x, y, z, log10_scale='x', xlab='Freq (Hz)', ylab='Azimuth (deg)'):
        """

       function which plots the polar data.

       returns plotted polar data

       """
        color_s = plt.cm.get_cmap(np.linspace(0, 1, 100))
        cmap = LinearSegmentedColormap.from_list("terrain_custom", color_s)

        if log10_scale == 'x':
            plt.imshow(z, aspect='auto', cmap=cmap, extent=(x[0], x[-1], y[0], y[-1]))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.xscale('log')
        elif log10_scale == 'y':
            plt.imshow(z.T, aspect='auto', cmap=cmap, extent=(x[0], x[-1], y[0], y[-1]))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.yscale('log')
        elif log10_scale == 'xy':
            plt.imshow(z, aspect='auto', cmap=cmap, extent=(x[0], x[-1], y[0], y[-1]))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.imshow(z, aspect='auto', cmap=cmap, extent=(x[0], x[-1], y[0], y[-1]))
            plt.xlabel(xlab)
            plt.ylabel(ylab)

        # add color bar
        cbar = plt.colorbar(label='Polar Ratio')
        cbar.ax.set_ylabel('Polar Ratio')
        plt.show()

    @staticmethod
    def plot_mean_hvsr(csv_file_path, metadata_csv_path, xlim_m=65, ylim_m=4,
                       xlim=(0.01, 50), ylim=(0, 5), robust_est=False, metadata_flag=True):
        """

        function which plots the mean HVSR data.

        Parameters
        ----------

        csv_file_path: string
            String indicating the path and filename to read mean.csv from.
        metadata_csv_path: string
            String indicating the path and the filename to read metadata.csv.
        xlim: range
            range of x limit
        ylim: range
            range of y limit
        robust_est: boolean
            Specifies whether robust_est was used to compute the HVSR, default=False
        metadata_flag: boolean
            Specifies whether to show the metadata as text near the plot. Default=True

        returns plotted mean HVSR data

        """
        if robust_est:
            # Read CSV file
            mean_df = pd.read_csv(csv_file_path)

            # Extract data
            mean_hv = mean_df['HVSR median']
            freq_hv = mean_df['freq_Hz']
            std_hv = mean_df['HVSR mad']

            # Plotting
            fig = plt.figure(figsize=(8, 6))
            plt.title('Mean HVSR Spectra')
            plt.semilogx(freq_hv, mean_hv, 'k', label='HVSR Mean')
            plt.semilogx(freq_hv, mean_hv + std_hv, '--', color='red', label='+/- 1 Std. Dev.')
            plt.semilogx(freq_hv, mean_hv - std_hv, '--', color='red')
            plt.fill_between(x=freq_hv, y1=mean_hv - std_hv, y2=mean_hv + std_hv, color='dimgray', alpha=0.3, zorder=2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('HVSR Amplitude')
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.legend(loc='upper right')

            if metadata_flag:
                # Read metadata CSV file
                metadata_df = pd.read_csv(metadata_csv_path)
                metadata = metadata_df.to_dict(orient='records')[0]
                metadata_text = "\n".join(f"{key}: {value}" for key, value in metadata.items())

                # Add metadata
                if metadata_text.strip():  # Check if metadata text is not empty or whitespace
                    plt.text(xlim_m, ylim_m, metadata_text, fontsize=12, ha='left', va='center')

            plt.show()

        else:
            # Read CSV file
            mean_df = pd.read_csv(csv_file_path)

            # Extract data
            mean_hv = mean_df['HVSR mean']
            freq_hv = mean_df['freq_Hz']
            std_hv = mean_df['HVSR sd']

            # Plotting
            fig = plt.figure(figsize=(8, 6))
            plt.title('Mean HVSR Spectra')
            plt.semilogx(freq_hv, mean_hv, 'k', label='HVSR Mean')
            plt.semilogx(freq_hv, mean_hv + std_hv, '--', color='red', label='+/- Std. Dev.')
            plt.semilogx(freq_hv, mean_hv - std_hv, '--', color='red')
            plt.fill_between(x=freq_hv, y1=mean_hv - std_hv, y2=mean_hv + std_hv, color='dimgray', alpha=0.3, zorder=2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('HVSR Amplitude')
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.legend(loc='upper right')

            if metadata_flag:
                # Read metadata CSV file
                metadata_df = pd.read_csv(metadata_csv_path)
                metadata = metadata_df.to_dict(orient='records')[0]
                metadata_text = "\n".join(f"{key}: {value}" for key, value in metadata.items())

                # Add metadata
                if metadata_text.strip():  # Check if metadata text is not empty or whitespace
                    plt.text(xlim_m, ylim_m, metadata_text, fontsize=12, ha='left', va='center')

            plt.show()

        return fig

    @staticmethod
    def plot_selected_time_series(csv_file_path):
        """

       function which plots the time series data.

       Parameters
       ----------

       csv_file_path: string
           String indicating the path and filename to read .csv from.

       returns plotted time series data

       """

        # Read CSV file
        ts_df = pd.read_csv(csv_file_path, header=None, low_memory=False)
        ts_df = ts_df[1:].astype(float)

        colors = ['yellow', 'lightblue'] * int(len(ts_df.columns) / 4)

        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(8, 6))

        time_list = list()
        for i in range(0, int(len(ts_df.columns) / 4)):
            t = ts_df[4 * i][1:]
            h1 = ts_df[4 * i + 1][1:]
            h2 = ts_df[4 * i + 2][1:]
            v = ts_df[4 * i + 3][1:]
            time_list.append(t)
            ax[0].plot(t, h1, color=colors[i])
            ax[1].plot(t, h2, color=colors[i])
            ax[2].plot(t, v, color=colors[i])

        ax[0].set_title('Horizontal - East')
        ax[0].set_xlim(0, max(t))
        ax[0].set_ylabel('Counts')
        ax[0].grid(True, which='both')
        ax[1].set_title('Horizontal - North')
        ax[1].set_xlim(0, max(t))
        ax[1].set_ylabel('Counts')
        ax[1].grid(True, which='both')
        ax[2].set_title('Vertical')
        ax[2].set_xlim(0, max(t))
        ax[2].set_ylabel('Counts')
        ax[2].set_xlabel('Time (Sec.)')
        ax[2].grid(True, which='both')

        plt.tight_layout()

        plt.show()

        return fig

    @staticmethod
    def plot_selected_hvsr(csv_file_path, xlim=(0.1, 50), ylim=(0, 8)):
        """

        function which plots the selected HVSR data.

        Parameters
        ----------

        csv_file_path: string
            String indicating the path and filename to read .csv from.
        xlim: range
            Range of x limit
        ylim: range
            Range of y limit

        returns plotted selected HVSR data

        """

        # Read CSV file
        sel_hv_df = pd.read_csv(csv_file_path, header=None)
        sel_hv_df = sel_hv_df[1:].astype(float)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(1, int(len(sel_hv_df.columns))):
            freq = sel_hv_df[0][1:]
            hvsr = sel_hv_df[i][1:]
            ax.semilogx(freq, hvsr, color='lightgray')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('HVSR Amplitude')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        plt.show()

        return fig

    @staticmethod
    def plot_fas(csv_path, xlim=(0.1, 50), ylim=(10e2, 10e8)):

        """

       function which plots the Fourier Amplitude Spectra (FAS) data.

       Parameters
       ----------

       csv_path: string
           String indicating the path and filename to read .csv from.
       xlim: range
            Range of x limit
       ylim: range
            Range of y limit

       returns plotted FAS data

       """
        # Read CSV file
        fas_df = pd.read_csv(csv_path)

        # Extract data
        freq = fas_df['freq_Hz']
        h1 = fas_df['FAS mean h1']
        h2 = fas_df['FAS mean h2']
        v = fas_df['FAS mean v']

        # Plot
        fig = plt.figure(figsize=(8, 6))
        plt.loglog(freq, h1, color='red', label='h1')
        plt.loglog(freq, h2, color='green', label='h2')
        plt.loglog(freq, v, color='blue', label='v')

        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FAS Amplitude')
        plt.legend()
        plt.show()

        return fig

    def hv_proc(self, is_noise=True, eqk_filepath=None, output_pf_flnm='Test_',
                distribution='normal', robust_est=False, time_cut=120, file_type=1, trim_flag=False,
                pre_filter_flag=True, pre_filter_is_causal=False,
                pre_filter_hpass_fc=0.042, pre_filter_lpass_fc=None, pre_filter_npole_hp=-5,
                pre_filter_npole_lp=4, pre_filter_order_zero_padding=0, pre_filter_t_front=10,
                pre_filter_t_end=10, filter_flag=False, is_causal=False, hpass_fc=0.0083, lpass_fc=None,
                npole_hp=-5, npole_lp=4, order_zero_padding=0, detrend_type=1,
                taper_flag=True, t_front=10, t_end=10, horizontal_comb='geometric_mean', ko_smooth_flag=True,
                ko_smooth_b=40, parzen_flag=False, parzen_bwidth=1.5, win_width=300, overlapping=0,
                sta_lta_flag=False, short_term_len=1, long_term_len=30, sta_lta_moving_term=1,
                deg_increment=10, resample_lin2log=True, deci_mean_factor=10, sjb_avg=False,
                deci_polar_factor=10, output_freq_min=0.01, output_freq_max=50, resampling_length = 2000, plot_ts=True, plot_hvsr=True,
                output_selected_ts=True, output_removed_ts=True,
                output_selected_hvsr=True, output_removed_hvsr=True,
                output_mean_curve=True, output_polar_curves=False, output_fas_mean_curve=True,
                output_metadata=True):
        """

        function combines all other functions within class HvsrProc and processes the time series data then
        computes the HVSR for the windowed time series data.

        Parameters
        ----------

        is_noise: boolean
            Indicates whether the data is microtremor (True) or earthquake (False) data. default = True.
        eqk_filepath: string
            Indicates the filepath where earthquake data is stored. Default = None.
        output_pf_flnm: string
            Indicates the output filename.
        distribution: string
            Indicates the type  of distribution. normal or log_normal, default = normal.
        robust_est: boolean
            Indicates whether robust_est is to be used. Default = False.
        time_cut: int
            Integer representing the amount cut from the time series in minutes. Default = 5 (min)
        file_type: int
            Specifies whether the filetype is a mseed file (1) or text file (2). Default = 1 (mseed)
        trim_flag: boolean
            Specifies whether to use the miniseed tool to trim the data
            based on specific start time and end time. Default = False
        pre_filter_flag: boolean
            Indicates whether pre-filtering is to be applied. Default = True.
        pre_filter_is_causal: boolean
            Indicates whether to apply a causal or acausal filter. Default = False indicating acausal.
        pre_filter_hpass_fc: float
            Indicates the high pass filtering value. Default = 0.042.
        pre_filter_lpass_fc: float
            Indicates the low pass filtering value. Default = None
        pre_filter_npole_hp: int
            Indicates whether to apply high pass. Default = -5
        pre_filter_npole_lp: int
            Indicates whether to apply low pass. Default = 4
        pre_filter_order_zero_padding: int
            Indicates what order of zero padding to be applied. Default = 4
        pre_filter_t_front: int
            Indicates the percentage of Tukey tapering be applied. Default = 10
        pre_filter_t_end: int
            Indicates the percentage of Tukey tapering to be applied. Default = 10
        filter_flag: boolean
            Indicates whether filtering is to be applied within windows.Default = True
        is_causal: boolean
            Indicates whether a causal or acausal filter is to be applied. Default = False indicating acausal.
        hpass_fc: float
            Indicates the high pass filtering value to apply to each window. Default = 0.0083.
        lpass_fc: float
            Indicates the low pass filtering value to apply to each window. Default = None
        npole_hp: int
            Indicates whether to apply a high pass filter. Default = -5
        npole_lp: int
            Indicates whether to apply a low pass filter. Default = 4
        order_zero_padding: int
            Indicates the order of zero padding to be applied to each window.
        detrend_type: int
            Indicates what type of detrend to be applied, 1 = mean removal, 2 = linear detrend,
            6 = 5th order polynomial detrend. default = 1.
        taper_flag: boolean
            Indicates whether to apply a Tukey windowing function. default = True
        t_front: int
            Indicates the percentage of tapering to be applied at the start of each window. default = 10.
        t_end: int
            Indicates the percentage of tapering to be applied at the end of each window. default = 10.
        horizontal_comb: string
            Indicates the type of combination to be done to the horizontal components, ps_RotD50 = Compute the RotD50
            of the two components, squared_average = Compute the squared average of the two components, geometric mean =
            Compute the geometric mean of the two components. Default = ps_RotD50.
        ko_smooth_flag: boolean
            Indicates the application of Konno and Ohmachi (KO) smoothing to the FAS. Default = True.
        ko_smooth_b: int
            Indicates the smoothing factor to be used in KO smoothing. Default = 40
        parzen_flag: boolean
            Indicates whether to apply parzen smoothing. Default = False
        parzen_bwidth: float
            Indicates the smoothing factor to be used in parzen smoothing. Default = 1.5.
        win_width: int
            Indicates the length each window should be. Default = 300.
        overlapping: int
            Indicates the amount of overlapping to be applied within each window. Default = 0
        sta_lta_flag: boolean
            Indicates whether to compute the short term and long term averages within each window. default = False.
        short_term_len: int
            Indicates the short term length. Default = 1
        long_term_len: int
            Indicates the long term length. Default = 30
        sta_lta_moving_term: int
            Indicates the sta/lta ratio term. Default = 1
        deg_increment: int
            Indicates the degree increment for computing the polar curve.
        resample_lin2log: boolean
            Indicated whether to resample the frequencies. Default = True
        deci_mean_factor: int
            Indicates the decimal mean factor to apply to resampled frequency. Default = 10
        sjb_avg: boolean
            Specifies whether to take the average smoothed FAS before computing HVSR. Default = False
        deci_polar_factor: int
            Indicates the decimal polar factor to apply to polar frequency. Default = 10
        output_freq_min: float
            Indicates the output minimum frequency. Default =  0.01
        output_freq_max: float
            Indicates the output maximum frequency. Default = 50.0
        resampling_length: int
            Indicates the number of points to resample the data. Defualt = 2000
        plot_ts: boolean
            Specifies whether to plot the time series for window selection, default = True.
        plot_hvsr: boolean
            Specifies whether to plot the FAS and HVSR for window selection, default = True.
        output_selected_ts: boolean
            Indicates whether to output the selected time series.
        output_removed_ts: boolean
            Indicates whether to output the removed time series.
        output_selected_hvsr: boolean
            Indicates whether to output the selected hvsr curves.
        output_removed_hvsr: boolean
            Indicates whether to output the removed hvsr curves.
        output_mean_curve: boolean
            Indicates whether to output the mean hvsr curves.
        output_polar_curves: boolean
            Indicates whether to output the polar curves.
        output_fas_mean_curve: boolean
            Indicates whether to output the mean fas curves for each component.
        output_metadata: boolean
            Indicates whether to output the metadata after processing.

        returns the HVSR mean curve, mean FAS, selected/unselected time series,
        selected/unselected HVSR curves, and the polar curves from the microtremor data input.

        """

        time, h1, h2, v, ts_wins, h1_wins, h2_wins, v_wins, h1_stalta, h2_stalta, v_stalta = \
            self.pre_process_noise_data(time_cut=time_cut, file_type=file_type, trim_flag=trim_flag,
                                        detrend_type=detrend_type, taper_flag=taper_flag, t_front=t_front, t_end=t_end,
                                        pre_filter_flag=pre_filter_flag, pre_filter_t_front=pre_filter_t_front,
                                        pre_filter_t_end=pre_filter_t_end, pre_filter_hpass_fc=pre_filter_hpass_fc,
                                        pre_filter_npole_hp=pre_filter_npole_hp,
                                        pre_filter_is_causal=pre_filter_is_causal,
                                        pre_filter_order_zero_padding=pre_filter_order_zero_padding,
                                        pre_filter_lpass_fc=pre_filter_lpass_fc,
                                        pre_filter_npole_lp=pre_filter_npole_lp, is_noise=is_noise, win_width=win_width,
                                        overlapping=overlapping, sta_lta_flag=sta_lta_flag,
                                        short_term_len=short_term_len, long_term_len=long_term_len,
                                        sta_lta_moving_term=sta_lta_moving_term,
                                        filter_flag=filter_flag, hpass_fc=hpass_fc, npole_hp=npole_hp,
                                        lpass_fc=lpass_fc, npole_lp=npole_lp, eqk_filepath=eqk_filepath,
                                        is_causal=is_causal, order_zero_padding=order_zero_padding)

        num_wins = len(h1_wins)

        if plot_ts:
            idx_select = self.td_plt_select(ts_wins=ts_wins, h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins,
                                            sta_lta_flag=sta_lta_flag, h1_stalta=h1_stalta, h2_stalta=h2_stalta,
                                            v_stalta=v_stalta)
        else:
            idx_select = list(range(num_wins))

        time_ts = time

        print("Time-domain selection is DONE!")

        # time-domain data output
        if len(idx_select) == 0:
            raise ValueError('No window is selected, please try different data!')
        else:

            if output_selected_ts:
                outputflname_ts_sel = output_pf_flnm + 'ts_sel.csv'
                max_win_len = max([len(win) for win in h1_wins])

                idx_remove = [i for i in range(1, num_wins + 1) if i not in idx_select]

                ts_sel_out = np.full((max_win_len, 4 * len(idx_select)), np.nan)
                colnames = []
                for i, idx in enumerate(idx_select):
                    start_col = 4 * i
                    # end_col = start_col + 4
                    if idx <= len(h1_wins) and idx not in idx_remove:
                        window_length = len(h1_wins[idx - 1]) * self.dt
                        start_time_idx = int(idx * window_length / self.dt)
                        start_time = time_ts[start_time_idx]
                        # Ensure the end time aligns with the length of the window
                        end_time_idx = start_time_idx + len(h1_wins[idx - 1])
                        end_time = time_ts[min(end_time_idx, len(time_ts) - 1)]
                        ts_sel_out[:len(h1_wins[idx - 1]), start_col] = np.linspace(start_time, end_time,
                                                                                    len(h1_wins[idx - 1]))
                        ts_sel_out[:len(h1_wins[idx - 1]), start_col + 1] = h1_wins[idx]
                        ts_sel_out[:len(h1_wins[idx - 1]), start_col + 2] = h2_wins[idx]
                        ts_sel_out[:len(h1_wins[idx - 1]), start_col + 3] = v_wins[idx]
                        colnames.extend([f"Time_s_{idx}", f"H1_amp_{idx}", f"H2_amp_{idx}", f"V_amp_{idx}"])

                ts_sel_out_df = pd.DataFrame(np.round(ts_sel_out[:, :4 * len(idx_select)], 5), columns=colnames)
                ts_sel_out_df.to_csv(os.path.join(self.output_dir, outputflname_ts_sel), index=False)

            if output_removed_ts:
                outputflname_ts_unsel = output_pf_flnm + 'ts_unsel.csv'
                max_win_len = max([len(win) for win in h1_wins])

                # Initialize idx_remove
                idx_remove = [i for i in range(1, num_wins) if i not in idx_select]

                ts_unsel_out = np.full((max_win_len, 4 * len(idx_remove)), np.nan)
                colnames = []
                for i, idx in enumerate(idx_remove):
                    start_col = 4 * i
                    if idx <= len(h1_wins):
                        window_length = len(h1_wins[idx - 1]) * self.dt
                        start_time_idx = int(idx * window_length / self.dt)
                        if start_time_idx >= len(time_ts):
                            continue
                        start_time = time_ts[start_time_idx]
                        end_time_idx = start_time_idx + len(h1_wins[idx - 1])
                        end_time = time_ts[min(end_time_idx, len(time_ts) - 1)]
                        ts_unsel_out[:len(h1_wins[idx - 1]), start_col] = np.linspace(start_time, end_time,
                                                                                      len(h1_wins[idx - 1]))
                        ts_unsel_out[:len(h1_wins[idx]), start_col + 1] = h1_wins[idx]
                        ts_unsel_out[:len(h1_wins[idx]), start_col + 2] = h2_wins[idx]
                        ts_unsel_out[:len(h1_wins[idx]), start_col + 3] = v_wins[idx]
                        colnames.extend([f"Time_s_{idx}", f"H1_amp_{idx}", f"H2_amp_{idx}", f"V_amp_{idx}"])

                ts_unsel_out_df = pd.DataFrame(np.round(ts_unsel_out[:, :4 * len(idx_remove)], 5), columns=colnames)
                ts_unsel_out_df.to_csv(os.path.join(self.output_dir, outputflname_ts_unsel), index=False)

        print("Preparing for frequency-domain, please wait...")

        min_win_len = min([len(win) for win in h1_wins])
        n_nyq_min = min_win_len // 2 + 1
        df = 1 / (min_win_len * self.dt)
        freq = np.arange(1, n_nyq_min) * df

        if freq[0] == 0:
            freq = freq[1:]
        if resample_lin2log:
            freq = np.logspace(np.log10(min(freq)), np.log10(max(freq)), num=resampling_length)
        if deci_mean_factor > 0:
            freq_hv_mean = freq[::int(np.floor(deci_mean_factor))]
        else:
            freq_hv_mean = freq

        hvsr_list = HvsrProc.hvsr_win_calc(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, ts_dt=self.dt,
                                           ko_smooth_b=ko_smooth_b, ko_smooth_flag=ko_smooth_flag,
                                           parzen_flag=parzen_flag, parzen_bwidth=parzen_bwidth,
                                           horizontal_comb=horizontal_comb, freq_hv_mean=freq_hv_mean,
                                           polar_curves_flag=False, sjb_avg=sjb_avg)

        fd_select = HvsrProc.fd_plt_select(hvsr_list=hvsr_list, robust_est=robust_est, freq_hv_mean=freq_hv_mean,
                                           distribution=distribution, plot_hvsr=plot_hvsr, sjb_avg=sjb_avg)

        iidx_select = fd_select['idx_select']

        print("Frequency-domain selection is DONE!")

        # Frequency-domain data output
        if len(iidx_select) == 0:
            raise ValueError('No window is selected, please try different data!')
        else:
            if output_selected_hvsr:
                hvsr_sel_out = np.full((len(freq_hv_mean), len(iidx_select) + 1), np.nan)
                hvsr_sel_out[:, 0] = freq_hv_mean

                selected_windows = []

                for i, idx in enumerate(iidx_select):
                    hv_ratio = hvsr_list['hv_ratio'][idx]
                    if hv_ratio is not None:
                        hvsr_sel_out[:, i + 1] = hv_ratio
                        selected_windows.append(idx + 1)

                colnames = ['freq_Hz'] + [f'HVSR_{window_idx}' for window_idx in selected_windows]

                hvsr_sel_out = pd.DataFrame(np.round(hvsr_sel_out, 5), columns=colnames)

                outputflname_hvsr_sel = output_pf_flnm + 'hvsr_sel.csv'

                if output_freq_min is not None and output_freq_min > np.min(hvsr_sel_out.iloc[:, 0]):
                    idx_min = np.where(hvsr_sel_out.iloc[:, 0] >= output_freq_min)[0]
                    hvsr_sel_out = hvsr_sel_out.iloc[idx_min]

                if output_freq_max is not None and output_freq_max < np.max(hvsr_sel_out.iloc[:, 0]):
                    idx_max = np.where(hvsr_sel_out.iloc[:, 0] <= output_freq_max)[0]
                    hvsr_sel_out = hvsr_sel_out.iloc[idx_max]

                hvsr_sel_out.to_csv(self.output_dir + '/' + outputflname_hvsr_sel, index=False)

            if output_mean_curve:
                outputflname_hvsr_mean = output_pf_flnm + 'hvsr_mean.csv'
                if robust_est:
                    hvsr_mean_out = pd.DataFrame({
                        'freq_Hz': np.round(freq_hv_mean, 5),
                        'HVSR median': np.round(fd_select['hvsr_mean'], 5),
                        'HVSR mad': np.round(fd_select['hvsr_sd'], 5),
                        'HVSR IQR': np.round(fd_select['hvsr_sd1'], 5)
                    })
                else:
                    hvsr_mean_out = pd.DataFrame({
                        'freq_Hz': np.round(freq_hv_mean, 5),
                        'HVSR mean': np.round(fd_select['hvsr_mean'], 5),
                        'HVSR sd': np.round(fd_select['hvsr_sd'], 5)
                    })
                if not np.isnan(output_freq_min) and output_freq_min > np.min(hvsr_mean_out['freq_Hz']):
                    hvsr_mean_out = hvsr_mean_out[hvsr_mean_out['freq_Hz'] >= output_freq_min]
                if not np.isnan(output_freq_max) and output_freq_max < np.max(hvsr_mean_out['freq_Hz']):
                    hvsr_mean_out = hvsr_mean_out[hvsr_mean_out['freq_Hz'] <= output_freq_max]
                hvsr_mean_out.to_csv(self.output_dir + '/' + outputflname_hvsr_mean, index=False)

            if output_fas_mean_curve:
                outputflname_fas_mean = output_pf_flnm + 'FAS_mean.csv'
                if robust_est:
                    fas_mean_out = pd.DataFrame({
                        'freq_Hz': np.round(freq_hv_mean, 5),
                        'FAS median h1': np.round(fd_select['FAS_h1_mean'], 5),
                        'FAS median h2': np.round(fd_select['FAS_h2_mean'], 5),
                        'FAS median v': np.round(fd_select['FAS_v_mean'], 5)
                    })
                else:
                    fas_mean_out = pd.DataFrame({
                        'freq_Hz': np.round(freq_hv_mean, 5),
                        'FAS mean h1': np.round(fd_select['FAS_h1_mean'], 5),
                        'FAS mean h2': np.round(fd_select['FAS_h2_mean'], 5),
                        'FAS mean v': np.round(fd_select['FAS_v_mean'], 5)
                    })
                if not np.isnan(output_freq_min) and output_freq_min > np.min(fas_mean_out['freq_Hz']):
                    fas_mean_out = fas_mean_out[fas_mean_out['freq_Hz'] >= output_freq_min]
                if not np.isnan(output_freq_max) and output_freq_max < np.max(fas_mean_out['freq_Hz']):
                    fas_mean_out = fas_mean_out[fas_mean_out['freq_Hz'] <= output_freq_max]
                fas_mean_out.to_csv(self.output_dir + '/' + outputflname_fas_mean, index=False)

            if output_removed_hvsr:
                if len(idx_select) > len(iidx_select):
                    idx_remove = [i for i in range(len(idx_select)) if idx_select[i] not in iidx_select]

                    outputflname_hvsr_unsel = output_pf_flnm + 'hvsr_unsel.csv'
                    hvsr_unsel_out = np.full((len(freq_hv_mean), len(idx_remove) + 1), np.nan)
                    hvsr_unsel_out[:, 0] = freq_hv_mean

                    removed_windows = []

                    for i, idx in enumerate(idx_remove):
                        hv_ratio = hvsr_list['hv_ratio'][idx]
                        if hv_ratio is not None:
                            hvsr_unsel_out[:, i + 1] = hv_ratio
                            removed_windows.append(idx_select[idx])

                    col_names = ['freq_Hz'] + [f'HVSR_{idx}' for idx in
                                               removed_windows]

                    hvsr_unsel_out = pd.DataFrame(np.round(hvsr_unsel_out, 5), columns=col_names)

                    if not np.isnan(output_freq_min) and output_freq_min > np.min(hvsr_unsel_out['freq_Hz']):
                        hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] >= output_freq_min]

                    if not np.isnan(output_freq_max) and output_freq_max < np.max(hvsr_unsel_out['freq_Hz']):
                        hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] <= output_freq_max]

                    hvsr_unsel_out.to_csv(os.path.join(self.output_dir, outputflname_hvsr_unsel), index=False)

            # Generate polar curves and output
            if output_polar_curves:
                print("Calculating and generating polar curve data, please wait......")
                outputflname_hvsr_polar = output_pf_flnm + 'hvsr_polar.csv'

                if deci_polar_factor > 0:
                    freq_polar = freq[::int(deci_polar_factor)]
                else:
                    freq_polar = freq

                hvsr_list = HvsrProc.hvsr_win_calc(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, ts_dt=self.dt,
                                                   ko_smooth_b=ko_smooth_b, ko_smooth_flag=ko_smooth_flag,
                                                   parzen_flag=parzen_flag, parzen_bwidth=parzen_bwidth,
                                                   horizontal_comb=horizontal_comb, freq_hv_mean=freq_hv_mean,
                                                   polar_curves_flag=True, freq_polar=freq_polar,
                                                   deg_increment=deg_increment,
                                                   sjb_avg=sjb_avg)

                polar_degs = np.arange(0, 180, deg_increment)
                polar_hvsr_mat = np.empty((len(freq_polar), len(polar_degs) * 3), dtype=np.float64)
                tmp_hvsr_mat = np.empty((len(freq_polar), len(idx_select)), dtype=np.float64)

                for i, deg in enumerate(polar_degs):
                    hvsr_data = hvsr_list['polar_hv_ratio']
                    tmp_hvsr_mat[:, :] = hvsr_data[:, i][:, np.newaxis]
                    tmp_hvsr_mat[tmp_hvsr_mat <= 0] = 10e-5  # avoid log(0 or negative)

                    if distribution == 'normal':
                        tmp_mean = np.mean(tmp_hvsr_mat, axis=1)
                        tmp_sd = np.std(tmp_hvsr_mat, axis=1)
                    elif distribution == 'log_normal':
                        tmp_mean = np.exp(np.mean(np.log(tmp_hvsr_mat), axis=1))
                        tmp_sd = np.std(np.log(tmp_hvsr_mat), axis=1)

                    polar_hvsr_mat[:, 3 * i] = np.round(freq_polar, 5)
                    polar_hvsr_mat[:, 3 * i + 1] = np.round(tmp_mean, 5)
                    polar_hvsr_mat[:, 3 * i + 2] = np.round(tmp_sd, 5)

                polar_hvsr_df = pd.DataFrame(polar_hvsr_mat)

                column_names = []
                for deg in polar_degs:
                    column_names.extend([f'Freq_{deg}', f'HVSR_{deg}', f'HVSR_sd_{deg}'])

                polar_hvsr_df.columns = column_names

                if not np.isnan(output_freq_min):
                    polar_hvsr_df = polar_hvsr_df[polar_hvsr_df['Freq_0'] >= output_freq_min]

                if not np.isnan(output_freq_max):
                    polar_hvsr_df = polar_hvsr_df[polar_hvsr_df['Freq_0'] <= output_freq_max]

                polar_hvsr_df.to_csv(os.path.join(self.output_dir, outputflname_hvsr_polar), index=False)

            # Output Metadata
            if output_metadata:
                outputflname_meta = output_pf_flnm + 'metadata.csv'
                meta_names = ['sample freq (Hz)', 'record duration (min)', 'pre_proc_applied',
                              'weighted_mean_removal', 'pre_proc_high_pass_filter_type',
                              'pre_proc_high_pass_corner_frequency', 'pre_proc_taper_type',
                              'pre_proc_taper_width_percent', 'detrend_type', 'time window (sec)',
                              'window overlap (sec)', 'taper type',
                              'front taper width (percentage)',
                              'end taper width (percentage)', 'horizontal combination', 'number of windows (total)',
                              'number of windows (selected)', 'high pass filter',
                              'high pass filter corner frequency (Hz)', 'high pass filter type',
                              'smoothing type', 'smoothing constant', 'data type', 'distribution',
                              'processing_comments']
                meta_output = pd.DataFrame(columns=meta_names)
                meta_output.loc[0] = [1 / self.dt, np.round(len(h1) * self.dt / 60, 4), 1 if pre_filter_flag else 0,
                                      1 if pre_filter_flag else 0, 'Butterworth', pre_filter_hpass_fc, 'Tukey',
                                      pre_filter_t_front,
                                      'no detrend' if detrend_type == 0 else 'mean removal' if detrend_type == 1 else
                                      'linear detrend' if detrend_type == 2 else 'fifth order polynomial detrend',
                                      win_width, overlapping, 'Tukey', t_front, t_end, horizontal_comb, num_wins,
                                      len(iidx_select), 0 if filter_flag and not np.isnan(hpass_fc) else 1,
                                      hpass_fc, 'Butterworth', 'KonnoOhmachi', ko_smooth_b, 0 if is_noise else 1,
                                      distribution, None]
                meta_output.to_csv(self.output_dir + '/' + outputflname_meta, index=False)

        win_result = {'ts_wins': ts_wins, 'h1_wins': h1_wins, 'h2_wins': h2_wins, 'v_wins': v_wins,
                      'h1_stalta': h1_stalta, 'h2_stalta': h2_stalta, 'v_stalta': v_stalta, 'idx_select_ts': idx_select}

        print("Everything is DONE, check out the results in the output folder!")

        return win_result, fd_select
