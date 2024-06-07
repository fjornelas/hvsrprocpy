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

"""Class definition for TsProc object."""

import os
import numpy as np
import pandas as pd

import obspy

import matplotlib.pyplot as plt

from scipy.signal.windows import tukey

__all__ = ['TsProc']


class TsProc:
    """
         This class is developed for processing noise-based
         and earthquake-based time series data.

    """

    @staticmethod
    def _check_type(name, input_type):
        """

        Function to check filetypes

        Parameters
        ----------
        name: string
            Indicates the name of the component or object.
        input_type: string
            Indicates the filename to check.

        returns the filename if the object is a string.

        """
        if not isinstance(input_type, str):
            msg = f"{name} must be a string indicating the file type, not {type(input_type)}."
            raise TypeError(msg)
        return input_type

    @staticmethod
    def _check_array(name, array_type):
        """

        Function to check whether input is array.

        Parameters
        ----------
        name: string
            Indicates the name of the component or object.
        array_type: ndarray
            Indicates the filename to check whether the file is an array.

        returns the filename if the object is an array.

        """
        if not isinstance(array_type, np.ndarray):
            msg = f"{name} must be a Numpy 1D array, not a {type(array_type)}."
            raise TypeError(msg)
        return array_type

    def __init__(self, h1, h2, v, dt, time, directory, output_dir, mseed_file_flag=True):

        """

        Parameters
        ----------
        h1 : string/ndarray
            string/array of h1 time series. Each row represents timeseries data
            from the first horizontal component.
        h2 :  string/ndarray
            string/array of h2 time series. Each row represents timeseries data
            from the second horizontal component.
        v : string/ndarray
            string/array of v time series. Each row represents timeseries data
            from the vertical component.
        directory: string
            directory where data is stored.
        output_dir : string
            directory where to save data.
        mseed_file_flag: boolean
            Indicates whether the input is an array or a string,
            True indicates string. Default = True.

        """

        if mseed_file_flag:
            self.h1 = TsProc._check_type('horizontal_1', h1)
            self.h2 = TsProc._check_type('horizontal_2', h2)
            self.v = TsProc._check_type('vertical', v)
            self.dt = dt
            self.time = TsProc._check_array('time', time)
            self.directory = TsProc._check_type('directory', directory)
            self.output_dir = TsProc._check_type('output_directory', output_dir)
        else:
            try:
                self.h1 = TsProc._check_array('horizontal_1', h1)
                self.h2 = TsProc._check_array('horizontal_2', h2)
                self.v = TsProc._check_array('vertical', v)
                self.dt = dt
                self.time = TsProc._check_array('time', time)
                self.directory = TsProc._check_type('directory', directory)
                self.output_dir = TsProc._check_type('output_directory', output_dir)
            except AttributeError as e:
                raise TypeError(f'component h1 must be a time series array not {type(self.h1)}') from e

    @staticmethod
    def add_mseed_tool(st):
        """

        Function to trim traces if necessary.

        Parameters
        ----------

        st: object
            Stream object containing traces to trim.

        returns trimmed stream.

        """
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

        """

        Function to process miniseed files.

        Parameters
        ----------

        file_direc: string
            Directory where files are stored.
        h1_fn: string
            Filename of h1
        h2_fn: string
            Filename of h2
        v_fn: string
            Filename of v
        trim_flag: boolean
            Indicates whether trimming is needed. Default = False
        time_cut: int
            Indicates the amount of time to be cut in sec. Default = 300 sec.

        returns the component in array format.

        """

        h1 = obspy.read(os.path.join(file_direc, h1_fn))
        h2 = obspy.read(os.path.join(file_direc, h2_fn))
        v = obspy.read(os.path.join(file_direc, v_fn))

        h1.merge(method=1, interpolation_samples=-1)
        h2.merge(method=1, interpolation_samples=-1)
        v.merge(method=1, interpolation_samples=-1)

        if trim_flag:
            h1 = TsProc.add_mseed_tool(h1)
            h2 = TsProc.add_mseed_tool(h2)
            v = TsProc.add_mseed_tool(v)

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

        """
        Function to process text files.

        Parameters
        ----------
        file_direc: string
            Directory where files are stored.
        h1_fn: string
            Filename of h1
        h2_fn: string
            Filename of h2
        v_fn: string
            Filename of v
        time_cut: int
            Indicates the amount of time to be cut in sec. Default = 300 sec.

        returns the component in array format.

        """

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
        mseed_flag = kwargs.get('mseed_flag', True)

        if mseed_flag:
            if file_type == 1:
                h1, h2, v, dt, time = TsProc.proc_mseed_data(file_direc=self.directory, h1_fn=self.h1,
                                                             h2_fn=self.h2, v_fn=self.v, trim_flag=trim_flag,
                                                             time_cut=time_cut)

            elif file_type == 2:
                h1, h2, v, dt, time = TsProc.proc_txt_data(file_direc=self.directory, h1_fn=self.h1,
                                                           h2_fn=self.h2, v_fn=self.v, time_cut=time_cut)
            else:
                raise ValueError("Invalid file type used. Use either 1 for mseed or 2 for text file")
        else:
            h1 = self.h1
            h2 = self.h2
            v = self.v

        return h1, h2, v

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

    @staticmethod
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

        t_front = kwargs.get('t_front', 5)
        t_end = kwargs.get('t_end', 5)
        sym = kwargs.get('sym', True)

        win = tukey(len(ts), alpha=((t_front + t_end) / 200), sym=sym)

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
            ts[i] = TsProc.detrend_ts(ts=ts[i], detrend_type=detrend_type, t_front=t_front, t_end=t_end)

            # Taper
            if taper_flag:
                ts[i] = TsProc.tukey_window(ts=ts[i], t_front=t_front, t_end=t_end)

            # Filter
            if filter_flag:
                ts[i], res = TsProc.apply_filter(ts=ts[i], fc=fc, dt=ts_dt, npole=npole, is_causal=is_causal,
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
        defaults = {'time_cut': 120, 'file_type': 1, 'trim_flag': False, 'mseed_flag': True, 'detrend_type': 1,
                    'taper_flag': True,
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
                                             trim_flag=kwargs['trim_flag'], mseed_flag=kwargs['mseed_flag'])

        # Pre-process noise data
        if kwargs['pre_filter_flag']:
            # Apply detrend
            det_h1 = TsProc.detrend_ts(ts=h1, detrend_type=kwargs['detrend_type'],
                                       t_front=kwargs['pre_filter_t_front'],
                                       t_end=kwargs['pre_filter_t_end'])
            det_h2 = TsProc.detrend_ts(ts=h2, detrend_type=kwargs['detrend_type'],
                                       t_front=kwargs['pre_filter_t_front'],
                                       t_end=kwargs['pre_filter_t_end'])
            det_v = TsProc.detrend_ts(ts=v, detrend_type=kwargs['detrend_type'], t_front=kwargs['pre_filter_t_front'],
                                      t_end=kwargs['pre_filter_t_end'])

            # Apply tukey window
            h1 = TsProc.tukey_window(ts=det_h1, t_front=kwargs['pre_filter_t_front'],
                                     t_end=kwargs['pre_filter_t_end'])

            h2 = TsProc.tukey_window(ts=det_h2, t_front=kwargs['pre_filter_t_front'],
                                     t_end=kwargs['pre_filter_t_end'])

            v = TsProc.tukey_window(ts=det_v, t_front=kwargs['pre_filter_t_front'], t_end=kwargs['pre_filter_t_end'])

            # Apply filtering of the time series
            if kwargs['pre_filter_hpass_fc'] is not None:
                h1, res = TsProc.apply_filter(ts=h1, fc=kwargs['pre_filter_hpass_fc'], dt=self.dt,
                                              npole=kwargs['pre_filter_npole_hp'],
                                              is_causal=kwargs['pre_filter_is_causal'],
                                              order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                h2, res = TsProc.apply_filter(ts=h2, fc=kwargs['pre_filter_hpass_fc'], dt=self.dt,
                                              npole=kwargs['pre_filter_npole_hp'],
                                              is_causal=kwargs['pre_filter_is_causal'],
                                              order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                v, res = TsProc.apply_filter(ts=v, fc=kwargs['pre_filter_hpass_fc'], dt=self.dt,
                                             npole=kwargs['pre_filter_npole_hp'],
                                             is_causal=kwargs['pre_filter_is_causal'],
                                             order_zero_padding=kwargs['pre_filter_order_zero_padding'])

            if kwargs['pre_filter_lpass_fc'] is not None:
                h1, res = TsProc.apply_filter(ts=h1, fc=kwargs['pre_filter_lpass_fc'], dt=self.dt,
                                              npole=kwargs['pre_filter_npole_lp'],
                                              is_causal=kwargs['pre_filter_is_causal'],
                                              order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                h2, res = TsProc.apply_filter(ts=h2, fc=kwargs['pre_filter_lpass_fc'], dt=self.dt,
                                              npole=kwargs['pre_filter_npole_lp'],
                                              is_causal=kwargs['pre_filter_is_causal'],
                                              order_zero_padding=kwargs['pre_filter_order_zero_padding'])

                v, res = TsProc.apply_filter(ts=v, fc=kwargs['pre_filter_lpass_fc'], dt=self.dt,
                                             npole=kwargs['pre_filter_npole_lp'],
                                             is_causal=kwargs['pre_filter_is_causal'],
                                             order_zero_padding=kwargs['pre_filter_order_zero_padding'])

        # split data into num_wins windows
        num_wins = int(np.floor(len(h1) * self.dt / kwargs['win_width']))
        npts_win = int(np.floor(kwargs['win_width'] / self.dt))
        npts_over = int(np.floor(kwargs['overlapping'] / self.dt))
        win_moving = npts_win - npts_over
        new_shape = (num_wins, win_moving)
        ts_wins = np.reshape(self.time[0:(win_moving * num_wins)], new_shape)
        h1_wins = np.reshape(h1[0:(win_moving * num_wins)], new_shape)
        h2_wins = np.reshape(h2[0:(win_moving * num_wins)], new_shape)
        v_wins = np.reshape(v[0:(win_moving * num_wins)], new_shape)

        h1_stalta, h2_stalta, v_stalta = [], [], []

        # Specify if the data is microtremor noise or earthquake noise
        if kwargs['is_noise']:
            # compute sta/lta
            if kwargs['sta_lta_flag']:
                short_term = int(np.floor(kwargs['short_term_len'] / self.dt))
                long_term = int(np.floor(kwargs['long_term_len'] / self.dt))
                sta_lta_moving = int(np.floor(kwargs['sta_lta_moving_term'] / self.dt))

                h1_stalta, h2_stalta, v_stalta = [], [], []
                for i in range(1, len(h1_wins)):
                    h1_stalta.append(TsProc.sta_lta_calc(ts=h1_wins[i], short_term=short_term, long_term=long_term,
                                                         moving_term=sta_lta_moving))
                    h2_stalta.append(TsProc.sta_lta_calc(ts=h2_wins[i], short_term=short_term, long_term=long_term,
                                                         moving_term=sta_lta_moving))
                    v_stalta.append(TsProc.sta_lta_calc(ts=v_wins[i], short_term=short_term, long_term=long_term,
                                                        moving_term=sta_lta_moving))

            # Post-processing
            if kwargs['filter_flag']:  # Apply filter
                if kwargs['hpass_fc'] is not None or kwargs['lpass_fc'] is not None:
                    if kwargs['hpass_fc'] is not None:  # Apply hpass
                        h1_wins = TsProc.pre_proc(ts=h1_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                  taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                  t_end=kwargs['t_end'],
                                                  filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                  npole=kwargs['npole_hp'],
                                                  is_causal=kwargs['is_causal'],
                                                  order_zero_padding=kwargs['order_zero_padding'])

                        h2_wins = TsProc.pre_proc(ts=h2_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                  taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                  t_end=kwargs['t_end'],
                                                  filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                  npole=kwargs['npole_hp'],
                                                  is_causal=kwargs['is_causal'],
                                                  order_zero_padding=kwargs['order_zero_padding'])

                        v_wins = TsProc.pre_proc(ts=v_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                 taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                 t_end=kwargs['t_end'],
                                                 filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                                 npole=kwargs['npole_hp'],
                                                 is_causal=kwargs['is_causal'],
                                                 order_zero_padding=kwargs['order_zero_padding'])

                    if kwargs['lpass_fc'] is not None:  # Apply lpass
                        h1_wins = TsProc.pre_proc(ts=h1_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                  taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                  t_end=kwargs['t_end'],
                                                  filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                                  npole=kwargs['npole_lp'],
                                                  is_causal=kwargs['is_causal'],
                                                  order_zero_padding=kwargs['order_zero_padding'])

                        h2_wins = TsProc.pre_proc(ts=h2_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
                                                  taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                                  t_end=kwargs['t_end'],
                                                  filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                                  npole=kwargs['npole_lp'],
                                                  is_causal=kwargs['is_causal'],
                                                  order_zero_padding=kwargs['order_zero_padding'])

                        v_wins = TsProc.pre_proc(ts=v_wins, ts_dt=self.dt, detrend_type=kwargs['detrend_type'],
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

        return self.time, h1, h2, v, ts_wins, h1_wins, h2_wins, v_wins, h1_stalta, h2_stalta, v_stalta

    def select_windows(self, h1_wins, h2_wins, v_wins, idx_select, cols=None,
                       sta_lta_flag=False, h1_stalta=None, h2_stalta=None, v_stalta=None):

        """

        function which plots the  time series windows.

        Parameters
        ----------
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

        while True:
            plt.figure(figsize=(8, 6))

            for i, i_plot in enumerate(idx_select):
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
                        range_h1 = max(h1_wins[i_plot]) - min(h1_wins[i_plot])
                        plt.text(np.mean(t_seq[idx]), min(h1_wins[i_plot][idx]) + range_h1 * 0.2,
                                 round(min(h1_stalta[i]), 1))
                        plt.text(np.mean(t_seq[idx]), max(h1_wins[i_plot][idx]) - range_h1 * 0.2,
                                 round(max(h1_stalta[i_plot]), 1), color='black')

                    if i_plot < len(h2_stalta):
                        plt.subplot(312)
                        range_h2 = max(h2_wins[i_plot]) - min(h2_wins[i_plot])
                        plt.text(np.mean(t_seq[idx]), min(h2_wins[i_plot][idx]) + range_h2 * 0.2,
                                 round(min(h2_stalta[i]), 1), color='black')
                        plt.text(np.mean(t_seq[idx]), max(h2_wins[i_plot][idx]) - range_h2 * 0.2,
                                 round(max(h2_stalta[i_plot]), 1), color='black')

                    if i_plot < len(v_stalta):
                        plt.subplot(313)
                        range_v = max(v_wins[i_plot]) - min(v_wins[i_plot])
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

            idx_select.remove(p_index)

            plt.show()

        plt.close()

        return idx_select

    def td_plt_select(self, h1_wins, h2_wins, v_wins, sta_lta_flag=False, h1_stalta=None,
                      h2_stalta=None, v_stalta=None):

        """

        function which plots the time series windows for the main function hv_proc.

        Parameters
        ----------
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

        idx_select = self.select_windows(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, idx_select=idx_select,
                                         cols=cols, sta_lta_flag=sta_lta_flag, h1_stalta=h1_stalta, h2_stalta=h2_stalta,
                                         v_stalta=v_stalta)

        return idx_select

    def ts_proc(self, **kwargs):

        """
        Function that processes microtremor time series.

        Parameters
        ----------
        **kwargs
            is_noise: boolean
                Indicates whether the data is microtremor (True) or earthquake (False) data. default = True.
            eqk_filepath: string
                Indicates the filepath where earthquake data is stored. Default = None.
            output_pf_flnm: string
                Indicates the output filename.
            time_cut: int
                Integer representing the amount cut from the time series in seconds Default = 120 (sec)
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
            plot_ts: boolean
                Specifies whether to plot the time series for window selection, default = True.
            output_selected_ts: boolean
                Indicates whether to output the selected time series.
            output_removed_ts: boolean
                Indicates whether to output the removed time series.

            returns the windowed time series after pre- and post-processing in units of counts.

        """

        # Set default values for all parameters
        defaults_ts_proc = {'time_cut': 120, 'file_type': 1, 'trim_flag': False, 'mseed_flag': True, 'detrend_type': 1,
                            'taper_flag': True, 't_front': 10, 't_end': 10, 'pre_filter_flag': True,
                            'pre_filter_t_front': 10, 'pre_filter_t_end': 10, 'pre_filter_hpass_fc': 0.042,
                            'pre_filter_npole_hp': -5, 'pre_filter_is_causal': False,
                            'pre_filter_order_zero_padding': 0, 'pre_filter_lpass_fc': None,
                            'pre_filter_npole_lp': None, 'is_noise': True, 'win_width': 300, 'overlapping': 0,
                            'sta_lta_flag': False, 'short_term_len': 1, 'long_term_len': 30, 'sta_lta_moving_term': 1,
                            'filter_flag': False, 'hpass_fc': 0.0083, 'npole_hp': -5, 'lpass_fc': None,
                            'npole_lp': None, 'eqk_filepath': None, 'is_causal': False, 'order_zero_padding': 0}

        # Update default values with user-provided values
        updated_kwargs = {**defaults_ts_proc, **kwargs}

        time, h1, h2, v, ts_wins, h1_wins, h2_wins, v_wins, h1_stalta, h2_stalta, v_stalta = \
            self.pre_process_noise_data(**updated_kwargs)

        num_wins = len(h1_wins)

        if kwargs.get('plot_ts', False):
            idx_select = self.td_plt_select(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins,
                                            sta_lta_flag=kwargs.get('sta_lta_flag', False), h1_stalta=h1_stalta,
                                            h2_stalta=h2_stalta, v_stalta=v_stalta)
        else:
            idx_select = list(range(num_wins))

        time_ts = time

        ts_results = {'dt': self.dt, 'idx_select': idx_select, 'time': time, 'h1': h1,
                      'h2': h2, 'v': v, 'ts_wins': ts_wins, 'h1_wins': h1_wins, 'h2_wins': h2_wins, 'v_wins': v_wins,
                      'h1_stalta': h1_stalta, 'h2_stalta': h2_stalta, 'v_stalta': v_stalta}

        print("Time-domain selection is DONE!")

        # time-domain data output
        if len(idx_select) == 0:
            raise ValueError('No window is selected, please try different data!')
        else:

            if kwargs.get('output_selected_ts', False):
                outputflname_ts_sel = kwargs.get('output_pf_flnm', 'Test_') + 'ts_sel.csv'
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

            if kwargs.get('output_removed_ts', False):
                outputflname_ts_unsel = kwargs.get('output_pf_flnm', 'Test_') + 'ts_unsel.csv'
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

        return updated_kwargs, ts_results
