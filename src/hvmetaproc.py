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

"""Class definition for HvMetaProc object."""

import pandas as pd
import numpy as np

__all__ = ['HvMetaProc']


class HvMetaProc:

    """
         This class is developed for processing and
         generating metadata from HVSR processing.

    """

    def __init__(self, ts_def_list, fd_def_list, ts_list, fd_list, output_dir):

        """

        Parameters
        ----------

        ts_def_list: list
            List of all time series defaults from tsproc
        fd_def_list: list
            List of all hvsr defaults from hvsproc
        ts_list:  list
            list of outputs from tsproc
        fd_list: list
            list of outputs from hvproc
        output_dir: string
            directory to output metadata into.

        """

        self.dt = ts_list['dt']
        self.pre_filter_flag = ts_def_list['pre_filter_flag']
        self.pre_filter_hpass_fc = ts_def_list['pre_filter_hpass_fc']
        self.pre_filter_t_front = ts_def_list['pre_filter_t_front']
        self.detrend_type = ts_def_list['detrend_type']
        self.win_width = ts_def_list['win_width']
        self.overlapping = ts_def_list['overlapping']
        self.t_front = ts_def_list['t_front']
        self.t_end = ts_def_list['t_end']
        self.filter_flag = ts_def_list['filter_flag']
        self.hpass_fc = ts_def_list['hpass_fc']
        self.is_noise = ts_def_list['is_noise']
        self.ko_smooth_b = fd_def_list['ko_smooth_b']
        self.horizontal_comb = fd_def_list['horizontal_comb']
        self.distribution = fd_def_list['distribution']
        self.idx_select = ts_list['idx_select']
        self.h1 = ts_list['h1']
        self.h1_wins = ts_list['h1_wins']

        self.iidx_select = fd_list['idx_select']

        self.output_dir = output_dir

    def gen_meta(self, **kwargs):

        """
        Function that generate metadata output from hvsr processing.

        Parameters
        ----------
        **kwargs
            output_pf_flnm: string
                Indicates the pre-fix to each filename.
            output_metadata: boolean
                Indicates whether to output the metadata after processing.

        returns metadata output and .csv file.

        """
        output_pf_flnm = kwargs.get('output_pf_flnm', 'Test_')

        num_wins = len(self.h1_wins)

        # Output Metadata
        if kwargs.get('output_metadata', True):
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
            meta_output.loc[0] = [1 / self.dt,
                                  np.round(len(self.h1) * self.dt / 60, 4), 1 if self.pre_filter_flag else 0,
                                  1 if self.pre_filter_flag else 0, 'Butterworth', self.pre_filter_hpass_fc, 'Tukey',
                                  self.pre_filter_t_front,
                                  'no detrend' if self.detrend_type == 0 else 'mean removal' if
                                  self.detrend_type == 1 else 'linear detrend' if
                                  self.detrend_type == 2 else 'fifth order polynomial detrend',
                                  self.win_width, self.overlapping, 'Tukey', self.t_front, self.t_end,
                                  self.horizontal_comb, num_wins,
                                  len(self.iidx_select), 0 if self.filter_flag and not np.isnan(self.hpass_fc) else 1,
                                  self.hpass_fc, 'Butterworth', 'KonnoOhmachi', self.ko_smooth_b,
                                  0 if self.is_noise else 1,
                                  self.distribution, None]
            meta_output.to_csv(self.output_dir + '/' + outputflname_meta, index=False)

            print("Everything is DONE, check out the results in the output folder!")

            return meta_output
