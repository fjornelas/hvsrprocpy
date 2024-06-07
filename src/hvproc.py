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

from parzenpy.parzen_smooth import parzenpy

__all__ = ['HvsrProc']


class HvsrProc:
    """
     This class is developed for processing and
     calculating HVSR from noise-based
     and earthquake-based time series data.

     """

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
            msg = f"{name} must be a windowed Numpy 1D array, not a {type(array_type)}."
            raise TypeError(msg)
        return array_type

    def __init__(self, h1_wins, h2_wins, v_wins, dt, idx_select_ts, output_dir):
        """

        Parameters
        ----------

        h1_wins: ndarray
            Windowed array of time series data.
        h2_wins: ndarray
            Windowed array of time series data.
        v_wins: ndarray
            Windowed array of time series data.
        dt: float
            Time step of time series.
        idx_select_ts: list
            Indicies of each window selected from tsproc.
        output_dir: string
            Directory for which to output .csv files.

        """

        self.h1_wins = HvsrProc._check_array("horizontal_1 windowed", h1_wins)
        self.h2_wins = HvsrProc._check_array("horizontal_2 windowed", h2_wins)
        self.v_wins = HvsrProc._check_array("vertical_windowed", v_wins)
        self.dt = dt
        self.idx_select_ts = idx_select_ts
        self.output_dir = output_dir

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
        with np.errstate(divide='ignore', invalid='ignore'):
            smooth_fas = np.divide(np.sum(weights * fas[:, np.newaxis], axis=2), np.sum(weights, axis=1))
        return smooth_fas

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
            smooth = parzenpy(freq=freq, fft=fas)
            smooth_fas = smooth.apply_smooth(fc=fc, b=parzen_bwidth, windowed_flag=True)

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

    def hvsr_win_calc(self, **kwargs):

        """

        function which computes the Horizontal-to-Vertical Spectral Ratio (HVSR)
        from each individual windowed time series.

        Parameters
        ----------

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

        defaults = {'ko_smooth_b': 40, 'ko_smooth_flag': True,
                    'parzen_flag': False, 'parzen_bwidth': 1.5, 'horizontal_comb': 'ps_RotD50',
                    'freq_hv_mean': None, 'polar_curves_flag': False, 'freq_polar': None,
                    'deg_increment': 10, 'sjb_avg': False}

        kwargs = {**defaults, **kwargs}

        res = {}

        h1_sub = self.h1_wins
        h2_sub = self.h2_wins
        v_sub = self.v_wins

        if kwargs['horizontal_comb'] == 'ps_RotD50':
            # Implementation for 'ps_RotD50'
            fas_h1 = np.fft.rfft(h1_sub)
            fas_h2 = np.fft.rfft(h2_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), self.dt)
            fas_h = HvsrProc.ang_pga_rotd50_calc(fft1=fas_h1, fft2=fas_h2, freq=freq, num_wins=len(self.h1_wins))
        elif kwargs['horizontal_comb'] == 'squared_average':
            # Implementation for 'squared_average'
            fas_h1 = np.fft.rfft(h1_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), self.dt)
            fas_h2 = np.fft.rfft(h2_sub)
            fas_h = np.sqrt((np.abs(fas_h1) ** 2 + np.abs(fas_h2) ** 2) / 2)
        elif kwargs['horizontal_comb'] == 'geometric_mean':
            # Implementation for 'geometric_mean'
            fas_h1 = np.fft.rfft(h1_sub)
            freq = np.fft.rfftfreq(len(h1_sub[0]), self.dt)
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
            freq = np.fft.rfftfreq(len(v_sub[0]), self.dt)
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

        print("Preparing for frequency-domain, please wait...")

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

        print("Frequency-domain selection is DONE!")

        return res

    def hv_proc(self, **kwargs):

        """
        Function that processes windowed time series to compute HVSR and FAS spectra.

        Parameters
        ----------

        **kwargs
            output_pf_flnm: string
                Indicates the output filename. Default = 'Test_'
            distribution: string
                Indicates the type  of distribution. normal or log_normal, default = normal.
            robust_est: boolean
                Indicates whether robust_est is to be used. Default = False.
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
                Indicates the number of points to resample the data. Default = 2000
            plot_hvsr: boolean
                Specifies whether to plot the FAS and HVSR for window selection, default = True.
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

        returns the mean and selected HVSR and FAS for all 3-components, also polar curve data.

        """

        output_pf_flnm = kwargs.get('output_pf_flnm', 'Test_')
        output_freq_min = kwargs.get('output_freq_min', 0.01)
        output_freq_max = kwargs.get('output_freq_max', 50)
        deci_polar_factor = kwargs.get('deci_polar_factor', 10)
        deg_increment = kwargs.get('deg_increment', 10)
        distribution = kwargs.get('distribution', 'normal')

        min_win_len = min([len(win) for win in self.h1_wins])
        n_nyq_min = min_win_len // 2 + 1
        df = 1 / (min_win_len * self.dt)
        freq = np.arange(1, n_nyq_min) * df

        if freq[0] == 0:
            freq = freq[1:]
        if kwargs.get('resample_lin2log', True):
            freq = np.logspace(np.log10(min(freq)), np.log10(max(freq)), num=kwargs.get('resampling_length', 2000))
        if kwargs.get('deci_mean_factor', 10) > 0:
            freq_hv_mean = freq[::int(np.floor(kwargs.get('deci_mean_factor', 10)))]
        else:
            freq_hv_mean = freq

        defaults_hv_proc = {'ko_smooth_b': 40, 'ko_smooth_flag': True, 'parzen_flag': False,
                            'parzen_bwidth': 1.5, 'horizontal_comb': 'geometric_mean', 'freq_hv_mean': freq_hv_mean,
                            'polar_curves_flag': False, 'freq_polar': None, 'deg_increment': 10,
                            'distribution': 'normal', 'robust_est': False, 'plot_hvsr': True, 'sjb_avg': False}

        updated_kwargs = {**defaults_hv_proc, **kwargs}

        hvsr_list = self.hvsr_win_calc(**updated_kwargs)

        fd_select = HvsrProc.fd_plt_select(hvsr_list=hvsr_list, robust_est=updated_kwargs['robust_est'],
                                           freq_hv_mean=freq_hv_mean,
                                           distribution=updated_kwargs['distribution'],
                                           plot_hvsr=updated_kwargs['plot_hvsr'],
                                           sjb_avg=updated_kwargs['sjb_avg'])

        iidx_select = fd_select['idx_select']
        # Frequency-domain data output
        if len(iidx_select) == 0:
            raise ValueError('No window is selected, please try different data!')
        else:
            if kwargs.get('output_selected_hvsr', True):
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

            if kwargs.get('output_mean_curve', True):
                outputflname_hvsr_mean = output_pf_flnm + 'hvsr_mean.csv'
                if kwargs.get('robust_est', False):
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

            if kwargs.get('output_fas_mean_curve', True):
                outputflname_fas_mean = output_pf_flnm + 'FAS_mean.csv'
                if kwargs.get('robust_est', False):
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

            if kwargs.get('output_removed_hvsr', True):
                if len(self.idx_select_ts) > len(iidx_select):
                    idx_remove = [i for i in range(len(self.idx_select_ts)) if self.idx_select_ts[i] not in iidx_select]

                    outputflname_hvsr_unsel = output_pf_flnm + 'hvsr_unsel.csv'
                    hvsr_unsel_out = np.full((len(freq_hv_mean), len(idx_remove) + 1), np.nan)
                    hvsr_unsel_out[:, 0] = freq_hv_mean

                    removed_windows = []

                    for i, idx in enumerate(idx_remove):
                        hv_ratio = hvsr_list['hv_ratio'][idx]
                        if hv_ratio is not None:
                            hvsr_unsel_out[:, i + 1] = hv_ratio
                            removed_windows.append(self.idx_select_ts[idx])

                    col_names = ['freq_Hz'] + [f'HVSR_{idx}' for idx in
                                               removed_windows]

                    hvsr_unsel_out = pd.DataFrame(np.round(hvsr_unsel_out, 5), columns=col_names)

                    if not np.isnan(output_freq_min) and output_freq_min > np.min(hvsr_unsel_out['freq_Hz']):
                        hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] >= output_freq_min]

                    if not np.isnan(output_freq_max) and output_freq_max < np.max(hvsr_unsel_out['freq_Hz']):
                        hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] <= output_freq_max]

                    hvsr_unsel_out.to_csv(os.path.join(self.output_dir, outputflname_hvsr_unsel), index=False)

            # Generate polar curves and output
            if kwargs.get('output_polar_curves', False):
                print("Calculating and generating polar curve data, please wait......")
                outputflname_hvsr_polar = output_pf_flnm + 'hvsr_polar.csv'

                if deci_polar_factor > 0:
                    freq_polar = freq[::int(deci_polar_factor)]
                else:
                    freq_polar = freq

                defaults_polar = {'ts_dt': 0.005, 'ko_smooth_b': 40, 'ko_smooth_flag': True, 'parzen_flag': False,
                                  'parzen_bwidth': 1.5, 'horizontal_comb': 'ps_RotD50', 'freq_hv_mean': freq_hv_mean,
                                  'polar_curves_flag': False, 'freq_polar': freq_polar, 'deg_increment': 10,
                                  'sjb_avg': False}

                polar_kwargs = {**defaults_polar, **kwargs}

                hvsr_list = self.hvsr_win_calc(**polar_kwargs)

                polar_degs = np.arange(0, 180, deg_increment)
                polar_hvsr_mat = np.empty((len(freq_polar), len(polar_degs) * 3), dtype=np.float64)
                tmp_hvsr_mat = np.empty((len(freq_polar), len(self.idx_select_ts)), dtype=np.float64)

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

        return updated_kwargs, fd_select
