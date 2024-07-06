# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

"""Functions to compute HVSR and process time series."""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .tdt import *
from .fdt import *

__all__ = ['_win_proc', 'process_noise_data', '_select_windows', '_ts_plt_select',
           'hvsr_and_fas_calc', '_plt_hvsr', '_hvsr_plt_select', 'hvsr']


def _win_proc(ts, **kwargs):
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
        ts[i] = detrend_ts(ts=ts[i], detrend_type=detrend_type, t_front=t_front, t_end=t_end)

        # Taper
        if taper_flag:
            ts[i] = tukey_window(ts=ts[i], t_front=t_front, t_end=t_end)

        # Filter
        if filter_flag:
            ts[i], res = apply_filter(ts=ts[i], fc=fc, dt=ts_dt, npole=npole, is_causal=is_causal,
                                      order_zero_padding=order_zero_padding)

    return ts


def process_noise_data(ts, dt, **kwargs):
    """
    Processes the time series data, by pre-processing of the raw time series data,
    then processes the data within windows.

    Parameters
    ----------
    ts: ndarray
        Time series array
    dt: float
        Time step of time series

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
                'is_noise': True, 'win_width': 300, 'overlapping': 0, 'sta_lta_flag': False, 'short_term_len': 1,
                'long_term_len': 30, 'sta_lta_moving_term': 1, 'filter_flag': False, 'hpass_fc': 0.0083,
                'npole_hp': -5,
                'lpass_fc': None, 'npole_lp': None, 'eqk_filepath': None, 'is_causal': False,
                'order_zero_padding': 0, 'norm_flag': False}

    # Update default values with user-provided values
    kwargs = {**defaults, **kwargs}

    # Pre-process noise data
    if kwargs['pre_filter_flag']:
        # Apply detrend
        ts_det = detrend_ts(ts=ts, detrend_type=kwargs['detrend_type'],
                            t_front=kwargs['pre_filter_t_front'],
                            t_end=kwargs['pre_filter_t_end'])

        # Apply tukey window
        ts_tukey = tukey_window(ts=ts_det, t_front=kwargs['pre_filter_t_front'],
                                t_end=kwargs['pre_filter_t_end'])

        # Apply filtering of the time series
        if kwargs['pre_filter_hpass_fc'] is not None:
            ts, res = apply_filter(ts=ts_tukey, fc=kwargs['pre_filter_hpass_fc'], dt=dt,
                                   npole=kwargs['pre_filter_npole_hp'],
                                   is_causal=kwargs['pre_filter_is_causal'],
                                   order_zero_padding=kwargs['pre_filter_order_zero_padding'])

        if kwargs['pre_filter_lpass_fc'] is not None:
            ts, res = apply_filter(ts=ts_tukey, fc=kwargs['pre_filter_lpass_fc'], dt=dt,
                                   npole=kwargs['pre_filter_npole_lp'],
                                   is_causal=kwargs['pre_filter_is_causal'],
                                   order_zero_padding=kwargs['pre_filter_order_zero_padding'])
    else:
        pass

    # Normalize time series
    if kwargs['norm_flag']:
        ts = apply_normalization(ts)
    else:
        pass

    # split data into num_wins windows
    ts_wins = split_into_windows(ts, dt, kwargs['win_width'], kwargs['overlapping'])

    ts_stalta = []

    # Specify if the data is microtremor noise or earthquake noise
    if kwargs['is_noise']:
        # compute sta/lta
        if kwargs['sta_lta_flag']:
            short_term = int(np.floor(kwargs['short_term_len'] / dt))
            long_term = int(np.floor(kwargs['long_term_len'] / dt))
            sta_lta_moving = int(np.floor(kwargs['sta_lta_moving_term'] / dt))

            ts_stalta = []
            for i in range(1, len(ts_wins)):
                ts_stalta.append(sta_lta_calc(ts=ts_wins[i], short_term=short_term, long_term=long_term,
                                              moving_term=sta_lta_moving))

        # Post-processing
        if kwargs['filter_flag']:  # Apply filter
            if kwargs['hpass_fc'] is not None or kwargs['lpass_fc'] is not None:
                if kwargs['hpass_fc'] is not None:
                    ts_wins = _win_proc(ts=ts_wins, ts_dt=dt, detrend_type=kwargs['detrend_type'],
                                        taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                        t_end=kwargs['t_end'], filter_flag=kwargs['filter_flag'], fc=kwargs['hpass_fc'],
                                        npole=kwargs['npole_hp'], is_causal=kwargs['is_causal'],
                                        order_zero_padding=kwargs['order_zero_padding'])

                if kwargs['lpass_fc'] is not None:
                    ts_wins = _win_proc(ts=ts_wins, ts_dt=dt, detrend_type=kwargs['detrend_type'],
                                        taper_flag=kwargs['taper_flag'], t_front=kwargs['t_front'],
                                        t_end=kwargs['t_end'], filter_flag=kwargs['filter_flag'], fc=kwargs['lpass_fc'],
                                        npole=kwargs['npole_lp'], is_causal=kwargs['is_causal'],
                                        order_zero_padding=kwargs['order_zero_padding'])

            else:
                pass
        else:
            pass

    # Assemble corrected earthquake strong motions
    if not kwargs['is_noise']:
        eqk_data = np.genfromtxt(kwargs['eqk_filepath'], delimiter=',')
        num_wins = int(eqk_data.shape[1] / 4)
        ts_wins = [None] * num_wins
        for i in range(num_wins):
            ts_wins[i] = eqk_data[:, 1 + (i - 1) * 4][~np.isnan(eqk_data[:, 1 + (i - 1) * 4])]
        print("Assembling earthquake strong motion is DONE!")

    return ts, ts_wins, ts_stalta


def _select_windows(h1_wins, h2_wins, v_wins, dt, idx_select, cols=None,
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
    dt: float
        Time step of time series
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
            t_seq = np.arange(1, len(h1_wins[i_plot]) + 1) * dt + (i_plot * len(h1_wins[i_plot])) * dt
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


def _ts_plt_select(h1_wins, h2_wins, v_wins, dt, sta_lta_flag=False, h1_stalta=None,
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
    dt: float
        Time step of time series.
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

    idx_select = _select_windows(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, dt=dt, idx_select=idx_select,
                                 cols=cols, sta_lta_flag=sta_lta_flag, h1_stalta=h1_stalta, h2_stalta=h2_stalta,
                                 v_stalta=v_stalta)

    return idx_select


def hvsr_and_fas_calc(h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar, **kwargs):
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
    dt: float
        Time step of time series
    freq_hv_mean: ndarray
        Array of resampled frequencies.
    freq_polar: ndarray
        Array of resampled polar frequencies

    **kwargs:

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
        polar_curves_flag: boolean
            Indicates whether to compute the polar curve of the mean HSVR curve.
        deg_increment: int
            Integer indicating the increment in which the polar curve is to be computed.



    returns a list of the FAS values for each window, the HVSR curve for each window,
    and the Polar Ratio for each azimuth.

    """

    defaults = {'ko_smooth_b': 40, 'ko_smooth_flag': True,
                'parzen_flag': False, 'parzen_bwidth': 1.5, 'horizontal_comb': 'ps_RotD50',
                'polar_curves_flag': False, 'deg_increment': 10, 'sjb_avg': False}

    kwargs = {**defaults, **kwargs}

    res = {}

    h1_sub = h1_wins
    h2_sub = h2_wins
    v_sub = v_wins

    fas_h, freq = compute_horizontal_combination(h1_sub=h1_sub, h2_sub=h2_sub, dt=dt,
                                                 horizontal_comb=kwargs['horizontal_comb'])

    # Compute the vertical FAS
    fas_v = np.fft.rfft(v_sub)

    # Smooth both the Horizontal and the Vertical FAS
    h_smooth = smooth_fas(fc=freq_hv_mean, fas=fas_h,
                          freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                          parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                          parzen_bwidth=kwargs['parzen_bwidth'])

    v_smooth = smooth_fas(fc=freq_hv_mean, fas=fas_v, freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                          parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                          parzen_bwidth=kwargs['parzen_bwidth'])

    # Compute HVSR
    hv_ratio = np.divide(h_smooth, v_smooth, out=np.zeros_like(h_smooth), where=v_smooth != 0)

    # Compute the smoothed horizontal components
    fas_h1 = np.fft.rfft(h1_sub)
    fas_h2 = np.fft.rfft(h2_sub)

    h1_smooth = smooth_fas(fc=freq_hv_mean, fas=fas_h1,
                           freq=freq, ko_smooth_flag=kwargs['ko_smooth_flag'],
                           parzen_flag=kwargs['parzen_flag'], ko_smooth_b=kwargs['ko_smooth_b'],
                           parzen_bwidth=kwargs['parzen_bwidth'])

    h2_smooth = smooth_fas(fc=freq_hv_mean, fas=fas_h2,
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
        h1_fft = np.fft.rfft(h1_sub)
        h2_fft = np.fft.rfft(h2_sub)

        fas_v = np.fft.rfft(v_sub)
        freq = np.fft.rfftfreq(len(h1_sub[0]), dt)
        v_smooth = smooth_fas(fc=freq_polar, fas=fas_v, freq=freq,
                              ko_smooth_flag=kwargs['ko_smooth_flag'], parzen_flag=kwargs['parzen_flag'],
                              ko_smooth_b=kwargs['ko_smooth_b'], parzen_bwidth=kwargs['parzen_bwidth'])
        hv_ratio_list = []
        for i in range(len(polar_degs)):
            angle_idx = polar_degs[i]
            fas_h = h1_fft * np.cos(np.radians(angle_idx)) + h2_fft * np.sin(np.radians(angle_idx))

            h_smooth = smooth_fas(fc=freq_polar, fas=fas_h, freq=freq,
                                  ko_smooth_flag=kwargs['ko_smooth_flag'], parzen_flag=kwargs['parzen_flag'],
                                  ko_smooth_b=kwargs['ko_smooth_b'], parzen_bwidth=kwargs['parzen_bwidth'])

            hv_ratio = np.divide(h_smooth, v_smooth, out=np.zeros_like(h_smooth), where=v_smooth != 0)

            hv_ratio_list.append(hv_ratio)

        res['polar_hv_ratio'] = np.array(hv_ratio_list)

    return res


def _plt_hvsr(freq, hvsr_mat, idx_select, hvsr_mean, hvsr_sd, hvsr_sd1,
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


def _hvsr_plt_select(hvsr_list, robust_est=False, freq_hv_mean=None,
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
            _plt_hvsr(freq_hv_mean, hvsr_mat, idx_select, hvsr_mean, hvsr_sd, hvsr_sd1, h1_mat, h2_mat, v_mat,
                      h1_mat_mean, h2_mat_mean, v_mat_mean, robust_est=robust_est, sjb_avg=sjb_avg,
                      distribution=distribution)
    else:
        pass

    plt.show()

    res = {'idx_select': idx_select, 'hvsr_mean': hvsr_mean, 'hvsr_sd': hvsr_sd, 'FAS_h1_mean': h1_mat_mean,
           'FAS_h2_mean': h2_mat_mean, 'FAS_v_mean': v_mat_mean}

    if robust_est:
        res['hvsr_sd1'] = hvsr_sd1

    return res


def hvsr(h1, h2, v, dt, time_ts, output_dir, **kwargs):
    """

    Function that processes the time series data then
    computes the FAS and HVSR for the windowed time series data.

    Parameters
    ----------
    h1: ndarray
        Array of amplitudes from h1 component.
    h2: ndarray
        Array of amplitudes from h2 component.
    v: ndarray
        Array of amplitudes from v component.
    dt: float
        Time step of time series.
    time_ts: ndarray
        Array of time values from time series.
    output_dir: string
        Specifies where to output metadata and output .csv files.

    **kwargs

        is_noise: boolean
            Indicates whether the data is microtremor (True) or earthquake (False) data. Default = True.
        eqk_filepath: string
            Indicates the filepath where earthquake data is stored. Default = None.
        output_pf_flnm: string
            Indicates the output filename. Default = 'Test_'
        distribution: string
            Indicates the type  of distribution. normal or log_normal. Default = normal.
        robust_est: boolean
            Indicates whether robust_est is to be used. Default = False.
        time_cut: int
            Integer representing the amount cut from the time series in seconds. Default = 120 (sec).
        file_type: int
            Specifies whether the filetype is a mseed file (1) or text file (2). Default = 1 (mseed).
        trim_flag: boolean
            Specifies whether to use the miniseed tool to trim the data
            based on specific start time and end time. Default = False.
        pre_filter_flag: boolean
            Indicates whether pre-filtering is to be applied. Default = True.
        pre_filter_is_causal: boolean
            Indicates whether to apply a causal or acausal filter. Default = False indicating acausal.
        pre_filter_hpass_fc: float
            Indicates the high pass filtering value. Default = 0.042.
        pre_filter_lpass_fc: float
            Indicates the low pass filtering value. Default = None.
        pre_filter_npole_hp: int
            Indicates whether to apply high pass. Default = -5.
        pre_filter_npole_lp: int
            Indicates whether to apply low pass. Default = 4.
        pre_filter_order_zero_padding: int
            Indicates what order of zero padding to be applied. Default = 4.
        pre_filter_t_front: int
            Indicates the percentage of Tukey tapering be applied. Default = 10.
        pre_filter_t_end: int
            Indicates the percentage of Tukey tapering to be applied. Default = 10.
        filter_flag: boolean
            Indicates whether filtering is to be applied within windows.Default = True.
        is_causal: boolean
            Indicates whether a causal or acausal filter is to be applied. Default = False indicating acausal.
        hpass_fc: float
            Indicates the high pass filtering value to apply to each window. Default = 0.0083.
        lpass_fc: float
            Indicates the low pass filtering value to apply to each window. Default = None.
        npole_hp: int
            Indicates whether to apply a high pass filter. Default = -5.
        npole_lp: int
            Indicates whether to apply a low pass filter. Default = 4.
        order_zero_padding: int
            Indicates the order of zero padding to be applied to each window.
        detrend_type: int
            Indicates what type of detrend to be applied, 1 = mean removal, 2 = linear detrend,
            6 = 5th order polynomial detrend. default = 1.
        taper_flag: boolean
            Indicates whether to apply a Tukey windowing function. default = True.
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
            Indicates the smoothing factor to be used in KO smoothing. Default = 40.
        parzen_flag: boolean
            Indicates whether to apply parzen smoothing. Default = False.
        parzen_bwidth: float
            Indicates the smoothing factor to be used in parzen smoothing. Default = 1.5.
        win_width: int
            Indicates the length each window should be. Default = 300.
        overlapping: int
            Indicates the amount of overlapping to be applied within each window. Default = 0.
        sta_lta_flag: boolean
            Indicates whether to compute the short term and long term averages within each window. default = False.
        short_term_len: int
            Indicates the short term length. Default = 1.
        long_term_len: int
            Indicates the long term length. Default = 30.
        sta_lta_moving_term: int
            Indicates the sta/lta ratio term. Default = 1.
        deg_increment: int
            Indicates the degree increment for computing the polar curve.
        resample_lin2log: boolean
            Indicated whether to resample the frequencies. Default = True.
        deci_mean_factor: int
            Indicates the decimal mean factor to apply to resampled frequency. Default = 10.
        sjb_avg: boolean
            Specifies whether to take the average smoothed FAS before computing HVSR. Default = False.
        deci_polar_factor: int
            Indicates the decimal polar factor to apply to polar frequency. Default = 10.
        output_freq_min: float
            Indicates the output minimum frequency. Default =  0.01.
        output_freq_max: float
            Indicates the output maximum frequency. Default = 50.0.
        resampling_length: int
            Indicates the number of points to resample the data. Default = 2000.
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

    defaults = {'is_noise': True, 'eqk_filepath': None, 'output_pf_flnm': 'Test_', 'distribution': 'normal',
                'norm_flag': False, 'robust_est': False, 'time_cut': 120, 'file_type': 1, 'trim_flag': False,
                'pre_filter_flag': True,
                'pre_filter_is_causal': False, 'pre_filter_hpass_fc': 0.042, 'pre_filter_lpass_fc': None,
                'pre_filter_npole_hp': -5, 'pre_filter_npole_lp': 4, 'pre_filter_order_zero_padding': 0,
                'pre_filter_t_front': 10, 'pre_filter_t_end': 10, 'filter_flag': False, 'is_causal': False,
                'hpass_fc': 0.0083, 'lpass_fc': None, 'npole_hp': -5, 'npole_lp': 4, 'order_zero_padding': 0,
                'detrend_type': 1, 'taper_flag': True, 't_front': 10, 't_end': 10, 'horizontal_comb': 'geometric_mean',
                'ko_smooth_flag': True, 'ko_smooth_b': 40, 'parzen_flag': False, 'parzen_bwidth': 1.5, 'win_width': 300,
                'overlapping': 0, 'sta_lta_flag': False, 'short_term_len': 1, 'long_term_len': 30,
                'sta_lta_moving_term': 1,
                'deg_increment': 10, 'resample_lin2log': True, 'deci_mean_factor': 10, 'sjb_avg': False,
                'deci_polar_factor': 10,
                'output_freq_min': 0.01, 'output_freq_max': 50, 'resampling_length': 2000, 'plot_ts': True,
                'plot_hvsr': True,
                'output_selected_ts': False, 'output_removed_ts': False, 'output_selected_hvsr': True,
                'output_removed_hvsr': False,
                'output_mean_curve': True, 'output_polar_curves': False, 'output_fas_mean_curve': True,
                'output_metadata': True}

    updated_kwargs = {**defaults, **kwargs}

    ts_wins = split_into_windows(time_ts, dt, updated_kwargs['win_width'], updated_kwargs['overlapping'])

    h1, h1_wins, h1_stalta = process_noise_data(ts=h1, dt=dt, **updated_kwargs)
    h2, h2_wins, h2_stalta = process_noise_data(ts=h2, dt=dt, **updated_kwargs)
    v, v_wins, v_stalta = process_noise_data(ts=v, dt=dt, **updated_kwargs)

    print("Pre-processing noise data is DONE!")

    num_wins = len(h1_wins)

    if updated_kwargs['plot_ts']:
        idx_select = _ts_plt_select(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, dt=dt,
                                    sta_lta_flag=updated_kwargs['sta_lta_flag'], h1_stalta=h1_stalta,
                                    h2_stalta=h2_stalta, v_stalta=v_stalta)
    else:
        idx_select = list(range(num_wins))

    print("Time-domain selection is DONE!")

    # time-domain data output
    if len(idx_select) == 0:
        raise ValueError('No window is selected, please try different data!')
    else:

        if updated_kwargs['output_selected_ts']:
            outputflname_ts_sel = updated_kwargs['output_pf_flnm'] + 'ts_sel.csv'
            max_win_len = max([len(win) for win in h1_wins])

            idx_remove = [i for i in range(1, num_wins + 1) if i not in idx_select]

            ts_sel_out = np.full((max_win_len, 4 * len(idx_select)), np.nan)
            colnames = []
            for i, idx in enumerate(idx_select):
                start_col = 4 * i
                # end_col = start_col + 4
                if idx <= len(h1_wins) and idx not in idx_remove:
                    window_length = len(h1_wins[idx - 1]) * dt
                    start_time_idx = int(idx * window_length / dt)
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
            ts_sel_out_df.to_csv(os.path.join(output_dir, outputflname_ts_sel), index=False)

        if updated_kwargs['output_removed_ts']:
            outputflname_ts_unsel = updated_kwargs['output_pf_flnm'] + 'ts_unsel.csv'
            max_win_len = max([len(win) for win in h1_wins])

            # Initialize idx_remove
            idx_remove = [i for i in range(1, num_wins) if i not in idx_select]

            ts_unsel_out = np.full((max_win_len, 4 * len(idx_remove)), np.nan)
            colnames = []
            for i, idx in enumerate(idx_remove):
                start_col = 4 * i
                if idx <= len(h1_wins):
                    window_length = len(h1_wins[idx - 1]) * dt
                    start_time_idx = int(idx * window_length / dt)
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
            ts_unsel_out_df.to_csv(os.path.join(output_dir, outputflname_ts_unsel), index=False)

    print("Preparing for frequency-domain, please wait...")

    min_win_len = min([len(win) for win in h1_wins])
    n_nyq_min = min_win_len // 2 + 1
    df = 1 / (min_win_len * dt)
    freq = np.arange(1, n_nyq_min) * df

    if freq[0] == 0:
        freq = freq[1:]
    if updated_kwargs['resample_lin2log']:
        freq = np.logspace(-2, 2, num=updated_kwargs['resampling_length'])
    #         freq = np.logspace(np.log10(min(freq)), np.log10(max(freq)), num=updated_kwargs['resampling_length'])
    if updated_kwargs['deci_mean_factor'] > 0:
        freq_hv_mean = freq[::int(np.floor(updated_kwargs['deci_mean_factor']))]
    else:
        freq_hv_mean = freq

    hvsr_list = hvsr_and_fas_calc(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, dt=dt, freq_hv_mean=freq_hv_mean,
                                  freq_polar=None, **updated_kwargs)

    fd_select = _hvsr_plt_select(hvsr_list=hvsr_list, robust_est=updated_kwargs['robust_est'],
                                 freq_hv_mean=freq_hv_mean, distribution=updated_kwargs['distribution'],
                                 plot_hvsr=updated_kwargs['plot_hvsr'], sjb_avg=updated_kwargs['sjb_avg'])

    iidx_select = fd_select['idx_select']

    print("Frequency-domain selection is DONE!")

    # Frequency-domain data output
    if len(iidx_select) == 0:
        raise ValueError('No window is selected, please try different data!')
    else:
        if updated_kwargs['output_selected_hvsr']:
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

            outputflname_hvsr_sel = updated_kwargs['output_pf_flnm'] + 'hvsr_sel.csv'

            if updated_kwargs['output_freq_min'] is not None and updated_kwargs['output_freq_min'] > np.min(
                    hvsr_sel_out.iloc[:, 0]):
                idx_min = np.where(hvsr_sel_out.iloc[:, 0] >= updated_kwargs['output_freq_min'])[0]
                hvsr_sel_out = hvsr_sel_out.iloc[idx_min]

            if updated_kwargs['output_freq_max'] is not None and updated_kwargs['output_freq_max'] < np.max(
                    hvsr_sel_out.iloc[:, 0]):
                idx_max = np.where(hvsr_sel_out.iloc[:, 0] <= updated_kwargs['output_freq_max'])[0]
                hvsr_sel_out = hvsr_sel_out.iloc[idx_max]

            hvsr_sel_out.to_csv(output_dir + '/' + outputflname_hvsr_sel, index=False)

        if updated_kwargs['output_mean_curve']:
            outputflname_hvsr_mean = updated_kwargs['output_pf_flnm'] + 'hvsr_mean.csv'
            if updated_kwargs['robust_est']:
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
            if not np.isnan(updated_kwargs['output_freq_min']) and updated_kwargs['output_freq_min'] > np.min(
                    hvsr_mean_out['freq_Hz']):
                hvsr_mean_out = hvsr_mean_out[hvsr_mean_out['freq_Hz'] >= updated_kwargs['output_freq_min']]
            if not np.isnan(updated_kwargs['output_freq_max']) and updated_kwargs['output_freq_max'] < np.max(
                    hvsr_mean_out['freq_Hz']):
                hvsr_mean_out = hvsr_mean_out[hvsr_mean_out['freq_Hz'] <= updated_kwargs['output_freq_max']]
            hvsr_mean_out.to_csv(output_dir + '/' + outputflname_hvsr_mean, index=False)

        if updated_kwargs['output_fas_mean_curve']:
            outputflname_fas_mean = updated_kwargs['output_pf_flnm'] + 'FAS_mean.csv'
            if updated_kwargs['robust_est']:
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
            if not np.isnan(updated_kwargs['output_freq_min']) and updated_kwargs['output_freq_min'] > np.min(
                    fas_mean_out['freq_Hz']):
                fas_mean_out = fas_mean_out[fas_mean_out['freq_Hz'] >= updated_kwargs['output_freq_min']]
            if not np.isnan(updated_kwargs['output_freq_max']) and updated_kwargs['output_freq_max'] < np.max(
                    fas_mean_out['freq_Hz']):
                fas_mean_out = fas_mean_out[fas_mean_out['freq_Hz'] <= updated_kwargs['output_freq_max']]
            fas_mean_out.to_csv(output_dir + '/' + outputflname_fas_mean, index=False)

        if updated_kwargs['output_removed_hvsr']:
            if len(idx_select) > len(iidx_select):
                idx_remove = [i for i in range(len(idx_select)) if idx_select[i] not in iidx_select]

                outputflname_hvsr_unsel = updated_kwargs['output_pf_flnm'] + 'hvsr_unsel.csv'
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

                if not np.isnan(updated_kwargs['output_freq_min']) and updated_kwargs['output_freq_min'] > np.min(
                        hvsr_unsel_out['freq_Hz']):
                    hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] >= updated_kwargs['output_freq_min']]

                if not np.isnan(updated_kwargs['output_freq_max']) and updated_kwargs['output_freq_max'] < np.max(
                        hvsr_unsel_out['freq_Hz']):
                    hvsr_unsel_out = hvsr_unsel_out[hvsr_unsel_out['freq_Hz'] <= updated_kwargs['output_freq_max']]

                hvsr_unsel_out.to_csv(os.path.join(output_dir, outputflname_hvsr_unsel), index=False)

        # Generate polar curves and output
        if updated_kwargs['output_polar_curves']:
            print("Calculating and generating polar curve data, please wait......")
            outputflname_hvsr_polar = updated_kwargs['output_pf_flnm'] + 'hvsr_polar.csv'

            if updated_kwargs['deci_polar_factor'] > 0:
                freq_polar = freq[::int(updated_kwargs['deci_polar_factor'])]
            else:
                freq_polar = freq

            hvsr_list = hvsr_and_fas_calc(h1_wins=h1_wins, h2_wins=h2_wins, v_wins=v_wins, dt=dt,
                                          freq_hv_mean=freq_hv_mean,
                                          freq_polar=freq_polar, polar_curves_flag=True, **updated_kwargs)

            polar_degs = np.arange(0, 180, updated_kwargs['deg_increment'])
            polar_hvsr_mat = np.empty((len(freq_polar), len(polar_degs) * 3), dtype=np.float64)
            tmp_hvsr_mat = np.empty((len(freq_polar), len(idx_select)), dtype=np.float64)

            hvsr_data = hvsr_list['polar_hv_ratio']

            for i, deg in enumerate(polar_degs):
                tmp_hvsr_mat[:, :] = hvsr_data[i, :].T
                tmp_hvsr_mat[tmp_hvsr_mat <= 0] = 10e-5

                if updated_kwargs['distribution'] == 'normal':
                    tmp_mean = np.mean(tmp_hvsr_mat, axis=1)
                    tmp_sd = np.std(tmp_hvsr_mat, axis=1)
                elif updated_kwargs['distribution'] == 'log_normal':
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

            if not np.isnan(updated_kwargs['output_freq_min']):
                polar_hvsr_df = polar_hvsr_df[polar_hvsr_df['Freq_0'] >= updated_kwargs['output_freq_min']]

            if not np.isnan(updated_kwargs['output_freq_max']):
                polar_hvsr_df = polar_hvsr_df[polar_hvsr_df['Freq_0'] <= updated_kwargs['output_freq_max']]

            polar_hvsr_df.to_csv(os.path.join(output_dir, outputflname_hvsr_polar), index=False)

        # Output Metadata
        if updated_kwargs['output_metadata']:
            outputflname_meta = updated_kwargs['output_pf_flnm'] + 'metadata.csv'
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
            meta_output.loc[0] = [1 / dt, np.round(len(h1) * dt / 60, 4), 1 if updated_kwargs['pre_filter_flag'] else 0,
                                  1 if updated_kwargs['pre_filter_flag'] else 0, 'Butterworth',
                                  updated_kwargs['pre_filter_hpass_fc'],
                                  'Tukey', updated_kwargs['pre_filter_t_front'],
                                  'no detrend' if updated_kwargs['detrend_type'] == 0 else 'mean removal' if
                                  updated_kwargs['detrend_type'] == 1 else
                                  'linear detrend' if updated_kwargs[
                                                          'detrend_type'] == 2 else 'fifth order polynomial detrend',
                                  updated_kwargs['win_width'], updated_kwargs['overlapping'], 'Tukey',
                                  updated_kwargs['t_front'], updated_kwargs['t_end'],
                                  updated_kwargs['horizontal_comb'], num_wins,
                                  len(iidx_select), 0 if updated_kwargs['filter_flag'] and not np.isnan(
                    updated_kwargs['hpass_fc']) else 1,
                                  updated_kwargs['hpass_fc'], 'Butterworth', 'KonnoOhmachi',
                                  updated_kwargs['ko_smooth_b'], 0 if updated_kwargs['is_noise'] else 1,
                                  updated_kwargs['distribution'], None]
            meta_output.to_csv(output_dir + '/' + outputflname_meta, index=False)

    win_result = {'ts_wins': ts_wins, 'h1_wins': h1_wins, 'h2_wins': h2_wins, 'v_wins': v_wins,
                  'h1_stalta': h1_stalta, 'h2_stalta': h2_stalta, 'v_stalta': v_stalta, 'idx_select_ts': idx_select}

    print("Everything is DONE, check out the results in the output folder!")

    return win_result, fd_select
