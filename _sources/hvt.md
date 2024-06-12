# _hvt_ (HVSR Processing Tools) 

---

This file contains a set of functions that can be used for processing microtremor or earthquake
based data. The tools included can perform time series and frequency based analysis to compute the 
Fourier Amplitude Spectrum (FAS) and Horizontal-to-Vertical Spectral Ratio (HVSR) of the site.

The definitions of the functions are below:

## _process_noise_data_

Functions to process the time series data, by applying pre-processing of the raw time series
data, then processes the data within the windows.

`process_noise_data(ts, dt, **kwargs)`

`Parameters:`

**ts** (ndarray) - Time series array

**dt** - (float) - Time step of the time series

`kwargs:`

**detrend_type** (int) - An integer indicating the type of detrending to be applied, default = 1. 0 = No detrend, 1 = Mean_Removal, 2 = Linear Removal, 6 = 5th Order Polynomial.

**taper_flag** (boolean) - Indicates whether to apply tapering to the data. Default = True.

**t_front** (int) - Integer representing the percentage to taper on start of the time series data. Default = 5

**t_end** (int) - Integer representing the percentage to taper on end of the time series data. Default = 5

**pre_filter_flag** (boolean) - Indicates whether to apply filtering to the data. Default = True.

**pre_filter_t_front** (int) - Integer representing the percentage to taper on start of the time series data. Default = 5

**pre_filter_t_end**  (int) -Integer representing the percentage to taper on end of the time series data. Default = 5

**pre_filter_hpass_fc** (float) - The high pass filtering frequency to be applied.

**pre_filter_npole_hp** (int) - A value specifying high pass filter, a value less than 0 applies to a high pass filter, default = -5.

**pre_filter_is_causal** (boolean) - Indicates whether to apply a causal or acausal filter to the time series.

**pre_filter_order_zero_padding** (int) - Indicates what order of zero padding to be applied.

**pre_filter_lpass_fc** (float) - The low pass filtering frequency to be applied. default = None

**pre_filter_npole_lp** (int) - A value specifying low pass, a value greater than 0 applies to a low pass filter, default = None.

**is_noise** (boolean) - Indicates whether the data being processed is microtremor data, default = True. False indicates earthquake data.

**win_width** (int) - Integer representing the size of the window that the time series data will be partitioned into discrete windows.

**overlapping** (int) - Integer indicating the amount of overlap within each window.

**sta_lta_flag** (boolean) - Indicates whether a user wants to compute the short term (STA) or long term average (LTA) of each window of the time series.

**short_term_len** (int) - Integer representing the short term length. Default = 1.

**long_term_len** (int) - Integer representing the long term length. Default = 30

**sta_lta_moving_term** (int) - Integer representing the sta/lta ratio moving term. Default=1

**filter_flag** (boolean) - Indicates whether to apply filtering to the data. Default=False

**hpass_fc** (float) - The high pass filtering frequency to be applied. Default=0.0083

**npole_hp** (int) - A value specifying high pass filter, a value less than 0 applies to a high pass filter, default = -5.

**lpass_fc** (float) - The low pass filtering frequency to be applied.

**npole_lp** (int) - A value specifying low pass filter, a value greater than 0 applies to a low pass filter. Default = None.

**eqk_filepath** (string) - The directory from which to extract earthquake data.

**is_causal** (boolean) - Indicates whether to apply acausal to causal filtering, default = False.

**order_zero_padding** (int) - Indicates the order to which to apply zero padding to the data.

**norm_flag** (boolean) - Indicates whether to normalize time series prior to windowing. Default = False.

`returns:`

A windowed time series and the sta/lta ratio for each component.

## _hvsr_and_fas_calc_

Function which computes the Horizontal-to-Vertical Spectral Ratio (HVSR)
from each individual windowed time series.

`hvsr_and_fas_calc(h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar, **kwargs)`

`Parameters:`

**h1_wins** (ndarray) - Array of windowed time series data for the first horizontal component.

**h2_wins** (ndarray) - Array of windowed time series data for the second horizontal component.

**v_wins** (ndarray) - Array of windowed time series data for the vertical component.

**dt** (float) - Time step of time series

**freq_hv_mean** (ndarray) - Array of resampled frequencies.

**freq_polar** (ndarray) - Array of resampled polar frequencies

`kwargs:`

**ko_smooth_b** (int) - bandwidth value to which apply Konno and Ohmachi Smoothing. Default = 40.

**ko_smooth_flag** (boolean) - Indicates whether to apply Konno and Ohmachi Smoothing.

**parzen_flag** (boolean) - Indicates whether to apply to parzen smoothing.

**parzen_bwidth** (float) - bandwidth value to which apply parzen smoothing.

**horizontal_comb** (string) - Indicates the type of combination to be done to the horizontal components. Default = 'ps_RotD50'.There are three type of combinations, ps_RotD50 which computes the median of the azimuth of the two components, squared_average which computes the squared average between the two components, and geometric_mean which computes the geometric mean of the two components.

**polar_curves_flag** (boolean) - Indicates whether to compute the polar curve of the mean HSVR curve.

**deg_increment** (int) - Integer indicating the increment in which the polar curve is to be computed.

`returns:` 

A list of the FAS values for each window, the HVSR curve for each window,
and the polar ratio for each azimuth.

## _hvsr_

Function that processes the time series data then
computes the FAS and HVSR for the windowed time series data.

`hvsr(h1, h2, v, dt, time_ts, output_dir, **kwargs)`

`Parameters:`

**h1** (ndarray) - Array of amplitudes from h1 component.

**h2** (ndarray) - Array of amplitudes from h2 component.

**v** (ndarray) - Array of amplitudes from v component.

**dt** (float) - Time step of time series.

**time_ts** (ndarray) - Array of time values from time series.

**output_dir** (string) - Specifies where to output metadata and output .csv files.

`kwargs:`

Please refer to kwargs in functions process_noise_data and hvsr_and_fas_calc.

`returns:`

The windowed time series, sta/lta ratios, HVSR mean curve, mean FAS, selected/unselected time series,
selected/unselected HVSR curves, HVSR processing metadata and the polar curves from the microtremor data input.






