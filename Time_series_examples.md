# tsproctools

A tool containing multiple functions which perform time series processing.

## detrend_ts

A function which applies detrending to time series

`detrend_ts(ts, **kwargs)`

Detrends a time series based on the specified method.

`Parameters`

**ts** (ndarray) - The input time series.

`kwargs`:

**detrend_type** (int):
- 0: No detrending.
- 1: Weighted mean detrending. (default)
- 2: Linear detrending.
- 6: 5th degree polynomial detrending.

**t_front** (int) - Percentage of data to be tapered at the beginning for Tukey window. Default is 5.

**t_end** (int) - Percentage of data to be tapered at the end for Tukey window. Default is 5.

`returns:`

Array of detrended time series.

## tukey_window

Applies Tukey window to the given time series.

`tukey_window(ts, **kwargs)`

`Parameters:`

**ts** (ndarray) - The input time series.

`kwargs`:

**t_front** (int) -Percentage of data to be tapered at the beginning for Tukey window. Default is 5.

**t_end** (int) - Percentage of data to be tapered at the end for Tukey window. Default is 5.

**sym** (boolean) - generates a symmetric window.

`returns:`

An array of the tapered time series.

## apply_filter

Calculate filter based on given parameters.

`apply_filter(ts, **kwargs)`

`Parameters:`

**ts** (ndarray) - An array representing the time series data.

`kwargs:`

**ts_dt** (float) - Time step of ts

**fc** (float) - The filtering frequency to be applied.

**npole** (int) - A value specifying low pass or high pass, a value less than 0 applies to a high pass filter, default = -5.

**is_causal** (boolean) - Indicates whether to apply a causal or acausal filter to the time series.

**order_zero_padding** (int) - Indicates what order of zero padding to be applied.

`returns`

**ts** (ndarray) - Time series data after applying the filter.

**res**  (dict) - A dictionary of results from computing the fft of the time series, containing:

- flt_ts: filtered time series
- flt_amp: amplitude of filtered FAS
- flt_resp: filter response
- flt_fft: filtered fft containing imaginary values

## apply_normalization

A function to normalize time series.

`apply_normalization(ts)`

`Parameters:`
  
**ts** (ndarray) - Time series array.

`returns:` 

normalized time series.

## sta_lta_calc

Function that computes the sta/lta ratio.

`sta_lta_calc(ts, **kwargs)`

`Parameters:`

**ts** (ndarray) - Time series array of microtremor/earthquake amplitudes

`kwargs:`

**short_term** (int) - Short term average length

**long_term** (int) - Long term average length

**moving_term** (int) - Moving term of the sta/lta length

`returns:` 

The ratio of the sta and lta

## split_into_windows

Function to split time series into individual discrete windows.

`split_into_windows(ts, dt, win_width, overlapping)`

`Parameters:`

**ts** (ndarray) - Array of time series data.

**dt** (float) - Time step of time series.

**win_width** (int) - Length of each individual window.

**overlapping** (int) - Length of overlapping between each window.

`returns:` 

A windowed time series





