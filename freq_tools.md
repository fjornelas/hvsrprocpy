# _fdt_

---

This contains a set of functions that 
can be used to process Fourier Amplitude Spectra (FAS)
and Horizontal-to-Vertical Spectral Ratio (HVSR), such as smoothing
and performing Fast Fourier Transforms (FFT).

The definitions of these functions are below:

## _compute_horizontal_combination_

Function to combine horizontal components

`compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='geometric_mean')`

`Parameters:`

**h1_sub** (ndarray) - Windowed array of h1 component

**h2_sub** (ndarray) - Windowed array of h2 component

**dt** (float) - Time step of time series

**horizontal_comb** (string) - Specifies which horizontal combination to apply to time series.

`returns:` 

combined horizontal windowed array.

## _smoothed_fas_ko_


Function applies Konno and Ohmachi smoothing to a fourier amplitude spectra(FAS)

`smoothed_fas_ko(fas, f, fc, b=40)`

`Parameters:`

**f** (ndarray) - Frequency array of the fft.

**fas** (ndarray) - Amplitude of the fft.

**fc** (ndarray) - Resampled frequency in which to apply smoothing

**b** (int) - Smoothing parameter, default = 40.

`returns:`

A smoothed FAS using KO smoothing.

## _smooth_fas_

Function that applies either Konno and Ohmachi (KO) Smoothing or Parzen Smoothing to a windowed time series

`smooth_fas(fc, fas, freq, ko_smooth_flag=True, parzen_flag=False, ko_smooth_b=40, parzen_bwidth=1.5)`

`Parameters:`

**fc** (ndarray) - Resampled frequencies to which apply smoothing.

**fas** (ndarray) - Amplitude of the fft.

**freq** (ndarray) - Frequency of the fft.

**ko_smooth_flag** (boolean) - Specifies whether to apply KO smoothing, default = True

**parzen_flag** (boolean) - Specifies whether to apply parzen smoothing, default = False

**ko_smooth_b** (int) - Smoothing parameter to apply KO smoothing, a smaller value will apply more smoothing, default = 40.

**parzen_bwidth** (float) - Smoothing parameter to apply parzen smoothing, a larger value to apply more smoothing, default = 1.5.

`returns:`

The smoothed FAS.

## _fas_cal_

Function that computes the FAS for a time series.

`fas_cal(ts, dt=0.005, max_order=0)`

`Parameters:`

**ts** (ndarray) - Array of  time series.

**dt** (float) - time step of time series.

**max_order** (int) - order in which to compute the number of points for the fft.

`returns:` 

A dictionary of outputs such as the time series, fas amp, fas freq, and phase.
