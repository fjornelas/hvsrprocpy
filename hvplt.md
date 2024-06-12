# _hvplt_ (HVSR Plotting Tools)

This file contains a set of functions that 
can be used to plot HVSR processed data, such as mean curves, FAS curves, 
polar curves, and time series.

The definitions of these functions are below:

## _process_polar_curve_

Function which processes polar curves to be in a specific format to plot using _plot_polar_ratio_.

`process_polar_curve(polar_data, azimuths, standard_freqs)`

`Parameters:`

**polar_data** (ndarray) - Array containing polar data.

**azimuths** (ndarray) - Array containing polar azimuths.

**standard_freqs** (ndarray) - Array containing the freqs to plot the polar data.

`returns:` 

Dataframe with an updated format.

## _plot_polar_ratio_

Function which plots the polar data.

`plot_polar_ratio(result_data)`

`Parameters:`

**result_data** (dataframe) - Dataframe containing polar ratio, azimuths, and frequencies.

`returns:` 

Plotted polar data

## _plot_mean_hvsr_

Function which plots the mean HVSR data.

`plot_mean_hvsr(csv_file_path, metadata_csv_path, xlim_m=65, ylim_m=4, xlim=(0.01, 50), ylim=(0, 5), robust_est=False, metadata_flag=True)`

`Parameters:`

**csv_file_path:** (string) - String indicating the path and filename to read mean.csv from.

**metadata_csv_path** (string) - String indicating the path and the filename to read metadata.csv.

**xlim_m** (int) - x-axis limit for metadata text, to be plotted adjacent to plot.

**ylim_m** (int) - y-axis limit for metadata text, to be plotted adjacent to plot.

**xlim** (range) - range of x limit.

**ylim** (range) - range of y limit.

**robust_est** (boolean) - Specifies whether robust_est was used to compute the HVSR. Default=False

**metadata_flag** (boolean) - Specifies whether to show the metadata as text near the plot. Default=True

`returns:`

Plotted mean HVSR data

## _plot_selected_time_series_

Function which plots the time series data.

`plot_selected_time_series(csv_file_path)`

`Parameters`

**csv_file_path** (string) - String indicating the path and filename to read .csv from.

`returns:` 

Plotted time series data

## _plot_selected_hvsr_

Function which plots the selected HVSR data.

`plot_selected_hvsr(csv_file_path, xlim=(0.1, 50), ylim=(0, 8))`

`Parameters:`

**csv_file_path** (string) - String indicating the path and filename to read .csv from.

**xlim** (range) - Range of x limit

**ylim** (range) - Range of y limit

`returns` 

Plotted selected HVSR data

## _plot_fas_

Function which plots the Fourier Amplitude Spectra (FAS) data.

`plot_fas(csv_path, xlim=(0.1, 50), ylim=(10e2, 10e8))`

`Parameters:`

**csv_path** (string) - String indicating the path and filename to read .csv from.

**xlim** (range) - Range of x limit

**ylim** (range) - Range of y limit

`returns:` 

Plotted FAS data