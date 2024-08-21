# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

"""Function definitions for HvsrPlot"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ['process_polar_curve', 'plot_polar_ratio', 'plot_mean_hvsr',
           'plot_selected_time_series', 'plot_selected_hvsr', 'plot_mean_fas']


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
    xlim_m: int
        x-axis limit for metadata text, to be plotted adjacent to plot.
    ylim_m: int
        y-axis limit for metadata text, to be plotted adjacent to plot.
    xlim: range
        range of x limit.
    ylim: range
        range of y limit.
    robust_est: boolean
        Specifies whether robust_est was used to compute the HVSR. Default=False
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


def plot_mean_fas(csv_path, xlim=(0.1, 50), ylim=(10e2, 10e8)):
    """

   Function which plots the Fourier Amplitude Spectra (FAS) data.

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


def plot_sel_fas(csv_path, h1_col='pink', h2_col='lightgreen', v_col='lightblue',
                 xlim=(0.1, 50), ylim=(10e2, 10e8)):
    """

       Function which plots the selected Fourier Amplitude Spectra (FAS) data.

       Parameters
       ----------

       csv_path: string
           String indicating the path and filename to read .csv from.
       h1_col: string
           Indicates the color the h1 component.
       h2_col: string
           Indicates the color the h2 component.
       v_col: string
           Indicates the color the v component.
       xlim: range
           Range of x limit
       ylim: range
           Range of y limit

       returns plotted selected FAS data

       """

    # Load the data
    fas_sel_out_df = pd.read_csv(csv_path, header=None)
    fas_sel_out_df = fas_sel_out_df[1:].astype(float)

    # Extract frequency values
    freq_hv_mean = fas_sel_out_df[0].values

    # Initialize the figure
    plt.figure()

    # Process each window
    num_windows = (len(fas_sel_out_df.columns) - 1) // 3  # Calculate the number of windows

    colors = [h1_col, h2_col, v_col]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['H1', 'H2', 'V']

    for i in range(num_windows):
        # Calculate column indices for the current window
        h1_col = 1 + 3 * i
        h2_col = 2 + 3 * i
        v_col = 3 + 3 * i

        # Extract data for the current window
        h1_smooth = fas_sel_out_df[h1_col].values
        h2_smooth = fas_sel_out_df[h2_col].values
        v_smooth = fas_sel_out_df[v_col].values

        # Plot the FAS curves for the current window
        plt.loglog(freq_hv_mean, h1_smooth, color=h1_col)
        plt.loglog(freq_hv_mean, h2_smooth, color=h2_col)
        plt.loglog(freq_hv_mean, v_smooth, color=v_col)

    # Add labels, title, and legend
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FAS Curves for Multiple Windows')
    plt.legend(lines, labels, loc='upper right')
    plt.grid(True)

    # Show the plot
    plt.show()
