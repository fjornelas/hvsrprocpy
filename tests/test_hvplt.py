import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import pymysql, json
from hvsrprocpy import hvsrplot  # Assuming your module is named hvsrprocpy and the file is hvsrplot.py

# Fixture to set up test data directory
@pytest.fixture
def setup_test_data():
    # Determine the path to the current directory where this test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(current_dir, 'test_data')
    return test_data_dir

# Fixture to load test data
@pytest.fixture
def test_data(setup_test_data):
    file_paths = {
        'polar_data': os.path.join(setup_test_data, 'Test_hvsr_polar.csv'),
        'mean_hvsr': os.path.join(setup_test_data, 'Test_hvsr_mean.csv'),
        'metadata': os.path.join(setup_test_data, 'Test_metadata.csv'),
        'time_series': os.path.join(setup_test_data, 'Test_ts_sel.csv'),
        'selected_hvsr': os.path.join(setup_test_data, 'Test_hvsr_sel.csv'),
        'fas_data': os.path.join(setup_test_data, 'Test_FAS_mean.csv')
    }
    
    # Load test data into a dictionary of DataFrames
    data = {}
    for key, file_path in file_paths.items():
        data[key] = pd.read_csv(file_path)
    return data

# Tests using fixtures
def test_process_polar_curve(test_data):
    polar_data = test_data['polar_data']
    deg_increment = 10
    azimuths = AZIMUTHS = list(range(0, 180, deg_increment))
    standard_freqs = json.load(open(os.path.join(setup_test_data, 'HVSR_VSPDB_standard_frequencies.json'))

    result = hvsrplot.process_polar_curve(polar_data, azimuths, standard_freqs)
    assert isinstance(result, pd.DataFrame)
    assert 'frequency' in result.columns
    assert 'ratio' in result.columns
    assert 'standard_deviation' in result.columns

def test_plot_polar_ratio(test_data):
    polar_data = test_data['polar_data']
    deg_increment = 10
    azimuths = AZIMUTHS = list(range(0, 180, deg_increment))
    standard_freqs = json.load(open(os.path.join(setup_test_data, 'HVSR_VSPDB_standard_frequencies.json'))

    processed_data = hvsrplot.process_polar_curve(polar_data, azimuths, standard_freqs)
    fig = hvsrplot.plot_polar_ratio(processed_data)
    assert isinstance(fig, plt.Figure)

def test_plot_mean_hvsr(test_data):
    mean_hvsr_data = test_data['mean_hvsr']
    metadata_data = test_data['metadata']

    fig = hvsrplot.plot_mean_hvsr(mean_hvsr_data, metadata_data)
    assert isinstance(fig, plt.Figure)

def test_plot_selected_time_series(test_data):
    time_series_data = test_data['time_series']
    fig = hvsrplot.plot_selected_time_series(time_series_data)
    assert isinstance(fig, plt.Figure)

def test_plot_selected_hvsr(test_data):
    selected_hvsr_data = test_data['selected_hvsr']
    fig = hvsrplot.plot_selected_hvsr(selected_hvsr_data)
    assert isinstance(fig, plt.Figure)

def test_plot_fas(test_data):
    fas_data = test_data['fas_data']
    fig = hvsrplot.plot_fas(fas_data)
    assert isinstance(fig, plt.Figure)
