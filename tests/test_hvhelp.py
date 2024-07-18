import pytest
import os
import numpy as np
from hvsrprocpy import _add_mseed_tool, proc_mseed_data, proc_txt_data, process_time_series

# Define test data directory and files
@pytest.fixture
def setup_test_data():
    # Determine the path to the current directory where this test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(current_dir, 'test_data')
    yield test_data_dir

def test_proc_mseed_data(setup_test_data):
    H1_FN = 'NX.USC3.HHE.20240530_000000'
    H2_FN = 'NX.USC3.HHN.20240530_000000'
    V_FN = 'NX.USC3.HHZ.20240530_000000'
    H1 = os.path.join(setup_test_data, H1_FN)
    H2 = os.path.join(setup_test_data, H2_FN)
    V = os.path.join(setup_test_data, V_FN)
    h1, h2, v, dt, time = proc_mseed_data(file_direc=setup_test_data, h1_fn=H1, h2_fn=H2, v_fn=V)
    assert isinstance(h1, np.ndarray)
    assert isinstance(dt, float)
    assert isinstance(time, np.ndarray)

def test_proc_txt_data(setup_test_data):
    H1_FN = '20240530184343_NX_USC3_HHE.txt'
    H2_FN = '20240530184343_NX_USC3_HHN.txt'
    V_FN = '20240530184343_NX_USC3_HHZ.txt'
    H1 = os.path.join(setup_test_data, H1_FN)
    H2 = os.path.join(setup_test_data, H2_FN)
    V = os.path.join(setup_test_data, V_FN)
    h1, h2, v, dt, time = proc_txt_data(file_direc=setup_test_data, h1_fn=H1, h2_fn=H2, v_fn=V)
    assert isinstance(h1, np.ndarray)
    assert isinstance(dt, float)
    assert isinstance(time, np.ndarray)

def test_process_time_series_mseed(setup_test_data):
    H1_FN = 'NX.USC3.HHE.20240530_000000'
    H2_FN = 'NX.USC3.HHN.20240530_000000'
    V_FN = 'NX.USC3.HHZ.20240530_000000'
    H1 = os.path.join(setup_test_data, H1_FN)
    H2 = os.path.join(setup_test_data, H2_FN)
    V = os.path.join(setup_test_data, V_FN)
    h1, h2, v, dt, time = process_time_series(h1_fn=H1, h2_fn=H2, v_fn=V, directory=setup_test_data, file_type=1)
    assert isinstance(h1, np.ndarray)
    assert isinstance(dt, float)
    assert isinstance(time, np.ndarray)

def test_process_time_series_txt(setup_test_data):
    H1_FN = '20240530184343_NX_USC3_HHE.txt'
    H2_FN = '20240530184343_NX_USC3_HHN.txt'
    V_FN = '20240530184343_NX_USC3_HHZ.txt'
    H1 = os.path.join(setup_test_data, H1_FN)
    H2 = os.path.join(setup_test_data, H2_FN)
    V = os.path.join(setup_test_data, V_FN)
    h1, h2, v, dt, time = process_time_series(h1_fn=H1, h2_fn=H2, v_fn=V, directory=setup_test_data, file_type=2)
    assert isinstance(h1, np.ndarray)
    assert isinstance(dt, float)
    assert isinstance(time, np.ndarray)

def test_process_time_series_invalid_file_type(setup_test_data):
    H1_FN = '20240530184343_NX_USC3_HHE.txt'
    H2_FN = '20240530184343_NX_USC3_HHN.txt'
    V_FN = '20240530184343_NX_USC3_HHZ.txt'
    H1 = os.path.join(setup_test_data, H1_FN)
    H2 = os.path.join(setup_test_data, H2_FN)
    V = os.path.join(setup_test_data, V_FN)
    with pytest.raises(ValueError):
        process_time_series(h1_fn=H1, h2_fn=H2, v_fn=V, directory=setup_test_data, file_type=3)

