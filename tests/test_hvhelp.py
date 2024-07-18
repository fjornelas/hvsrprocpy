import pytest
import os
import numpy as np
from hvsrprocpy import _add_mseed_tool, proc_mseed_data, proc_txt_data, process_time_series

# Define test data directory and files
TEST_DATA_DIR = 'test_data'
H1_FN = 'h1.mseed'
H2_FN = 'h2.mseed'
V_FN = 'v.mseed'

@pytest.fixture(scope='module')
def setup_test_data():
    # Create or ensure the existence of test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Create dummy files for testing
    with open(os.path.join(TEST_DATA_DIR, H1_FN), 'w') as f:
        f.write("Dummy data for h1.mseed")
    with open(os.path.join(TEST_DATA_DIR, H2_FN), 'w') as f:
        f.write("Dummy data for h2.mseed")
    with open(os.path.join(TEST_DATA_DIR, V_FN), 'w') as f:
        f.write("Dummy data for v.mseed")

    yield

    # Teardown: clean up the test data directory after tests
    for file in [H1_FN, H2_FN, V_FN]:
        os.remove(os.path.join(TEST_DATA_DIR, file))
    os.rmdir(TEST_DATA_DIR)

def test_add_mseed_tool():
    # Dummy Stream object for testing _add_mseed_tool
    dummy_stream = None  # Replace with a proper Stream object if necessary
    result = _add_mseed_tool(dummy_stream)
    # Assert expected behavior here based on your function implementation

def test_proc_mseed_data(setup_test_data):
    h1, h2, v, dt, time = proc_mseed_data(file_direc=TEST_DATA_DIR, h1_fn=H1_FN, h2_fn=H2_FN, v_fn=V_FN)
    # Add assertions to verify correctness of returned values

def test_proc_txt_data(setup_test_data):
    h1, h2, v, dt, time = proc_txt_data(file_direc=TEST_DATA_DIR, h1_fn=H1_FN, h2_fn=H2_FN, v_fn=V_FN)
    # Add assertions to verify correctness of returned values

def test_process_time_series_mseed(setup_test_data):
    h1, h2, v, dt, time = process_time_series(h1_fn=H1_FN, h2_fn=H2_FN, v_fn=V_FN, directory=TEST_DATA_DIR, file_type=1)
    # Add assertions to verify correctness of returned values

def test_process_time_series_txt(setup_test_data):
    h1, h2, v, dt, time = process_time_series(h1_fn=H1_FN, h2_fn=H2_FN, v_fn=V_FN, directory=TEST_DATA_DIR, file_type=2)
    # Add assertions to verify correctness of returned values

def test_process_time_series_invalid_file_type(setup_test_data):
    with pytest.raises(ValueError):
        process_time_series(h1_fn=H1_FN, h2_fn=H2_FN, v_fn=V_FN, directory=TEST_DATA_DIR, file_type=3)

