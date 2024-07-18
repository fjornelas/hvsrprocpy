import pytest
import pandas as pd
import os
from hvsrprocpy import HvsrMetaTools

@pytest.fixture
def setup_test_data():
    # Determine the path to the current directory where this test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(current_dir, 'test_data')
    yield test_data_dir

def test_process_xml_file(setup_test_data):
    xml_file = 'example.xml'  # Located in the test_data directory
    xml_file_path = os.path.join(setup_test_data, xml_file)
    sta, cha = HvsrMetaTools.process_xml_file(xml_file_path)
    assert sta is not None
    assert cha is not None

def test_add_to_dataframe():
    df = pd.DataFrame(columns=['name', 'longitude', 'latitude', 'elevation', 'start_date', 'end_date',
                               'start_time', 'end_time', 'field_crew', 'seismic_recorder',
                               'recorder_serial_number', 'gain', 'sensor_manufacturer', 'sensor',
                               'sensor_name', 'sensor_serial_number', 'sensor_corner_frequency',
                               'weather', 'ground_type', 'monochromatic_noise_source',
                               'sensor_ground_coupling', 'building', 'transients', 'azimuth',
                               'user', 'comments', 'data_type', 'mass_position_w', 'mass_position_v',
                               'mass_position_u', 'mag_dec'])

    sta = {'description': 'Test Station', 'site': None, 'code': 'TS1'}
    cha = {'longitude': 0.0, 'latitude': 0.0, 'elevation': 100.0, 'start_date': '2024-07-18', 'end_date': '2024-07-18',
           'data_logger': None, 'sensor': None}
    df = HvsrMetaTools.add_to_dataframe(df, sta, cha, 'John Doe', 'Weather', 'Ground', 'Buried', 'Yes', 'No',
                                        0, 0.0083, 1, 'No comments')
    assert len(df) == 1

def test_process_hvsr_metadata_single_site():
    df = HvsrMetaTools.process_hvsr_metadata_single_site('John Doe', 'Weather', 'Ground', 'Buried')
    assert len(df) == 1

def test_create_mean_curves_csv(setup_test_data):
    add_sim_path = 'additional_path'
    df = HvsrMetaTools.create_mean_curves_csv(setup_test_data, 'test_mean_curves.csv', add_sim_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)

def test_combine_metadata(setup_test_data):
    add_sim_path = 'additional_path'
    df = HvsrMetaTools.combine_metadata(setup_test_data, add_sim_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
