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
    xml_file = 'NX.USC3_20240530T184335Z.xml'  
    xml_file_path = os.path.join(setup_test_data, xml_file)
    sta, cha = HvsrMetaTools.process_xml_file(xml_file_path)
    assert sta is not None
    assert cha is not None

def test_process_hvsr_metadata(setup_test_data):
    
    field_crew = None
    user = 'john doe'
    weather = None
    ground_type = None
    sensor_ground_coupling = None
    monochromatic = None
    building = None
    transients = None
    data_type = 0
    sensor_corner_frequency = 0.0083
    gain = 1
    comments = None

    df = HvsrMetaTools.process_hvsr_metadata(field_crew, user, weather, ground_type, sensor_ground_coupling,
                                            monochromatic, building, transients, data_type,
                                            sensor_corner_frequency, gain, comments, setup_test_data)

    assert len(df) == 1

def test_process_hvsr_metadata_for_a_single_site(setup_test_data):
    
    field_crew = None
    user = 'john doe'
    weather = None
    ground_type = None
    sensor_ground_coupling = None
    xml_file = 'NX.USC3_20240530T184335Z.xml' 
    xml_file_path = os.path.join(setup_test_data, xml_file)
    inventory_path=xml_file_path

    df = HvsrMetaTools.process_hvsr_metadata_single_site(field_crew, user, weather, ground_type, sensor_ground_coupling, inventory_path=inventory_path)

    assert len(df) == 1

def test_create_mean_curves_csv(setup_test_data):
    add_sim_path = 'Text_File_data/Raw_mseed_PEG_HH'
    df = HvsrMetaTools.create_mean_curves_csv(setup_test_data, 'Test_hvsr_mean.csv', add_sim_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)

def test_combine_metadata(setup_test_data):
    add_sim_path = 'Text_File_data/Raw_mseed_PEG_HH'
    df = HvsrMetaTools.combine_metadata(setup_test_data, add_sim_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
