from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import hvsrprocpy

import hvsrprocpy.tdt
from hvsrprocpy.hvt import _win_proc, process_noise_data, hvsr_and_fas_calc, hvsr


def test_win_proc():
    ts = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])]
    kwargs = {
        'ts_dt': 0.01,
        'detrend_type': 2,
        'taper_flag': False,
        't_front': 3,
        't_end': 3,
        'filter_flag': True,
        'fc': 0.05,
        'npole': -3,
        'is_causal': True,
        'order_zero_padding': 1
    }
    processed_ts = _win_proc(ts, **kwargs)
    assert len(processed_ts) == len(ts)
    for ts_data in processed_ts:
        assert isinstance(ts_data, np.ndarray)

def test_process_noise_data():
    ts = np.random.rand(100)
    dt = 0.01
    kwargs = {
        'pre_filter_flag': True,
        'detrend_type': 1,
        'pre_filter_t_front': 5,
        'pre_filter_t_end': 5,
        'pre_filter_hpass_fc': 0.1,
        'pre_filter_npole_hp': -4,
        'pre_filter_is_causal': False,
        'pre_filter_order_zero_padding': 2,
        'norm_flag': True,
        'win_width': 10,
        'overlapping': 0.5,
        'is_noise': True,
        'sta_lta_flag': True,
        'short_term_len': 2,
        'long_term_len': 10,
        'sta_lta_moving_term': 5,
        'filter_flag': True,
        'hpass_fc': 0.2,
        'npole_hp': -3,
        'lpass_fc': 0.05,
        'npole_lp': -2,
        'eqk_filepath': 'earthquake_data.csv'
    }
    ts_processed, ts_wins, ts_stalta = process_noise_data(ts, dt, **kwargs)
    assert isinstance(ts_processed, np.ndarray)
    assert isinstance(ts_wins, np.ndarray)
    assert isinstance(ts_stalta, list)

@pytest.fixture
def setup_data():
    # Fixture to set up common test data
    h1 = np.linspace(0, 100, 10000)
    h2 = np.linspace(0, 100, 10000)
    v = np.linspace(0, 100, 10000)
    dt = 0.1
    h1_wins = hvsrprocpy.tdt.split_into_windows(h1, dt,win_width=300, overlapping=0)
    h2_wins = hvsrprocpy.tdt.split_into_windows(h2, dt, win_width=300,  overlapping=0)
    v_wins = hvsrprocpy.tdt.split_into_windows(v, dt, win_width=300, overlapping=0)
    freq_hv_mean = np.linspace(0.01, 50, 200)
    freq_polar = np.linspace(0.01, 50, 200)
    return h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar

def test_default_parameters(setup_data):
    # Test with default parameters
    h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar = setup_data
    res = hvsr_and_fas_calc(h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar)

    assert 'hv_ratio' in res
    assert 'h1_smooth' in res
    assert 'h2_smooth' in res
    assert 'v_smooth' in res

    assert isinstance(res['hv_ratio'], np.ndarray)
    assert isinstance(res['h1_smooth'], np.ndarray)
    assert isinstance(res['h2_smooth'], np.ndarray)
    assert isinstance(res['v_smooth'], np.ndarray)

def test_custom_parameters(setup_data):
    # Test with custom parameters
    h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar = setup_data
    custom_kwargs = {
        'ko_smooth_b': 20,
        'parzen_bwidth': 2.0,
        'sjb_avg': True,
        'polar_curves_flag': True,
        'deg_increment': 20
    }
    res = hvsr_and_fas_calc(h1_wins, h2_wins, v_wins, dt, freq_hv_mean, freq_polar, **custom_kwargs)

    assert 'hv_ratio' in res
    assert 'h1_smooth' in res
    assert 'h2_smooth' in res
    assert 'v_smooth' in res

    assert isinstance(res['hv_ratio'], np.ndarray)
    assert isinstance(res['h1_smooth'], np.ndarray)
    assert isinstance(res['h2_smooth'], np.ndarray)
    assert isinstance(res['v_smooth'], np.ndarray)

    if custom_kwargs.get('polar_curves_flag', False):
        assert 'polar_hv_ratio' in res
        assert isinstance(res['polar_hv_ratio'], np.ndarray)

@pytest.fixture
def mock_inputs():
    # Mocking the inputs
    h1 = np.array([1.0, 2.0, 3.0])
    h2 = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    dt = 0.1
    time_ts = np.array([0.0, 0.1, 0.2, 0.3])
    output_dir = '/mock/output/dir'

    kwargs = {
        'output_selected_ts': True,
        'output_removed_ts': False,
        'output_selected_hvsr': True,
        'output_removed_hvsr': False,
        'output_mean_curve': True,
        'output_polar_curves': False,
        'output_fas_mean_curve': True,
        'output_metadata': True
    }

    return h1, h2, v, dt, time_ts, output_dir, kwargs

def test_hvsr(mock_inputs):
    h1, h2, v, dt, time_ts, output_dir, kwargs = mock_inputs

    with patch('hvsrprocpy.split_into_windows') as mock_split, \
            patch('hvsrprocpy.process_noise_data') as mock_process_noise, \
            patch('hvsrprocpy.hvsr_and_fas_calc') as mock_hvsr_and_fas_calc, \
            patch('hvsrprocpy.hvt._ts_plt_select') as mock_ts_plt_select, \
            patch('hvsrprocpy.hvt._hvsr_plt_select') as mock_hvsr_plt_select:
        # Mock the return values for functions used inside hvsr
        mock_split.return_value = [np.array([time_ts]), np.array([time_ts]), np.array([time_ts])]
        ts_processed_mock = [h1, h2, v]
        ts_wins_mock = [h1, h2, v]
        ts_stalta_mock = [MagicMock(), MagicMock(), MagicMock()]
        mock_process_noise.side_effect = list(zip(ts_processed_mock, ts_wins_mock, ts_stalta_mock))
        mock_hvsr_and_fas_calc.return_value = {
            'hv_ratio': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'polar_hv_ratio': np.array([[1.0, 2.0], [3.0, 4.0]])
        }
        mock_ts_plt_select.return_value = [0]
        mock_hvsr_plt_select.return_value = {'idx_select': [0]}

        # Call the function
        hvsr(h1, h2, v, dt, time_ts, output_dir, **kwargs)

        # Add assertions based on the expected behavior
        mock_split.assert_called_once()
        mock_process_noise.assert_any_call(ts=h1, dt=dt, **kwargs)
        mock_process_noise.assert_any_call(ts=h2, dt=dt, **kwargs)
        mock_process_noise.assert_any_call(ts=v, dt=dt, **kwargs)
        mock_ts_plt_select.assert_called_once()
        mock_hvsr_plt_select.assert_called_once()
        mock_hvsr_and_fas_calc.assert_called_once()
