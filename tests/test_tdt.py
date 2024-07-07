# tests/test_tdt.py

import pytest
import numpy as np
from scipy.signal.windows import tukey
from hvsrprocpy.tdt import detrend_ts, tukey_window, apply_filter, apply_normalization, sta_lta_calc, split_into_windows


def test_detrend_ts():
    ts = np.linspace(1, 100, 200)

    detrended = detrend_ts(ts, detrend_type=1)
    win = tukey(len(ts), alpha=((10 + 10) / 200))  # Default Tukey window with 10% taper on each side
    ts_avg = np.average(ts, weights=win)
    assert np.allclose(detrended, ts - ts_avg)

    detrended = detrend_ts(ts, detrend_type=0)
    assert np.array_equal(detrended, ts)

    detrended = detrend_ts(ts, detrend_type=2)
    x = np.arange(1, 101)
    expected = ts - np.polyval(np.polyfit(x, ts, 1), x)
    assert np.allclose(detrended, expected)

    with pytest.raises(ValueError):
        detrend_ts(ts, detrend_type=99)


def test_tukey_window():
    ts = np.linspace(1, 100, 200)
    windowed = tukey_window(ts, t_front=10, t_end=10)
    assert len(windowed) == len(ts)


def test_apply_filter():
    ts = np.random.randn(200)
    filtered, res = apply_filter(ts, ts_dt=0.005, fc=0.042, npole=-5, is_causal=False, order_zero_padding=2)
    assert len(filtered) == len(ts)
    assert 'flt_ts' in res
    assert 'flt_amp' in res
    assert 'flt_resp' in res
    assert 'flt_fft' in res


def test_apply_normalization():
    ts = np.linspace(1, 100, 200)
    normalized = apply_normalization(ts)
    assert np.isclose(np.linalg.norm(normalized), 1)


def test_sta_lta_calc():
    ts = np.random.randn(200)
    ratio = sta_lta_calc(ts, short_term=5, long_term=20, moving_term=1)
    assert len(ratio) == (len(ts) - 5) // 1 + 1


def test_split_into_windows():
    ts = np.random.randn(200)
    windows = split_into_windows(ts, dt=1, win_width=10, overlapping=5)
    assert windows.shape[0] == (len(ts) // (10 - 5))
    assert windows.shape[1] == (10 - 5)


if __name__ == "__main__":
    pytest.main()
