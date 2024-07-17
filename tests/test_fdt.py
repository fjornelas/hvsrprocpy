# tests/test_fdt.py

import pytest
import numpy as np
import hvsrprocpy as hv
from hvsrprocpy.fdt import *


def test_compute_horizontal_combination():
    h1_sub = np.random.randn(5, 100)
    h2_sub = np.random.randn(5, 100)
    dt = 0.01

    fas_h, freq = compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='geometric_mean')
    assert fas_h.shape == (5, 51)
    assert freq.shape == (51,)

    fas_h, freq = compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='squared_average')
    assert fas_h.shape == (5, 51)

    fas_h, freq = compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='ps_RotD50')
    assert fas_h.shape == (5, 51)

    with pytest.raises(KeyError):
        compute_horizontal_combination(h1_sub, h2_sub, dt, horizontal_comb='invalid_combination')


def test_smoothed_fas_ko():
    f = np.linspace(0.01, 50, 100)
    fas = np.random.rand(5, 100)
    fc = np.linspace(0.1, 25, 50)
    b = 40

    smoothed = smoothed_fas_ko(fas, f, fc, b)
    assert smoothed.shape == (5, 50)


def test_smooth_fas():
    fc = np.logspace(-2, 2, 200)
    h1 = np.random.rand(778012)
    dt = 0.05
    
    h1_win = hv.split_into_windows(h1,dt, win_width=300, overlapping =0)
    fas = np.fft.rfft(h1_win)
    freq = np.fft.rfftfreq(len(h1_win[0]),dt)
    
    
    smoothed = hv.smooth_fas(fc, np.abs(fas), freq, ko_smooth_flag=True, parzen_flag=False, ko_smooth_b=40, parzen_bwidth=1.5)
    assert smoothed.shape == (129, 200)
    
    smoothed = hv.smooth_fas(fc, np.abs(fas), freq, ko_smooth_flag=False, parzen_flag=True, ko_smooth_b=40, parzen_bwidth=1.5)
    assert smoothed.shape == (129, 200)


def test_fas_cal():
    ts = np.random.randn(100)
    dt = 0.01

    result = fas_cal(ts, dt, max_order=0)
    assert 'fft_ts' in result
    assert 'amp' in result
    assert 'freq' in result
    assert 'phase' in result

    assert result['fft_ts'].shape == (51,)
    assert result['amp'].shape == (51,)
    assert result['freq'].shape == (51,)
    assert result['phase'].shape == (51,)


if __name__ == "__main__":
    pytest.main()
