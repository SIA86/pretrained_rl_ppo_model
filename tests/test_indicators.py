import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scr.indicators import (
    sma_numba,
    ema_numba,
    ema_numba_safe,
    adx_numba,
    atr_numba,
    rsi_numba,
    macd_numba,
    stoch_numba,
    bollinger_numba,
    obv_numba,
    cci_numba,
    williams_r_numba,
    mfi_numba,
    roc_numba,
    slope_numba,
    vwap_numba,
    zigzag_pivots_highlow_numba,
    expand_pivots,
    PEAK,
    VALLEY,
)


def generate_data(n=100):
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(size=n)) + 100
    high = close + rng.random(n)
    low = close - rng.random(n)
    volume = rng.integers(1, 1000, size=n).astype(float)
    return high, low, close, volume


def assert_finite_or_nan(arr):
    assert np.all(np.isfinite(arr) | np.isnan(arr))


def test_indicators_safety():
    high, low, close, volume = generate_data()

    assert_finite_or_nan(sma_numba(close, 10))
    assert np.all(np.isnan(sma_numba(close, 0)))

    assert_finite_or_nan(ema_numba(close, 10))
    assert np.all(np.isnan(ema_numba(close, 0)))

    arr = close.copy()
    arr[10] = np.nan
    safe = ema_numba_safe(arr, 10)
    unsafe = ema_numba(arr, 10)
    assert not np.isnan(safe[10])
    assert np.isnan(unsafe[10])

    assert_finite_or_nan(adx_numba(high, low, close, 14))
    assert np.all(np.isnan(adx_numba(high, low, close, 0)))

    assert_finite_or_nan(atr_numba(high, low, close, 14))
    assert np.all(np.isnan(atr_numba(high, low, close, 0)))

    assert_finite_or_nan(rsi_numba(close, 14))
    assert np.all(np.isnan(rsi_numba(close, 0)))

    macd, signal, hist = macd_numba(close)
    assert_finite_or_nan(macd)
    assert_finite_or_nan(signal)
    assert_finite_or_nan(hist)

    k, d = stoch_numba(high, low, close, 14, 3)
    assert_finite_or_nan(k)
    assert_finite_or_nan(d)

    mu, upper, lower = bollinger_numba(close, 20, 2.0)
    assert_finite_or_nan(mu)
    assert_finite_or_nan(upper)
    assert_finite_or_nan(lower)

    assert_finite_or_nan(obv_numba(close, volume))

    assert_finite_or_nan(cci_numba(high, low, close, 20))

    assert_finite_or_nan(williams_r_numba(high, low, close, 14))

    assert_finite_or_nan(mfi_numba(high, low, close, volume, 14))

    assert_finite_or_nan(roc_numba(close, 12))

    assert_finite_or_nan(slope_numba(close, 10))
    assert np.all(np.isnan(slope_numba(close, 1)))

    assert_finite_or_nan(vwap_numba(high, low, close, volume, 20))

    piv = zigzag_pivots_highlow_numba(high, low, close, 0.05, -0.05)
    assert set(np.unique(piv)).issubset({PEAK, VALLEY, 0})
    exp = expand_pivots(piv, 2)
    assert set(np.unique(exp)).issubset({PEAK, VALLEY, 0})


def test_slope_numba_nan_handling():
    values = np.array(
        [
            np.nan,
            np.nan,
            1.0,
            2.0,
            3.0,
            4.0,
            6.0,
            np.nan,
            8.0,
            9.0,
            10.0,
            11.0,
        ],
        dtype=np.float64,
    )
    period = 3
    res = slope_numba(values, period)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isnan(res[2])
    assert np.isnan(res[3])
    assert np.isfinite(res[4])

    assert np.isnan(res[7])
    assert np.isnan(res[8])
    assert np.isnan(res[9])
    assert np.isfinite(res[10])
    assert np.isfinite(res[11])


def test_slope_numba_linear_trend():
    values = np.arange(20.0)
    period = 5
    res = slope_numba(values, period)
    expected = 1.0 / np.sqrt((period * period - 1.0) / 12.0)
    valid = res[period - 1:]
    assert np.all(np.isfinite(valid))
    assert np.allclose(valid, expected)

