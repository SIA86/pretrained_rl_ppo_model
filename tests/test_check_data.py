import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr.check_data import prepare_time_series, _robust_infer_freq


def test_unix_seconds_to_europe_riga_correct_conversion():
    ts = [1704067200, 1704070800]
    df = pd.DataFrame({"timestamp": ts, "price": [1, 2]})
    out = prepare_time_series(df, "timestamp", tz="Europe/Riga")
    assert str(out.index[0]) == "2024-01-01 02:00:00+02:00"


def test_infer_freq_tz_aware_no_crash():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    inferred = _robust_infer_freq(idx)
    assert to_offset(inferred) == to_offset("h")


def test_missing_fallback_freq_no_regularization():
    df = pd.DataFrame({"timestamp": [1704067200, 1704067800], "x": [1.0, 2.0]})
    out = prepare_time_series(df, "timestamp", tz="UTC", fallback_freq=None)
    assert len(out) == 2
    assert list(out.index) == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:10:00Z"),
    ]


def test_invalid_fallback_freq_raises_valueerror():
    df = pd.DataFrame({"timestamp": [1704067200, 1704067800], "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        prepare_time_series(df, "timestamp", tz="UTC", fallback_freq="definitely_not_a_freq")


def test_dst_spring_forward_europe_riga_nonexistent_local_hour():
    df = pd.DataFrame({
        "timestamp": [1711843200, 1711846800, 1711850400],
        "price": [1.0, 2.0, 3.0],
    })
    out = prepare_time_series(df, "timestamp", tz="Europe/Riga", fallback_freq="1h")
    idx = out.index
    has_02 = any((i.year, i.month, i.day, i.hour) == (2024, 3, 31, 2) for i in idx)
    has_03 = any((i.year, i.month, i.day, i.hour) == (2024, 3, 31, 3) for i in idx)
    has_04 = any((i.year, i.month, i.day, i.hour) == (2024, 3, 31, 4) for i in idx)
    assert has_02 and has_04 and not has_03
    assert idx.is_monotonic_increasing
    assert str(idx.tz) == "Europe/Riga"


def test_string_iso_with_tz_no_localize_error():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
        "v": [1, 2],
    })
    out = prepare_time_series(df, "timestamp", tz="Europe/Riga")
    assert out.index.tz.zone in ["Europe/Riga", "EET"]


def test_regularization_and_fill():
    df = pd.DataFrame({"timestamp": [1704067200, 1704070800],
                       "price": [1.0, np.nan], "Volume": [np.nan, 5.0]})
    out = prepare_time_series(df, "timestamp", tz="UTC", fallback_freq="1h")
    assert out.loc[out.index[1], "price"] == 1.0
    assert out["Volume"].fillna(0).iloc[0] == 0.0


def test_dedup_drop_vs_agg_behavior():
    df = pd.DataFrame({"timestamp": [1704067200, 1704067200], "a": [1, 2], "b": [10, 20]})
    out_drop = prepare_time_series(df, "timestamp")
    assert len(out_drop) == 1 and out_drop["a"].iloc[0] == 1
    out_agg = prepare_time_series(df, "timestamp", dedup_agg={"a": "max", "b": "sum"})
    assert out_agg["a"].iloc[0] == 2 and out_agg["b"].iloc[0] == 30


def test_bad_timestamp_raises():
    df = pd.DataFrame({"timestamp": ["not-a-date"], "x": [1]})
    with pytest.raises(ValueError):
        prepare_time_series(df, "timestamp")


def test_window_slice_tz_mixing():
    df = pd.DataFrame({"timestamp": [1704067200, 1704070800], "x": [1, 2]})
    out = prepare_time_series(
        df,
        "timestamp",
        tz="UTC",
        from_date=pd.Timestamp("2024-01-01T00:30Z"),
        to_date="2024-01-01T01:30Z",
    )
    assert len(out) == 1 and out.index[0] == pd.Timestamp("2024-01-01T01:00Z")
