import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr.q_labels_matching import (
    next_exit_exec_arrays,
    enrich_q_labels_trend_one_side,
    soft_signal_labels_gaussian,
)

# ---------------------------
# Вспомогательные генераторы
# ---------------------------


def _mk_df(open_, high=None, low=None, close=None, sig=None):
    n = len(open_)
    return pd.DataFrame(
        {
            "Open": np.asarray(open_, dtype=float),
            "High": np.asarray(high if high is not None else open_, dtype=float),
            "Low": np.asarray(low if low is not None else open_, dtype=float),
            "Close": np.asarray(close if close is not None else open_, dtype=float),
            "Signal_Rule": np.asarray(
                sig if sig is not None else np.zeros(n, dtype=int)
            ),
        }
    )


# ---------------------------
# next_exit_exec_arrays
# ---------------------------


def test_exit_execution_is_next_open_long():
    """
    Сигнал выхода (sell) в t=2 -> исполнение на t=3 (Open[3]).
    Для t=0 ближайший исполняемый exit должен быть e=3.
    """
    open_px = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    sell = np.array([0, 0, 1, 0, 0], dtype=bool)
    buy = np.array([0, 0, 0, 0, 0], dtype=bool)
    idx, px = next_exit_exec_arrays(open_px, buy, sell, side_long=True)
    assert idx[0] == 3
    assert px[0] == open_px[3]


def test_exit_execution_is_next_open_short():
    """
    В short-only выходом считается buy-сигнал.
    """
    open_px = np.array([100.0, 90.0, 80.0, 85.0, 70.0])
    buy = np.array([0, 0, 1, 0, 0], dtype=bool)  # buy@2 -> exec@3
    sell = np.array([0, 0, 0, 0, 0], dtype=bool)
    idx, px = next_exit_exec_arrays(open_px, buy, sell, side_long=False)
    assert idx[0] == 3
    assert px[0] == open_px[3]


# ---------------------------
# TD-λ
# ---------------------------


def test_tdlambda_weighted_average_open_long():
    """
    TD(λ): Q_Open — средневзвешенная смесь n-step возвратов.
    Проверяем, что при константном росте значения разумные и маска корректна.
    """
    open_px = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    df = _mk_df(open_px)
    out = enrich_q_labels_trend_one_side(
        df, H_max=3, lam=0.5, side_long=True, fee=0, slippage=0
    )
    # t=0: доступны n=1..3 → маска валидна
    assert out.loc[0, "Mask_Open"] == 1
    assert not np.isnan(out.loc[0, "Q_Open"])


def test_tdlambda_masks_invalid_when_no_future():
    """
    Если для строки нет ни одного валидного будущего барра (все fut NaN),
    маски Open/Hold должны быть 0, а Q — NaN.
    """
    open_px = np.array([10.0, 11.0, np.nan, np.nan])
    df = _mk_df(open_px)

    out = enrich_q_labels_trend_one_side(df, H_max=3, side_long=True)
    # Последний бар точно невалиден
    last = len(df) - 1
    assert out.loc[last, "Mask_Open"] == 0
    assert out.loc[last, "Mask_Hold"] == 0
    assert np.isnan(out.loc[last, "Q_Open"])
    assert np.isnan(out.loc[last, "Q_Hold"])

    # Для любой строки с Mask_Open==1 — Q_Open обязан быть числом
    idx_valid = np.where(out["Mask_Open"] == 1)[0]
    assert all(~pd.isna(out.loc[idx_valid, "Q_Open"]))


# ---------------------------
# NaN / граничные случаи
# ---------------------------


def test_nan_inputs_propagate_to_q_and_masks():
    """
    Если Open[t+1] = NaN — действия, зависящие от exec_next_open, невалидны.
    """
    open_px = np.array([100.0, np.nan, 110.0, 120.0])
    df = _mk_df(open_px)

    out = enrich_q_labels_trend_one_side(
        df, H_max=2, side_long=True, fee=0, slippage=0
    )
    # На t=0 exec_next_open = Open[1] = NaN -> Open/Wait только маска Wait=1, Open=0
    assert out.loc[0, "Mask_Open"] == 0 or np.isnan(out.loc[0, "Q_Open"])
    # На последних барах без будущего горизонта маска Open=0
    assert out.loc[len(df) - 1, "Mask_Open"] == 0


# ---------------------------
# Комиссии
# ---------------------------


def test_commissions_subtracted_from_q():
    open_px = np.array([100.0, 100.0, 100.0])
    df = _mk_df(open_px)
    out = enrich_q_labels_trend_one_side(
        df, H_max=1, side_long=True, fee=0.001, slippage=0.002
    )
    c = 0.001 + 0.002
    scale = 2e-3
    np.testing.assert_allclose(out.loc[0, "Q_Open"], -(2 * c) / scale, rtol=1e-7)
    np.testing.assert_allclose(out.loc[0, "Q_Hold"], -c / scale, rtol=1e-7)
    np.testing.assert_allclose(out.loc[0, "Q_Close"], -c / scale, rtol=1e-7)


def test_volatility_scaling_divides_by_causal_vol():
    open_px = np.array([1.0, 1.2, 1.0, 1.3, 1.1, 1.4])
    df = _mk_df(open_px)
    out = enrich_q_labels_trend_one_side(
        df,
        H_max=2,
        side_long=True,
        fee=0,
        slippage=0,
        scale_mode="vol",
        vol_window=2,
    )
    ret = pd.Series(open_px).pct_change()
    vol = ret.rolling(2).std().shift(1).to_numpy()
    vol = np.where(np.isfinite(vol) & (vol > 0.0), vol, 2e-3)
    lam = 0.9
    w = np.array([(1 - lam) * (lam ** (k - 1)) for k in range(1, 3)], dtype=np.float64)
    w = w / w.sum()
    idx = 3
    r2 = open_px[idx + 2] / open_px[idx + 1] - 1.0
    q_unscaled = w[1] * r2
    expected = q_unscaled / vol[idx]
    np.testing.assert_allclose(out.loc[idx, "Q_Open"], expected, rtol=1e-7)
    np.testing.assert_allclose(out.loc[idx, "Q_Hold"], expected, rtol=1e-7)


# ---------------------------
# soft_signal_labels_gaussian
# ---------------------------


def test_soft_labels_gaussian_blur_and_normalisation():
    """Размытие действует только влево, Open/Close достигают 0.9."""
    open_px = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    sig = np.array([0, 1, 0, -1, 0])
    df = _mk_df(open_px, sig=sig)

    out = soft_signal_labels_gaussian(df, blur_window=1, blur_sigma=1.0, mae_lambda=0.0)

    # Open воздействует только слева и максимум = 0.9
    np.testing.assert_allclose(out.loc[1, "A_Open"], 0.9, rtol=1e-7)
    assert out.loc[0, "A_Open"] < out.loc[1, "A_Open"]
    assert out.loc[2, "A_Open"] == 0.0

    # Close воздействует только слева и максимум = 0.9
    np.testing.assert_allclose(out.loc[3, "A_Close"], 0.9, rtol=1e-7)
    assert out.loc[2, "A_Close"] < out.loc[3, "A_Close"]
    assert out.loc[4, "A_Close"] == 0.0

    # Wait/Hold дополняют вероятности до 1 и убывают к сигналу
    np.testing.assert_allclose(out.loc[1, "A_Wait"], 0.1, rtol=1e-7)
    assert out.loc[0, "A_Wait"] > out.loc[1, "A_Wait"]
    np.testing.assert_allclose(out.loc[3, "A_Hold"], 0.1, rtol=1e-7)
    assert out.loc[2, "A_Hold"] > out.loc[3, "A_Hold"]

    # Каждая строка нормализована
    sums = out[["A_Open", "A_Close", "A_Hold", "A_Wait"]].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-7)

    # Смоделированная позиция доступна
    assert "Pos" in out.columns
    expected_pos = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(out["Pos"].to_numpy(np.int8), expected_pos)


def test_soft_labels_mae_penalty_shifts_to_close():
    """MAE-штраф увеличивает вес Close относительно Hold."""
    open_px = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 60.0])
    sig = np.array([1, 0, 0, 0, -1, 0])
    df = _mk_df(open_px, sig=sig)

    out_pen = soft_signal_labels_gaussian(
        df, blur_window=1, blur_sigma=1.0, mae_lambda=0.5
    )
    out_nopen = soft_signal_labels_gaussian(
        df, blur_window=1, blur_sigma=1.0, mae_lambda=0.0
    )

    # t=2: в позиции и в просадке, вес Close должен вырасти
    assert out_pen.loc[2, "A_Close"] > out_nopen.loc[2, "A_Close"]

    # Позиция совпадает с ожидаемой
    assert "Pos" in out_pen.columns
    expected_pos = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(out_pen["Pos"].to_numpy(np.int8), expected_pos)


def test_soft_labels_mae_penalty_shifts_to_open():
    """MAE-штраф уменьшает вес Wait и увеличивает вес Open на растущем рынке."""
    open_px = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    df = _mk_df(open_px, sig=np.zeros_like(open_px))

    out_pen = soft_signal_labels_gaussian(
        df, blur_window=1, blur_sigma=1.0, mae_lambda=0.5
    )
    out_nopen = soft_signal_labels_gaussian(
        df, blur_window=1, blur_sigma=1.0, mae_lambda=0.0
    )

    assert out_pen.loc[1, "A_Open"] > out_nopen.loc[1, "A_Open"]
    assert out_pen.loc[1, "A_Wait"] < out_nopen.loc[1, "A_Wait"]


def test_position_metrics_are_computed():
    """Проверяем расчёт нереализованного PnL, счётчиков шагов и просадки."""
    open_px = np.array([100.0, 110.0, 105.0, 120.0])
    high_px = np.array([100.0, 112.0, 107.0, 122.0])
    low_px = np.array([100.0, 108.0, 100.0, 115.0])
    sig = np.array([1, 0, 0, 0])
    df = _mk_df(open_px, high=high_px, low=low_px, close=open_px, sig=sig)

    out_soft = soft_signal_labels_gaussian(
        df, side_long=True, blur_window=1, blur_sigma=1.0, mae_lambda=0.0
    )
    out_q = enrich_q_labels_trend_one_side(
        df, side_long=True, fee=0, slippage=0
    )

    expected_unreal = np.array([0.0, 0.0, 105 / 110 - 1.0, 120 / 110 - 1.0])
    expected_flat = np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32)
    expected_hold = np.array([0.0, 0.001, 0.002, 0.003], dtype=np.float32)
    expected_dd = np.array([0.0, 108 / 110 - 1.0, 100 / 110 - 1.0, 100 / 110 - 1.0])

    for out in (out_soft, out_q):
        for col in ["Unreal_PnL", "Flat_Steps", "Hold_Steps", "Drawdown"]:
            assert col in out.columns
        np.testing.assert_allclose(out["Unreal_PnL"], expected_unreal, rtol=1e-7)
        np.testing.assert_allclose(out["Flat_Steps"], expected_flat, rtol=1e-7)
        np.testing.assert_allclose(out["Hold_Steps"], expected_hold, rtol=1e-7)
        np.testing.assert_allclose(out["Drawdown"], expected_dd, rtol=1e-7)
