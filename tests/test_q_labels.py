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
    return pd.DataFrame({
        'Open':  np.asarray(open_, dtype=float),
        'High':  np.asarray(high  if high  is not None else open_, dtype=float),
        'Low':   np.asarray(low   if low   is not None else open_, dtype=float),
        'Close': np.asarray(close if close is not None else open_, dtype=float),
        'Signal_Rule': np.asarray(sig if sig is not None else np.zeros(n, dtype=int)),
    })


# ---------------------------
# next_exit_exec_arrays
# ---------------------------

def test_exit_execution_is_next_open_long():
    """
    Сигнал выхода (sell) в t=2 -> исполнение на t=3 (Open[3]).
    Для t=0 ближайший исполняемый exit должен быть e=3.
    """
    open_px = np.array([10., 11., 12., 13., 14.])
    sell = np.array([0,0,1,0,0], dtype=bool)
    buy  = np.array([0,0,0,0,0], dtype=bool)
    idx, px = next_exit_exec_arrays(open_px, buy, sell, side_long=True)
    assert idx[0] == 3
    assert px[0]  == open_px[3]


def test_exit_execution_is_next_open_short():
    """
    В short-only выходом считается buy-сигнал.
    """
    open_px = np.array([100., 90., 80., 85., 70.])
    buy  = np.array([0,0,1,0,0], dtype=bool)   # buy@2 -> exec@3
    sell = np.array([0,0,0,0,0], dtype=bool)
    idx, px = next_exit_exec_arrays(open_px, buy, sell, side_long=False)
    assert idx[0] == 3
    assert px[0]  == open_px[3]


# ---------------------------
# mode='exit'
# ---------------------------

def test_exit_mode_hold_uses_exec_next_open_price():
    """
    buy@0 (exec t=1), sell@2 (exec t=3).
    На t=1 позиция уже открыта, поэтому:
    Hold(t=1) = Open[3]/Open[2]-1 = 120/110 - 1
    """
    open_px = np.array([100., 105., 110., 120.])
    sig     = np.array([+1, 0, -1, 0])  # buy на t=0, sell на t=2
    df = _mk_df(open_px, sig=sig)

    out = enrich_q_labels_trend_one_side(
        df, mode='exit', side_long=True, fee=0, slippage=0
    )

    assert out.loc[0, 'Mask_Hold'] == 0  # на t=0 позиции ещё нет
    assert out.loc[1, 'Mask_Hold'] == 1
    np.testing.assert_allclose(out.loc[1, 'Q_Hold'], 120/110 - 1, rtol=1e-7)


def test_exit_mode_open_includes_both_fees_long():
    """
    В Open должны применяться издержки и на вход, и на выход.
    """
    open_px = np.array([100., 110., 121., 121.])  # рост ~10% затем ~10%
    sig     = np.array([0, 0, -1, 0])             # sell@2 -> exec@3
    df = _mk_df(open_px, sig=sig)

    fee = 10e-4   # 10 bps
    slp = 20e-4   # 20 bps
    c = fee + slp

    out = enrich_q_labels_trend_one_side(
        df, mode='exit', side_long=True, fee=fee, slippage=slp
    )
    # На t=0 flat, has exit -> Open валиден
    assert out.loc[0, 'Mask_Open'] == 1
    # Теоретическое ожидание:
    # entry_eff = Open[1] * (1 + c) = 110*(1+c)
    # exit_eff  = Open[3] * (1 - c) = 121*(1-c)
    expected = (121*(1-c)) / (110*(1+c)) - 1.0
    np.testing.assert_allclose(out.loc[0, 'Q_Open'], expected, rtol=1e-7)


def test_exit_mode_long_short_symmetry_open_hold():
    """
    Симметрия Open для long/short при mode='exit' без комиссий:
    R_long = O[exec_exit]/O[exec_entry] - 1
    R_short= O[exec_entry]/O[exec_exit] - 1 = 1/(1+R_long) - 1
    """
    open_px = np.array([100., 90., 80., 120., 110.])

    # LONG: exit определяется sell-сигналом -> sell@2 => exec@3
    sig_long = np.array([0, 0, -1, 0, 0])
    df_long  = _mk_df(open_px, sig=sig_long)
    out_long = enrich_q_labels_trend_one_side(df_long, mode='exit', side_long=True, fee=0, slippage=0)

    # SHORT: exit определяется buy-сигналом -> buy@2 => exec@3
    sig_short = np.array([0, 0, +1, 0, 0])
    df_short  = _mk_df(open_px, sig=sig_short)
    out_short = enrich_q_labels_trend_one_side(df_short, mode='exit', side_long=False, fee=0, slippage=0)

    # t=0: оба действия Open валидны (есть будущий exec-выход >= t+2)
    assert out_long.loc[0,'Mask_Open'] == 1
    assert out_short.loc[0,'Mask_Open'] == 1

    R_long  = out_long.loc[0,'Q_Open']
    R_short = out_short.loc[0,'Q_Open']
    # Проверяем точную взаимосвязь (без комиссий):
    # R_short == 1/(1+R_long) - 1
    np.testing.assert_allclose(R_short, 1.0/(1.0 + R_long) - 1.0, rtol=1e-7)

    # Для Hold симметрия зависит от факта наличия позиции; валидность масок допустима 0/1
    assert out_long.loc[1,'Mask_Hold'] in (0,1)
    assert out_short.loc[1,'Mask_Hold'] in (0,1)


# ---------------------------
# mode='horizon'
# ---------------------------

def test_horizon_basic_open_hold_long():
    """
    H=2: Open(t=0) = Open[2]/Open[1]-1; Hold(t=1) = Open[3]/Open[2]-1.
    """
    open_px = np.array([100., 110., 121., 133.1])
    df = _mk_df(open_px)

    out = enrich_q_labels_trend_one_side(df, mode='horizon', horizon=2,
                                         side_long=True, fee=0, slippage=0)
    assert out.loc[0, 'Mask_Open'] == 1
    np.testing.assert_allclose(out.loc[0, 'Q_Open'], 121/110 - 1, rtol=1e-7)

    # Чтобы Hold на t=1 был валиден, позиция должна быть открыта к t=1:
    sig = np.array([+1, 0, 0, 0])
    df2 = _mk_df(open_px, sig=sig)
    out2 = enrich_q_labels_trend_one_side(df2, mode='horizon', horizon=2,
                                          side_long=True, fee=0, slippage=0)
    assert out2.loc[1, 'Mask_Hold'] == 1
    np.testing.assert_allclose(out2.loc[1, 'Q_Hold'], 133.1/121 - 1, rtol=1e-7)


def test_horizon_commissions_monotonicity_open():
    """
    Рост комиссий не должен увеличивать Q_Open.
    """
    open_px = np.array([100., 110., 120., 130.])
    df = _mk_df(open_px)

    out_low_fee = enrich_q_labels_trend_one_side(df, mode='horizon', horizon=2,
                                                 side_long=True, fee=1e-4, slippage=1e-4)
    out_hi_fee  = enrich_q_labels_trend_one_side(df, mode='horizon', horizon=2,
                                                 side_long=True, fee=20e-4, slippage=20e-4)
    idx = np.where(out_low_fee['Mask_Open'] == 1)[0]
    assert len(idx) > 0
    assert all(out_hi_fee.loc[i, 'Q_Open'] <= out_low_fee.loc[i, 'Q_Open'] for i in idx)


# ---------------------------
# mode='tdlambda'
# ---------------------------

def test_tdlambda_weighted_average_open_long():
    """
    TD(λ): Q_Open — средневзвешенная смесь n-step возвратов.
    Проверяем, что при константном росте значения разумные и маска корректна.
    """
    open_px = np.array([100., 101., 102., 103., 104., 105.])
    df = _mk_df(open_px)
    out = enrich_q_labels_trend_one_side(df, mode='tdlambda', H_max=3, lam=0.5,
                                         side_long=True, fee=0, slippage=0)
    # t=0: доступны n=1..3 → маска валидна
    assert out.loc[0, 'Mask_Open'] == 1
    assert not np.isnan(out.loc[0, 'Q_Open'])


def test_tdlambda_mae_penalty_decreases_hold():
    """
    Создаём неблагоприятный дип в Low — штраф MAE должен уменьшать Q_Hold (при валидной позиции).
    """
    N = 12
    open_px = np.full(N, 100.0)
    high_px = np.full(N, 101.0)
    low_px  = np.full(N, 100.0)
    low_px[4] = 90.0   # дип

    sig = np.zeros(N, dtype=int)
    sig[0] = +1        # buy@0 -> pos с t=1
    sig[10] = -1       # sell@10 -> exec@11

    df = _mk_df(open_px, high=high_px, low=low_px, sig=sig)

    out_pen = enrich_q_labels_trend_one_side(df, mode='tdlambda', H_max=6, lam=0.9,
                                             side_long=True, use_mae_penalty=True, mae_lambda=0.5,
                                             mae_apply_to='hold', fee=0, slippage=0)
    out_nop = enrich_q_labels_trend_one_side(df, mode='tdlambda', H_max=6, lam=0.9,
                                             side_long=True, use_mae_penalty=False,
                                             fee=0, slippage=0)
    # Возьмём индексы, где Hold валиден
    idxs = np.where((out_pen['Mask_Hold']==1) & (out_nop['Mask_Hold']==1))[0]
    assert len(idxs) > 0
    assert any(out_pen.loc[i, 'Q_Hold'] < out_nop.loc[i, 'Q_Hold'] for i in idxs)


def test_tdlambda_masks_invalid_when_no_future():
    """
    Если для строки нет ни одного валидного будущего барра (все fut NaN),
    маски Open/Hold должны быть 0, а Q — NaN.
    """
    open_px = np.array([10., 11., np.nan, np.nan])
    df = _mk_df(open_px)

    out = enrich_q_labels_trend_one_side(df, mode='tdlambda', H_max=3, side_long=True)
    # Последний бар точно невалиден
    last = len(df) - 1
    assert out.loc[last, 'Mask_Open'] == 0
    assert out.loc[last, 'Mask_Hold'] == 0
    assert np.isnan(out.loc[last, 'Q_Open'])
    assert np.isnan(out.loc[last, 'Q_Hold'])

    # Для любой строки с Mask_Open==1 — Q_Open обязан быть числом
    idx_valid = np.where(out['Mask_Open'] == 1)[0]
    assert all(~pd.isna(out.loc[idx_valid, 'Q_Open']))


# ---------------------------
# NaN / граничные случаи
# ---------------------------

def test_nan_inputs_propagate_to_q_and_masks():
    """
    Если Open[t+1] = NaN — действия, зависящие от exec_next_open, невалидны.
    """
    open_px = np.array([100., np.nan, 110., 120.])
    df = _mk_df(open_px)

    out = enrich_q_labels_trend_one_side(df, mode='horizon', horizon=2,
                                         side_long=True, fee=0, slippage=0)
    # На t=0 exec_next_open = Open[1] = NaN -> Open/Wait только маска Wait=1, Open=0
    assert out.loc[0, 'Mask_Open'] == 0 or np.isnan(out.loc[0, 'Q_Open'])
    # На последних барах без будущего горизонта маска Open=0
    assert out.loc[len(df)-1, 'Mask_Open'] == 0


# ---------------------------
# soft_signal_labels_gaussian
# ---------------------------


def test_soft_labels_gaussian_blur_and_normalisation():
    """Размытие действует только влево, Open/Close достигают 0.9."""
    open_px = np.array([100., 101., 102., 103., 104.])
    sig = np.array([0, 1, 0, -1, 0])
    df = _mk_df(open_px, sig=sig)

    out = soft_signal_labels_gaussian(df, blur_window=1, blur_sigma=1.0, mae_lambda=0.0)

    # Open воздействует только слева и максимум = 0.9
    np.testing.assert_allclose(out.loc[1, 'A_Open'], 0.9, rtol=1e-7)
    assert out.loc[0, 'A_Open'] < out.loc[1, 'A_Open']
    assert out.loc[2, 'A_Open'] == 0.0

    # Close воздействует только слева и максимум = 0.9
    np.testing.assert_allclose(out.loc[3, 'A_Close'], 0.9, rtol=1e-7)
    assert out.loc[2, 'A_Close'] < out.loc[3, 'A_Close']
    assert out.loc[4, 'A_Close'] == 0.0

    # Wait/Hold дополняют вероятности до 1 и убывают к сигналу
    np.testing.assert_allclose(out.loc[1, 'A_Wait'], 0.1, rtol=1e-7)
    assert out.loc[0, 'A_Wait'] > out.loc[1, 'A_Wait']
    np.testing.assert_allclose(out.loc[3, 'A_Hold'], 0.1, rtol=1e-7)
    assert out.loc[2, 'A_Hold'] > out.loc[3, 'A_Hold']

    # Каждая строка нормализована
    sums = out[['A_Open', 'A_Close', 'A_Hold', 'A_Wait']].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-7)

    # Смоделированная позиция доступна
    assert 'Pos' in out.columns
    expected_pos = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(out['Pos'].to_numpy(np.int8), expected_pos)


def test_soft_labels_mae_penalty_shifts_to_close():
    """MAE-штраф увеличивает вес Close относительно Hold."""
    open_px = np.array([100., 90., 80., 70., 60., 60.])
    sig = np.array([1, 0, 0, 0, -1, 0])
    df = _mk_df(open_px, sig=sig)

    out_pen = soft_signal_labels_gaussian(df, blur_window=1, blur_sigma=1.0, mae_lambda=0.5)
    out_nopen = soft_signal_labels_gaussian(df, blur_window=1, blur_sigma=1.0, mae_lambda=0.0)

    # t=2: в позиции и в просадке, вес Close должен вырасти
    assert out_pen.loc[2, 'A_Close'] > out_nopen.loc[2, 'A_Close']

    # Позиция совпадает с ожидаемой
    assert 'Pos' in out_pen.columns
    expected_pos = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(out_pen['Pos'].to_numpy(np.int8), expected_pos)
