"""Утилиты для построения оконных датасетов с опциональными весами.

Модуль включает функции для очистки Q‑значений и масок, извлечения
признаков, нарезки временных рядов окнами и создания объектов
``tf.data.Dataset``. Код адаптирован из пользовательской спецификации и
рассчитан на работу с DataFrame, подготовленным модулем
``q_labels_matching``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .normalisation import NormalizationStats

# ---------------------------------------------------------------------------
# Константы, описывающие колоноки действий
# ---------------------------------------------------------------------------

ACTIONS = ["Open", "Close", "Hold", "Wait"]
NUM_CLASSES = len(ACTIONS)
Q_COLS = {a: f"Q_{a}" for a in ACTIONS}
MASK_COLS = {a: f"Mask_{a}" for a in ACTIONS}
A_COLS = {a: f"A_{a}" for a in ACTIONS}


def _sanitize_W_M(W: np.ndarray, M: np.ndarray):
    """Очистить Q‑значения ``W`` и маски ``M``.

    Нефинитные элементы ``W`` при маске ``1`` считаются недопустимыми
    действиями: соответствующая маска обнуляется, а значение заменяется на
    ``0``. Строки без валидных действий помечаются для последующей фильтрации.

    Возвращает
    -------
    tuple
        ``(W_clean, M_clean, row_valid)`` где ``row_valid`` — булев массив,
        показывающий, в каких строках осталось хотя бы одно допустимое действие.
    """

    assert W.shape == M.shape
    bad = (~np.isfinite(W)) & (M > 0.0)
    if bad.any():
        M = M.copy()
        M[bad] = 0.0
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0, copy=True).astype(
            np.float32
        )
    else:
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0, copy=True).astype(
            np.float32
        )

    row_valid = M.sum(axis=1) > 0.0
    return W.astype(np.float32), M.astype(np.float32), row_valid.astype(np.bool_)


# ----------------- утилиты -----------------


def _clean_soft_labels(Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    Y = np.clip(Y, 0.0, 1.0, out=Y)
    s = Y.sum(axis=1, keepdims=True)
    ok = (s > eps).squeeze(1)
    Y[ok] = Y[ok] / s[ok]
    if (~ok).any():
        Y[~ok] = 1.0 / Y.shape[1]
    return Y


def _clean_masks(M: np.ndarray) -> np.ndarray:
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return (M >= 0.5).astype(np.float32)


def extract_features(df: pd.DataFrame, drop_cols=None):
    drop = set(drop_cols or [])
    exclude_prefixes = ("Q_", "Mask_", "A_")
    cols = [
        c
        for c in df.columns
        if (c not in drop)
        and (not any(c.startswith(p) for p in exclude_prefixes))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[cols].to_numpy(np.float32), cols


def _split_indices(n: int, ratios=(0.7, 0.15, 0.15)):
    assert abs(sum(ratios) - 1.0) < 1e-8
    n_tr = int(n * ratios[0])
    n_va = int(n * (ratios[0] + ratios[1]))
    return 0, n_tr, n_va, n


# ----------------- Y/M/W/R из вашего df -----------------


def build_W_M_Y_R_from_df(
    df: pd.DataFrame,
    labels_from: Literal["q", "a"] = "q",
    tau: float = 0.5,
    r_mode: Literal["teacher", "oracle"] = "teacher",
):
    """Extract arrays W, M, Y and R from a DataFrame."""

    cols = set(df.columns)

    if all(Q_COLS[a] in cols for a in ACTIONS):
        W_raw = df[[Q_COLS[a] for a in ACTIONS]].to_numpy(np.float32)
    else:
        W_raw = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    if all(MASK_COLS[a] in cols for a in ACTIONS):
        M_raw = df[[MASK_COLS[a] for a in ACTIONS]].to_numpy(np.float32)
    elif "Pos" in cols:
        pos = df["Pos"].to_numpy()
        inpos = pos != 0
        flat = ~inpos
        M_raw = np.column_stack([flat, inpos, inpos, flat]).astype(np.float32)
    else:
        M_raw = np.ones((len(df), NUM_CLASSES), dtype=np.float32)

    M_raw = _clean_masks(M_raw)

    W, M, row_valid = _sanitize_W_M(W_raw, M_raw)

    if labels_from == "q":
        very_neg = -1e9
        logits = W / max(tau, 1e-6)
        logits = np.where(M > 0.0, logits, very_neg)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        expv = np.exp(logits) * M
        denom = expv.sum(axis=1, keepdims=True)
        Y = np.where(denom > 0.0, expv / np.maximum(denom, 1e-8), 1.0 / NUM_CLASSES)
    else:
        Y = df[[A_COLS[a] for a in ACTIONS]].to_numpy(np.float32)
        Y = _clean_soft_labels(Y * M)

    if r_mode == "teacher":
        very_neg = -1e9
        logits_for_R = np.where(M > 0.0, W, very_neg)
        a_star = np.argmax(logits_for_R, axis=1)
        # выбираем действие teacher'а только среди допустимых; иначе R=0
        R = np.where(
            row_valid,
            W[np.arange(len(W)), a_star],
            0.0,
        ).astype(np.float32)
    else:
        R = np.max(np.where(M > 0.0, W, -np.inf), axis=1)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return (
        W.astype(np.float32),
        M.astype(np.float32),
        Y.astype(np.float32),
        R.astype(np.float32),
    )


# ----------------- генераторы sample_weight -----------------


def gen_sw_R_based(R: np.ndarray, power: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    w = np.power(np.abs(R) + eps, power)
    return w.astype(np.float32)


def gen_sw_volume_based(
    vol: np.ndarray, mode: Literal["log1p", "raw"] = "log1p", eps: float = 1e-8
) -> np.ndarray:
    v = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "log1p":
        v = np.log1p(np.maximum(v, 0.0))
    return v.astype(np.float32)


def gen_sw_class_balancing(
    Y: np.ndarray, train_slice: slice, eps: float = 1e-8
) -> np.ndarray:
    Ytr = Y[train_slice]
    freq = Ytr.sum(axis=0) / max(Ytr.sum(), eps)
    invfreq = 1.0 / np.maximum(freq, eps)
    w = (Y * invfreq[None, :]).sum(axis=1)
    return w.astype(np.float32), invfreq.astype(np.float32)


def normalize_and_clip_weights(
    w: np.ndarray, train_slice: slice, wmin: float = 1e-3, wmax: float = 100.0
) -> np.ndarray:
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, wmin, wmax)
    mu = (
        np.mean(w[train_slice])
        if (train_slice.stop - train_slice.start) > 0
        else np.mean(w)
    )
    mu = max(mu, 1e-8)
    w = w / mu
    return w.astype(np.float32)


def _window_segment(
    X: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    seq_len: int,
    stride: int = 1,
    SW: Optional[np.ndarray] = None,
    return_index: bool = False,
    start_offset: int = 0,
    valid_indices: Optional[np.ndarray] = None,
):
    assert X.ndim == 2 and Y.ndim == 2 and M.ndim == 2 and W.ndim == 2 and R.ndim == 1
    N, D = X.shape
    C = Y.shape[1]
    if N < seq_len:
        zX = np.empty((0, seq_len, D), np.float32)
        zC = np.empty((0, C), np.float32)
        zR = np.empty((0,), np.float32)
        zSW = None if SW is None else np.empty((0,), np.float32)
        if return_index:
            zI = np.empty((0,), np.int64)
            return zX, zC, zC, zC, zR, zSW, zI
        return zX, zC, zC, zC, zR, zSW

    starts = np.arange(0, N - seq_len + 1, dtype=np.int64)
    if stride > 1:
        starts = starts[::stride]
    K = len(starts)

    offs = np.arange(seq_len, dtype=np.int64)[None, :]
    idx = starts[:, None] + offs

    Xw = X[idx, :]
    Yv = Y[idx, :]
    Mv = M[idx, :]
    Wv = W[idx, :]
    Rv = R[idx]
    if SW is not None:
        SWv = SW[idx]

    Yw = Yv[:, -1, :]
    Mw = Mv[:, -1, :]
    Ww = Wv[:, -1, :]
    Rw = Rv[:, -1]
    SWw = None if SW is None else SWv[:, -1].astype(np.float32)

    idx_end = idx[:, -1]
    idx_end_global = idx_end + start_offset

    keep = Mw.sum(axis=1) > 0.0
    if valid_indices is not None:
        valid_idx = np.asarray(valid_indices, dtype=np.int64)
        if valid_idx.size == 0:
            keep = np.zeros_like(keep, dtype=bool)
        else:
            keep = keep & np.isin(idx_end_global, valid_idx)
    if keep.any():
        Xw = Xw[keep]
        Yw = Yw[keep]
        Mw = Mw[keep]
        Ww = Ww[keep]
        Rw = Rw[keep]
        if SWw is not None:
            SWw = SWw[keep]
        if return_index:
            idx_end = idx_end_global[keep]
    else:
        zX = np.empty((0, seq_len, D), np.float32)
        zC = np.empty((0, C), np.float32)
        zR = np.empty((0,), np.float32)
        zSW = None if SW is None else np.empty((0,), np.float32)
        if return_index:
            zI = np.empty((0,), np.int64)
            return zX, zC, zC, zC, zR, zSW, zI
        return zX, zC, zC, zC, zR, zSW

    Kf = Xw.shape[0]
    assert Yw.shape == (Kf, C) and Mw.shape == (Kf, C) and Ww.shape == (Kf, C) and Rw.shape == (Kf,)
    if SWw is not None:
        assert SWw.shape == (Kf,)
    out = (
        Xw.astype(np.float32),
        Yw.astype(np.float32),
        Mw.astype(np.float32),
        Ww.astype(np.float32),
        Rw.astype(np.float32),
        None if SWw is None else SWw.astype(np.float32),
    )
    if return_index:
        out = out + (idx_end.astype(np.int64),)
    return out


def build_tf_dataset(
    Xw,
    Yw,
    Mw,
    Ww,
    Rw,
    SW=None,
    batch_size=256,
    shuffle=True,
    cache=True,
):
    import tensorflow as tf  # локальный импорт, чтобы избежать жёсткой зависимости при импорте модуля

    if SW is None:
        ds = tf.data.Dataset.from_tensor_slices((Xw, (Yw, Mw, Ww, Rw)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (Xw, (Yw, Mw, Ww, Rw, SW.astype(np.float32).reshape(-1)))
        )
    if cache:
        # кешируем перед перемешиванием, чтобы каждую эпоху перемешивать заново
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(min(len(Xw), 100_000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _assert_shapes_align(Xw, Yw, Mw, Ww, Rw, SW=None):
    n = len(Xw)
    assert Yw.shape[0] == n and Mw.shape[0] == n and Ww.shape[0] == n and Rw.shape[0] == n
    if SW is not None:
        assert len(SW) == n, f"SW length {len(SW)} != {n}"


def _downsample_hold_wait(
    X,
    Y,
    M,
    W,
    R,
    SW=None,
    I=None,
    *,
    betta: float = 0.0,
    rng=None,
):
    if betta <= 0.0:
        return (X, Y, M, W, R, SW) if I is None else (X, Y, M, W, R, SW, I)
    cls = np.argmax(Y, axis=1)
    hw_mask = np.isin(
        cls, [ACTIONS.index("Hold"), ACTIONS.index("Wait")]
    )
    n_hw = int(hw_mask.sum())
    if n_hw == 0:
        return (X, Y, M, W, R, SW) if I is None else (X, Y, M, W, R, SW, I)
    rng = rng or np.random.default_rng(0)
    drop = int(n_hw * betta)
    if drop <= 0:
        return (X, Y, M, W, R, SW) if I is None else (X, Y, M, W, R, SW, I)
    hw_idx = np.where(hw_mask)[0]
    drop_idx = rng.choice(hw_idx, size=drop, replace=False)
    keep = np.ones(len(Y), dtype=bool)
    keep[drop_idx] = False
    X = X[keep]
    Y = Y[keep]
    M = M[keep]
    W = W[keep]
    R = R[keep]
    if SW is not None:
        SW = SW[keep]
    if I is not None:
        I = I[keep]
        return X, Y, M, W, R, SW, I
    return X, Y, M, W, R, SW


# ----------------- основной класс -----------------


@dataclass
class DatasetBuilderForYourColumns:
    seq_len: int
    feature_cols: List[str]
    account_cols: List[str]
    stride: int = 1
    norm: Literal["zscore", "minmax", "robust", "none"] = "zscore"
    labels_from: Literal["q", "a"] = "q"
    tau: float = 0.5
    r_mode: Literal["teacher", "oracle"] = "teacher"
    splits: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    batch_size: int = 256
    drop_cols: Optional[List[str]] = None

    betta: float = 0.0
    sw_mode: Optional[Literal["R", "Volume", "ClassBalance"]] = None
    sw_R_power: float = 1.0
    sw_volume_col: str = "Volume"
    sw_volume_mode: Literal["log1p", "raw"] = "log1p"
    sw_clip_min: float = 1e-3
    sw_clip_max: float = 100.0

    stats_features: Optional[NormalizationStats] = None
    feature_names: Optional[List[str]] = None
    account_names: Optional[List[str]] = None
    invfreq_: Optional[np.ndarray] = None

    def fit_transform(self, df: pd.DataFrame, return_indices: bool = False):
        X = df[self.feature_cols].to_numpy(np.float32)
        A = df[self.account_cols].to_numpy(np.float32)
        A[:] = 0 # заменяем все на нули, чтобы не подсказывать на обучении
        if self.drop_cols:
            X = df[
                [c for c in self.feature_cols if c not in set(self.drop_cols)]
            ].to_numpy(np.float32)
            self.feature_cols = [
                c for c in self.feature_cols if c not in set(self.drop_cols)
            ]
        W, M, Y, R = build_W_M_Y_R_from_df(
            df, labels_from=self.labels_from, tau=self.tau, r_mode=self.r_mode
        )

        s0, s1, s2, s3 = _split_indices(len(X), ratios=self.splits)
        train_slice = slice(s0, s1)
        sw_train_slice = slice(train_slice.start + self.seq_len - 1, train_slice.stop)

        if self.norm == "none":
            Xn = X
        else:
            self.stats_features = NormalizationStats(kind=self.norm).fit(X[train_slice])
            Xn = self.stats_features.transform(X)
        Xall = np.concatenate([Xn, A], axis=1)

        SW_full = None
        if self.sw_mode is not None:
            if self.sw_mode == "R":
                SW_full = gen_sw_R_based(R, power=self.sw_R_power)
            elif self.sw_mode == "Volume":
                if self.sw_volume_col not in df.columns:
                    raise ValueError(
                        f"Колонка {self.sw_volume_col} не найдена для Volume‑весов"
                    )
                SW_full = gen_sw_volume_based(
                    df[self.sw_volume_col].to_numpy(), mode=self.sw_volume_mode
                )
            elif self.sw_mode == "ClassBalance":
                SW_full, invfreq = gen_sw_class_balancing(Y, train_slice=sw_train_slice)
                self.invfreq_ = invfreq
            else:
                raise ValueError("sw_mode ∈ {None,'R','Volume','ClassBalance'}")
            SW_full = normalize_and_clip_weights(
                SW_full,
                train_slice=sw_train_slice,
                wmin=self.sw_clip_min,
                wmax=self.sw_clip_max,
            )

        def cut(Xs, Ys, Ms, Ws, Rs, start, end):
            kwargs = dict(
                seq_len=self.seq_len,
                stride=self.stride,
                SW=None if SW_full is None else SW_full[start:end],
                return_index=return_indices,
                start_offset=start,
            )
            return _window_segment(
                Xs[start:end],
                Ys[start:end],
                Ms[start:end],
                Ws[start:end],
                Rs[start:end],
                **kwargs,
            )

        tr = cut(Xall, Y, M, W, R, s0, s1)
        va = cut(Xall, Y, M, W, R, s1, s2)
        te = cut(Xall, Y, M, W, R, s2, s3)
        if self.betta > 0.0:
            tr = _downsample_hold_wait(*tr, betta=self.betta)
            # va = _downsample_hold_wait(*va, betta=self.betta)
            # te = _downsample_hold_wait(*te, betta=self.betta)

        if return_indices:
            (Xtr, Ytr, Mtr, Wtr, Rtr, SWtr, Itr) = tr
            (Xva, Yva, Mva, Wva, Rva, SWva, Iva) = va
            (Xte, Yte, Mte, Wte, Rte, SWte, Ite) = te
        else:
            (Xtr, Ytr, Mtr, Wtr, Rtr, SWtr) = tr
            (Xva, Yva, Mva, Wva, Rva, SWva) = va
            (Xte, Yte, Mte, Wte, Rte, SWte) = te

        self.feature_names = list(self.feature_cols)
        self.account_names = list(self.account_cols)

        if return_indices:
            return {
                "train": (Xtr, Ytr, Mtr, Wtr, Rtr, SWtr, Itr),
                "val": (Xva, Yva, Mva, Wva, Rva, SWva, Iva),
                "test": (Xte, Yte, Mte, Wte, Rte, SWte, Ite),
            }
        return {
            "train": (Xtr, Ytr, Mtr, Wtr, Rtr, SWtr),
            "val": (Xva, Yva, Mva, Wva, Rva, SWva),
            "test": (Xte, Yte, Mte, Wte, Rte, SWte),
        }

    def as_tf_datasets(self, splits):
        (Xtr, Ytr, Mtr, Wtr, Rtr, SWtr, *_) = splits["train"]
        (Xva, Yva, Mva, Wva, Rva, SWva, *_) = splits["val"]
        (Xte, Yte, Mte, Wte, Rte, SWte, *_) = splits["test"]
        _assert_shapes_align(Xtr, Ytr, Mtr, Wtr, Rtr, SWtr)

        ds_tr = build_tf_dataset(
            Xtr,
            Ytr,
            Mtr,
            Wtr,
            Rtr,
            SW=SWtr,
            batch_size=self.batch_size,
            shuffle=True,
            cache=True,
        )
        ds_va = build_tf_dataset(
            Xva,
            Yva,
            Mva,
            Wva,
            Rva,
            SW=SWva,
            batch_size=self.batch_size,
            shuffle=False,
            cache=True,
        )
        ds_te = build_tf_dataset(
            Xte,
            Yte,
            Mte,
            Wte,
            Rte,
            SW=SWte,
            batch_size=self.batch_size,
            shuffle=False,
            cache=True,
        )
        return ds_tr, ds_va, ds_te


__all__ = ["DatasetBuilderForYourColumns"]
