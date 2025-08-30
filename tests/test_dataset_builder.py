import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.dataset_builder import (
    ACTIONS,
    Q_COLS,
    MASK_COLS,
    DatasetBuilderForYourColumns,
    build_W_M_Y_R_from_df,
    _window_segment,
    extract_features,
)


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    }
    actions = ["Open", "Close", "Hold", "Wait"]
    for a in actions:
        data[f"Q_{a}"] = rng.normal(size=n)
        data[f"Mask_{a}"] = np.ones(n, dtype=np.float32)
    return pd.DataFrame(data)

def df_from_WM(W, M, extra_cols=None):
    d = {}
    for i, a in enumerate(ACTIONS):
        d[Q_COLS[a]] = W[:, i]
        d[MASK_COLS[a]] = M[:, i]
    if extra_cols:
        d.update(extra_cols)
    return pd.DataFrame(d)

def test_fit_transform_shapes():
    df = _make_df(30)
    builder = DatasetBuilderForYourColumns(seq_len=3, norm="none", splits=(0.5, 0.25, 0.25))
    splits = builder.fit_transform(df)
    Xtr, Ytr, Mtr, Wtr, Rtr, SWtr = splits["train"]

    assert Xtr.shape[1:] == (3, 2)
    assert Ytr.shape[1] == 4
    assert Mtr.shape == Ytr.shape
    assert Wtr.shape == Ytr.shape
    assert Rtr.shape[0] == Xtr.shape[0]
    assert SWtr is None

def test_teacher_R_all_zero_mask_rows():
    W = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, -5.0, 7.0, 1.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    M = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    df = df_from_WM(W, M)
    _, _, _, R = build_W_M_Y_R_from_df(df, labels_from="q", r_mode="teacher")
    assert np.allclose(R, np.array([4.0, 0.0, 6.0], dtype=np.float32))


def test_oracle_R_all_zero_is_zero():
    W = np.array([[1, 2, 3, 4]], dtype=np.float32)
    M = np.array([[0, 0, 0, 0]], dtype=np.float32)
    df = df_from_WM(W, M)
    _, _, _, R = build_W_M_Y_R_from_df(df, labels_from="q", r_mode="oracle")
    assert R.shape == (1,) and R[0] == 0.0


def test_window_segment_drops_windows_with_invalid_last_mask():
    N, D, C = 4, 2, 4
    X = np.arange(N * D, dtype=np.float32).reshape(N, D)
    Y = np.zeros((N, C), np.float32)
    Y[:, 0] = 1.0
    W = np.ones((N, C), np.float32)
    M = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    R = np.arange(N, dtype=np.float32)

    Xw, Yw, Mw, Ww, Rw, SWw = _window_segment(
        X, Y, M, W, R, seq_len=2, stride=1, SW=None
    )
    assert Xw.shape[0] == 1 and Yw.shape == (1, C) and Rw.shape == (1,)


def test_extract_features_excludes_Q_Mask_A_and_respects_drop():
    df = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [0.1, 0.2, 0.3],
            "Q_Open": [1, 2, 3],
            "Mask_Close": [1, 1, 1],
            "A_Hold": [0.2, 0.3, 0.5],
        }
    )
    X, cols = extract_features(df, drop_cols=["feat2"])
    assert cols == ["feat1"] and X.shape == (3, 1)


def test_fit_transform_returns_indices():
    df = _make_df(20)
    builder = DatasetBuilderForYourColumns(seq_len=3, norm="none", splits=(0.7, 0.15, 0.15))
    splits = builder.fit_transform(df, return_indices=True)
    Xte, _, _, _, _, _, idx = splits["test"]
    assert idx.shape[0] == Xte.shape[0]
    assert idx[0] == 19


