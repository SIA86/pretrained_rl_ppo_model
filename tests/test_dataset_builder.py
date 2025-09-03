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
    _downsample_hold_wait,
)


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "Pos": rng.integers(-1, 2, size=n),
        "un_pnl": rng.normal(scale=0.1, size=n),
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
    builder = DatasetBuilderForYourColumns(
        seq_len=3,
        feature_cols=["f1", "f2"],
        account_cols=["Pos", "un_pnl"],
        norm="none",
        splits=(0.5, 0.25, 0.25),
    )
    splits = builder.fit_transform(df)
    Xtr, Ytr, Mtr, Wtr, Rtr, SWtr = splits["train"]

    assert Xtr.shape[1:] == (3, 4)
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


def test_build_W_M_Y_R_from_df_accepts_A_and_pos():
    df = pd.DataFrame(
        {
            "Pos": [0, 0, 1, 1],
            "A_Open": [0.7, 0.6, 0.1, 0.2],
            "A_Close": [0.1, 0.2, 0.6, 0.5],
            "A_Hold": [0.1, 0.1, 0.2, 0.2],
            "A_Wait": [0.1, 0.1, 0.1, 0.1],
        }
    )
    W, M, Y, R = build_W_M_Y_R_from_df(df, labels_from="a")
    assert W.shape == M.shape == Y.shape == (4, 4)
    expected_M = np.array(
        [[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0]],
        dtype=np.float32,
    )
    assert np.array_equal(M, expected_M)
    assert np.allclose(W, 0.0)
    expected_Y = df[[f"A_{a}" for a in ACTIONS]].to_numpy(np.float32) * expected_M
    s = expected_Y.sum(axis=1, keepdims=True)
    expected_Y = expected_Y / np.maximum(s, 1e-8)
    assert np.allclose(Y, expected_Y)
    assert np.allclose(R, 0.0)


def test_window_segment_drops_windows_with_invalid_last_mask():
    N, D, C = 4, 2, 4
    X = np.arange(N * D, dtype=np.float32).reshape(N, D)
    A = np.arange(N * 2, dtype=np.float32).reshape(N, 2)
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

    Xall = np.concatenate([X, A], axis=1)
    Xw, Yw, Mw, Ww, Rw, SWw = _window_segment(
        Xall, Y, M, W, R, seq_len=2, stride=1, SW=None
    )
    assert Xw.shape[0] == 1 and Xw.shape[2] == 4
    assert Yw.shape == (1, C) and Rw.shape == (1,)


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
    builder = DatasetBuilderForYourColumns(
        seq_len=3,
        feature_cols=["f1", "f2"],
        account_cols=["Pos", "un_pnl"],
        norm="none",
        splits=(0.7, 0.15, 0.15),
    )
    splits = builder.fit_transform(df, return_indices=True)
    Xte, _, _, _, _, _, idx = splits["test"]
    assert idx.shape[0] == Xte.shape[0]
    assert idx[0] == 19


def test_class_balance_weights_excludes_unused_rows():
    n = 10
    classes = [0, 1, 2, 3, 0, 1, 0, 0, 1, 2]
    data = {
        "f1": np.arange(n, dtype=np.float32),
        "f2": np.arange(n, dtype=np.float32),
        "Pos": np.zeros(n, dtype=np.float32),
        "un_pnl": np.zeros(n, dtype=np.float32),
    }
    for a in ACTIONS:
        data[f"Q_{a}"] = np.zeros(n, dtype=np.float32)
        data[f"Mask_{a}"] = np.ones(n, dtype=np.float32)
        data[f"A_{a}"] = np.zeros(n, dtype=np.float32)
    for i, cls in enumerate(classes):
        data[f"A_{ACTIONS[cls]}"][i] = 1.0
    df = pd.DataFrame(data)

    builder = DatasetBuilderForYourColumns(
        seq_len=3,
        feature_cols=["f1", "f2"],
        account_cols=["Pos", "un_pnl"],
        norm="none",
        labels_from="a",
        sw_mode="ClassBalance",
    )
    splits = builder.fit_transform(df)
    SWtr = splits["train"][5]

    expected_invfreq = np.array([2.5, 5.0, 5.0, 5.0], dtype=np.float32)
    expected_sw = np.array([1.25, 1.25, 0.625, 1.25, 0.625], dtype=np.float32)

    assert np.allclose(builder.invfreq_, expected_invfreq)
    assert np.allclose(SWtr, expected_sw)


def test_downsample_hold_wait():
    n = 40
    data = {
        "f1": np.zeros(n, dtype=np.float32),
        "f2": np.zeros(n, dtype=np.float32),
        "Pos": np.zeros(n, dtype=np.float32),
        "un_pnl": np.zeros(n, dtype=np.float32),
        "Volume": np.arange(n, dtype=np.float32),
    }
    for a in ACTIONS:
        data[f"Q_{a}"] = np.zeros(n, dtype=np.float32)
        data[f"Mask_{a}"] = np.ones(n, dtype=np.float32)
        data[f"A_{a}"] = np.zeros(n, dtype=np.float32)
    for i in range(10):
        data["A_Open"][i] = 1.0
    for i in range(10, 20):
        data["A_Close"][i] = 1.0
    for i in range(20, 30):
        data["A_Hold"][i] = 1.0
    for i in range(30, 40):
        data["A_Wait"][i] = 1.0
    df = pd.DataFrame(data)

    base_builder = DatasetBuilderForYourColumns(
        seq_len=1,
        feature_cols=["f1", "f2"],
        account_cols=["Pos", "un_pnl"],
        norm="none",
        labels_from="a",
        sw_mode="Volume",
        sw_volume_col="Volume",
    )
    base_splits = base_builder.fit_transform(df)

    builder = DatasetBuilderForYourColumns(
        seq_len=1,
        feature_cols=["f1", "f2"],
        account_cols=["Pos", "un_pnl"],
        norm="none",
        labels_from="a",
        sw_mode="Volume",
        sw_volume_col="Volume",
        betta=0.5,
    )
    splits = builder.fit_transform(df)

    base_val = base_splits["val"]
    down_val = splits["val"]
    expected_val = _downsample_hold_wait(*base_val, betta=0.5)

    for arr_d, arr_e in zip(down_val, expected_val):
        assert np.array_equal(arr_d, arr_e)

    cls_before = np.argmax(base_val[1], axis=1)
    hw_before = np.isin(cls_before, [2, 3]).sum()
    cls_after = np.argmax(down_val[1], axis=1)
    hw_after = np.isin(cls_after, [2, 3]).sum()
    assert hw_after == hw_before - int(hw_before * 0.5)
