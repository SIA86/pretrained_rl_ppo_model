import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.dataset_builder import DatasetBuilderForYourColumns


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

