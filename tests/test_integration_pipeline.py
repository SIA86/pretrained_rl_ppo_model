import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.q_labels_matching import enrich_q_labels_trend_one_side
from scr.dataset_builder import DatasetBuilderForYourColumns, NUM_CLASSES
from scr.residual_lstm import build_stacked_residual_lstm
from scr.train_eval import _unpack_batch
from scr.calibrate import calibrate_model
from scr.backtest_env import run_backtest_with_logits


def _make_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prices = rng.normal(scale=0.01, size=n).cumsum() + 100
    df = pd.DataFrame(
        {
            "Open": prices + rng.normal(scale=0.01, size=n),
            "High": prices + rng.normal(scale=0.01, size=n) + 0.01,
            "Low": prices + rng.normal(scale=0.01, size=n) - 0.01,
            "Close": prices,
            "Signal_Rule": rng.integers(-1, 2, size=n),
            "Pos": np.zeros(n),
            "un_pnl": np.zeros(n),
            "flat_steps": np.zeros(n),
            "hold_steps": np.zeros(n),
            "drawdown": np.zeros(n),
        }
    )
    df = enrich_q_labels_trend_one_side(df, mode="horizon", horizon=3)
    return df


def test_full_pipeline(tmp_path):
    df = _make_df()
    builder = DatasetBuilderForYourColumns(
        seq_len=5,
        feature_cols=["Open", "High", "Low", "Close", "Signal_Rule"],
        account_cols=["Pos", "un_pnl", "flat_steps", "hold_steps", "drawdown"],
        norm="none",
        splits=(0.5, 0.25, 0.25),
        batch_size=8,
    )
    splits = builder.fit_transform(df, return_indices=True)
    ds_tr, ds_va, ds_te = builder.as_tf_datasets(splits)

    model = build_stacked_residual_lstm(
        seq_len=5,
        feature_dim=len(builder.feature_names),
        account_dim=len(builder.account_names),
        units_per_layer=(8, 8),
    )

    batch = next(iter(ds_tr.take(1)))
    xb, accb, mb, yb, *_ = _unpack_batch(batch)
    out = model([xb, accb], training=False)
    assert out.shape == (xb.shape[0], NUM_CLASSES)
    assert np.isfinite(out.numpy()).all()

    metrics = calibrate_model(
        model, ds_va, plot=False, batch_size=4, save_dir=str(tmp_path)
    )
    for key in ["T", "ece_before", "ece_after", "nll_before", "nll_after"]:
        assert key in metrics and np.isfinite(metrics[key])

    idx = splits["test"][-1]
    start = int(idx[0])
    env = run_backtest_with_logits(
        df,
        model,
        feature_stats=builder.stats_features,
        seq_len=5,
        start=start,
        feature_cols=builder.feature_names,
        price_col="Close",
        state_stats=builder.stats_account,
    )
    assert env.history
    log = env.logs()
    assert np.isfinite(log["equity"]).all()
    report = env.metrics_report()
    assert "Equity" in report and "nan" not in report.lower()

