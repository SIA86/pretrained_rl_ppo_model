import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.optuna_search import optimize_hyperparams


def test_optimize_hyperparams_returns_dict():
    root = pathlib.Path(__file__).resolve().parents[1]
    params = optimize_hyperparams(
        str(root / "dummy_df.joblib"),
        n_trials=1,
        max_rows=100,
        search_space={
            "BATCH_SIZE": [4],
            "UNITS_PER_LAYER": [(8,)],
            "DROPOUT": [0.0],
            "LR": [1e-3],
            "GRAD_CLIP_NORM": [1.0],
            "EPOCHS": [1],
            "EARLY_STOPPING_PATIENCE": [1],
            "LR_MODE": [None],
            "ONECYCLE_MAX_LR": [1e-3],
            "ONECYCLE_PCT_START": [0.3],
            "LR_RESTART_PATIENCE": [1],
            "LR_RESTART_SHRINK": [0.5],
        },
    )
    assert isinstance(params, dict)
    assert "BATCH_SIZE" in params
