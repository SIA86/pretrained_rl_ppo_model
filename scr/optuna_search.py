"""Optuna-based hyperparameter search utilities."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional

import joblib
import optuna

from .dataset_builder import DatasetBuilderForYourColumns
from .residual_lstm import build_stacked_residual_lstm
from .train_eval import fit_model

# Default discrete search space for hyperparameters
DEFAULT_SPACE: Dict[str, Sequence] = {
    "BATCH_SIZE": [64],
    "UNITS_PER_LAYER": [(128, 128, 64)],
    "DROPOUT": [0.3],
    "LR": [3e-4],
    "GRAD_CLIP_NORM": [1.0],
    "EPOCHS": [10],
    "EARLY_STOPPING_PATIENCE": [5],
    "LR_MODE": ["onecycle"],
    "ONECYCLE_MAX_LR": [3e-3],
    "ONECYCLE_PCT_START": [0.5],
    "LR_RESTART_PATIENCE": [4],
    "LR_RESTART_SHRINK": [0.5],
}


def optimize_hyperparams(
    path_to_data: str,
    n_trials: int = 20,
    seq_len: int = 30,
    max_rows: Optional[int] = None,
    search_space: Optional[Dict[str, Sequence]] = None,
) -> Dict[str, Any]:
    """Search for best hyperparameters using Optuna.

    Parameters
    ----------
    path_to_data:
        Path to ``.joblib`` file containing a pandas ``DataFrame``.
    n_trials:
        Number of Optuna trials.
    seq_len:
        Sequence length for the dataset builder.
    max_rows:
        Optional limit on number of rows loaded from the dataset for faster
        experimentation.
    search_space:
        Optional dictionary overriding :data:`DEFAULT_SPACE`.

    Returns
    -------
    dict
        Dictionary of best found hyperparameters.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = joblib.load(path_to_data)
    if max_rows is not None:
        df = df.head(max_rows)

    space = DEFAULT_SPACE.copy()
    if search_space:
        space.update(search_space)

    def objective(trial: optuna.Trial) -> float:
        params = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}

        builder = DatasetBuilderForYourColumns(seq_len=seq_len, batch_size=params["BATCH_SIZE"])
        splits = builder.fit_transform(df)
        ds_tr, ds_va, _ = builder.as_tf_datasets(splits)
        feat_dim = splits["train"][0].shape[-1]

        model = build_stacked_residual_lstm(
            seq_len=seq_len,
            feature_dim=feat_dim,
            units_per_layer=params["UNITS_PER_LAYER"],
            dropout=params["DROPOUT"],
        )

        best_path = f"optuna_trial_{trial.number}.weights.h5"
        history = fit_model(
            model,
            ds_tr,
            ds_va,
            epochs=params["EPOCHS"],
            lr=params["LR"],
            grad_clip_norm=params["GRAD_CLIP_NORM"],
            early_stopping_patience=params["EARLY_STOPPING_PATIENCE"],
            lr_mode=params["LR_MODE"],
            onecycle_max_lr=params["ONECYCLE_MAX_LR"],
            onecycle_pct_start=params["ONECYCLE_PCT_START"],
            lr_restart_patience=params["LR_RESTART_PATIENCE"],
            lr_restart_shrink=params["LR_RESTART_SHRINK"],
            best_path=best_path,
        )
        val_loss = history["val"]["loss"][-1]

        try:
            import os
            os.remove(best_path)
        except OSError:
            pass
        return float(val_loss)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

__all__ = ["optimize_hyperparams"]
