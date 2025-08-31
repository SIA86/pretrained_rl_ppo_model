from __future__ import annotations

from typing import Dict, List

import optuna
import tensorflow as tf
from optuna import Trial

from .residual_lstm import build_stacked_residual_lstm
from .train_eval import fit_model


def optimize_hyperparameters(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    seq_len: int,
    feature_dim: int,
    n_trials: int = 10,
    epochs: int = 10,
) -> Dict[str, object]:
    """Optimize network hyperparameters using Optuna.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset yielding batches as produced by dataset_builder.
    val_ds : tf.data.Dataset
        Validation dataset in the same format as ``train_ds``.
    seq_len : int
        Sequence length for the model input.
    feature_dim : int
        Number of features per timestep.
    n_trials : int, optional
        Number of Optuna trials to run, by default 10.
    epochs : int, optional
        Number of epochs for each trial, by default 10.

    Returns
    -------
    Dict[str, object]
        Best hyperparameters: ``units_per_layer``, ``dropout`` and ``lr``.
    """

    def objective(trial: Trial) -> float:
        units = [
            trial.suggest_int("units_l1", 32, 256, step=32),
            trial.suggest_int("units_l2", 32, 256, step=32),
            trial.suggest_int("units_l3", 32, 256, step=32),
        ]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        model = build_stacked_residual_lstm(
            seq_len=seq_len,
            feature_dim=feature_dim,
            units_per_layer=units,
            dropout=dropout,
        )

        history = fit_model(
            model,
            train_ds,
            val_ds,
            epochs=epochs,
            lr=lr,
            early_stopping_patience=1,
            lr_mode=None,
            best_path="best_lstm.weights.h5",
        )

        return float(history["val"]["loss"][-1])

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best_units: List[int] = [best["units_l1"], best["units_l2"], best["units_l3"]]
    return {
        "units_per_layer": best_units,
        "dropout": best["dropout"],
        "lr": best["lr"],
    }
