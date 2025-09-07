from __future__ import annotations

from typing import Dict, List

import optuna
import tensorflow as tf
from optuna import Trial

from .residual_lstm import build_backbone, build_head, NUM_CLASSES
from .train_eval import fit_model


def optimize_hyperparameters(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    seq_len: int,
    feature_dim: int,
    acc_dim: int,
    n_trials: int = 10,
    epochs: int = 10,
) -> Dict[str, object]:
    """Оптимизировать гиперпараметры сети с помощью Optuna.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Обучающий датасет, предоставляющий батчи формата dataset_builder.
    val_ds : tf.data.Dataset
        Валидационный датасет в том же формате, что ``train_ds``.
    seq_len : int
        Длина входной последовательности.
    feature_dim : int
        Число признаков на шаг.
    n_trials : int, optional
        Количество проб Optuna, по умолчанию 10.
    epochs : int, optional
        Число эпох на каждую пробу, по умолчанию 10.

    Returns
    -------
    Dict[str, object]
        Лучшие гиперпараметры: ``units_per_layer``, ``dropout`` и ``lr``.
    """

    def objective(trial: Trial) -> float:
        units = [
            trial.suggest_int("units_l1", 32, 256, step=32),
            trial.suggest_int("units_l2", 32, 256, step=32),
            trial.suggest_int("units_l3", 32, 256, step=32),
        ]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        backbone = build_backbone(
            seq_len=seq_len,
            feature_dim=feature_dim + acc_dim,
            units_per_layer=units,
            dropout=dropout,
        )
        model = build_head(backbone, NUM_CLASSES)

        def _merge(ds: tf.data.Dataset) -> tf.data.Dataset:
            def _map(inputs, target):
                feats, acc = inputs
                acc_rep = tf.repeat(acc[:, None, :], repeats=seq_len, axis=1)
                feats = tf.concat([feats, acc_rep], axis=-1)
                return feats, target

            return ds.map(_map)

        train_merged = _merge(train_ds)
        val_merged = _merge(val_ds)

        history = fit_model(
            model,
            train_merged,
            val_merged,
            epochs=epochs,
            lr=lr,
            early_stopping_patience=1,
            lr_mode=None,
            best_path="best_lstm.weights.h5",
            backbone_path="best_backbone.weights.h5",
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
