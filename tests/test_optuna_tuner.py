import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scr.optuna_tuner import optimize_hyperparameters


def _make_dataset(samples: int = 20, seq_len: int = 5, feature_dim: int = 3):
    x = tf.random.normal((samples, seq_len, feature_dim))
    y = tf.random.uniform((samples,), maxval=4, dtype=tf.int32)
    y = tf.one_hot(y, 4)
    m = tf.ones_like(y)
    ds = tf.data.Dataset.from_tensor_slices((x, (y, m)))
    return ds.batch(4)


def test_optuna_tune_runs():
    train_ds = _make_dataset()
    val_ds = _make_dataset()
    params = optimize_hyperparameters(
        train_ds, val_ds, seq_len=5, feature_dim=3, n_trials=1, epochs=1
    )
    assert set(params) == {"units_per_layer", "dropout", "lr"}
    assert len(params["units_per_layer"]) == 3
