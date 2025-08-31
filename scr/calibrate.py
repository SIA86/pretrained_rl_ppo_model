"""Model calibration utilities via temperature scaling.

This module provides a small set of helpers to post‑hoc calibrate a
classification model that accepts ``(xb, mb)`` inputs and predicts four
actions.  The main entry point is :func:`calibrate_model` which performs
temperature scaling on a validation dataset and reports calibration
metrics.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TemperatureScaling(tf.keras.layers.Layer):
    """Calibrate logits by dividing them with a temperature parameter.

    Parameters
    ----------
    init_T:
        Initial temperature value.  Must be strictly positive.
    per_class:
        If ``True`` a separate temperature is learned for each class,
        otherwise a single scalar temperature is optimised.
    """

    def __init__(self, init_T: float = 1.0, per_class: bool = False, name: str = "temperature_scaling", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.init_T = float(init_T)
        self.per_class = bool(per_class)

    def build(self, input_shape: tf.TensorShape) -> None:
        if not (self.init_T > 0.0):
            raise ValueError(f"Temperature must be > 0, got {self.init_T}")
        k = int(input_shape[-1])
        init = tf.math.log(tf.convert_to_tensor(self.init_T, dtype=self.dtype or tf.float32))
        shape = (k,) if self.per_class else ()
        self.logT = self.add_weight(
            name="log_temperature",
            shape=shape,
            dtype=self.dtype or tf.float32,
            initializer=tf.keras.initializers.Constant(init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        scale = tf.cast(tf.exp(self.logT), x.dtype)
        return tf.math.divide_no_nan(x, scale)


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE)."""

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(confidences, bins, right=True)
    inds = np.clip(inds, 1, n_bins)  # include 0.0 and 1.0 in extreme bins
    ece = 0.0
    for b in range(1, n_bins + 1):
        m = inds == b
        if not np.any(m):
            continue
        ece += abs(accs[m].mean() - confidences[m].mean()) * m.mean()
    return float(ece)


def plot_reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15, title: str = "Reliability") -> None:
    """Plot a reliability diagram for the given probabilities."""

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    inds = np.digitize(confidences, bins, right=True)
    inds = np.clip(inds, 1, n_bins)
    acc_bin = np.zeros(n_bins)
    mask = np.zeros(n_bins, dtype=bool)
    for b in range(1, n_bins + 1):
        m = inds == b
        if np.any(m):
            acc_bin[b - 1] = accs[m].mean()
            mask[b - 1] = True
    plt.figure(figsize=(5.5, 4.0))
    plt.bar(centers[mask], acc_bin[mask], width=(1.0 / n_bins) * 0.9, alpha=0.6, label="Accuracy per bin")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Calibration workflow
# ---------------------------------------------------------------------------


def _collect_validation_arrays(val_ds: tf.data.Dataset, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise the validation dataset to NumPy arrays."""

    from .train_eval import _unpack_batch
    Xb_list, Mb_list, Y_oh_list = [], [], []
    for batch in val_ds:
        xb, mb, yb, *_ = _unpack_batch(batch)
        xb_np = xb.numpy()
        mb_np = mb.numpy()
        yb_np = yb.numpy()
        if yb_np.ndim == 2 and yb_np.shape[1] == num_classes:
            y_onehot = yb_np.astype(np.float32)
        else:
            y_idx = np.squeeze(yb_np).astype(np.int64)
            y_onehot = np.eye(num_classes, dtype=np.float32)[y_idx]
        Xb_list.append(xb_np)
        Mb_list.append(mb_np)
        Y_oh_list.append(y_onehot)
    Xb_val = np.concatenate(Xb_list, axis=0)
    Mb_val = np.concatenate(Mb_list, axis=0)
    Y_val_onehot = np.concatenate(Y_oh_list, axis=0)
    Y_val_int = Y_val_onehot.argmax(axis=1)
    return Xb_val, Mb_val, Y_val_onehot, Y_val_int


def calibrate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    *,
    per_class: bool = False,
    init_T: float = 1.0,
    n_bins: int = 15,
    batch_size: int = 2048,
    save_dir: str = "artifacts/calibration",
    plot: bool = True,
) -> Dict[str, Any]:
    """Perform temperature scaling on ``model`` using ``val_ds``.

    Returns a dictionary with the calibration metrics and the learnt
    temperature.
    """

    num_classes = int(model.output_shape[-1])
    Xb_val, Mb_val, Y_val_onehot, Y_val_int = _collect_validation_arrays(val_ds, num_classes)

    has_mask = isinstance(model.input_shape, list) and len(model.input_shape) == 2
    probe = model.predict((Xb_val[:256], Mb_val[:256]) if has_mask else Xb_val[:256], verbose=0)

    def looks_like_probs(a: np.ndarray) -> bool:
        if not np.isfinite(a).all():
            return False
        sums = a.sum(axis=1)
        return (a.min() >= -1e-6) and (a.max() <= 1.0 + 1e-6) and np.allclose(sums, 1.0, atol=1e-3)

    is_probs = looks_like_probs(probe) if isinstance(probe, np.ndarray) else False

    inp_x = tf.keras.Input(shape=Xb_val.shape[1:], name="xb")
    if has_mask:
        inp_m = tf.keras.Input(shape=Mb_val.shape[1:], name="mb")
        raw_out = model([inp_x, inp_m], training=False)
        base_inputs = [inp_x, inp_m]
    else:
        raw_out = model(inp_x, training=False)
        base_inputs = [inp_x]

    def to_logits(t: tf.Tensor) -> tf.Tensor:
        return tf.math.log(tf.clip_by_value(t, 1e-8, 1.0))

    out_logits = tf.keras.layers.Lambda(to_logits, name="to_logits")(raw_out) if is_probs else raw_out
    base_logits_model = tf.keras.Model(base_inputs, out_logits, name="base_logits_model")

    cal_logits = TemperatureScaling(init_T=init_T, per_class=per_class)(base_logits_model.outputs[0])
    cal_model = tf.keras.Model(base_logits_model.inputs, cal_logits, name="calibrated_model")
    base_logits_model.trainable = False

    cal_model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-2),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalCrossentropy(from_logits=True, name="nll")],
    )

    inputs = (Xb_val, Mb_val) if has_mask else Xb_val
    val_np_ds = (
        tf.data.Dataset.from_tensor_slices((inputs, Y_val_onehot))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="nll", patience=5, restore_best_weights=True)]
    cal_model.fit(val_np_ds, epochs=100, verbose=0, callbacks=callbacks)

    logits_before = base_logits_model.predict((Xb_val, Mb_val) if has_mask else Xb_val, batch_size=batch_size, verbose=0)
    logits_after = cal_model.predict((Xb_val, Mb_val) if has_mask else Xb_val, batch_size=batch_size, verbose=0)

    probs_before = tf.nn.softmax(logits_before).numpy()
    probs_after = tf.nn.softmax(logits_after).numpy()

    nll_before = tf.keras.losses.categorical_crossentropy(Y_val_onehot, logits_before, from_logits=True).numpy().mean().item()
    nll_after = tf.keras.losses.categorical_crossentropy(Y_val_onehot, logits_after, from_logits=True).numpy().mean().item()

    ece_before = compute_ece(probs_before, Y_val_int, n_bins=n_bins)
    ece_after = compute_ece(probs_after, Y_val_int, n_bins=n_bins)

    if plot:
        plot_reliability_diagram(probs_before, Y_val_int, n_bins=n_bins, title="Reliability before calibration")
        plot_reliability_diagram(probs_after, Y_val_int, n_bins=n_bins, title="Reliability after calibration")

    T_value = None
    for layer in cal_model.layers:
        if isinstance(layer, TemperatureScaling):
            T_value = tf.exp(layer.logT).numpy()
            break

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    temp_path = Path(save_dir) / "temperature.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"T": T_value.tolist() if hasattr(T_value, "tolist") else float(T_value)}, f, ensure_ascii=False, indent=2)

    print(f"NLL before: {nll_before:.6f} | ECE before: {ece_before:.6f}")
    print(f"NLL after:  {nll_after:.6f} | ECE after:  {ece_after:.6f}")
    print("Temperature T:", T_value)
    print("Saved:", temp_path)

    return {
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "T": T_value.tolist() if hasattr(T_value, "tolist") else float(T_value),
    }
