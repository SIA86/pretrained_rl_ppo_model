"""High level training and evaluation helpers.

This module bundles together a small training loop, evaluation helpers and a
few utilities for visualising the results.  The implementation is adapted from
the user's specification and is deliberately light‑weight so it can be reused
in notebooks or scripts.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .dataset_builder import ACTIONS, NUM_CLASSES
from .residual_lstm import (
    apply_action_mask,
    masked_accuracy,
    masked_categorical_crossentropy,
)


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------


class CosineWithWarmup:
    """Cosine schedule with optional linear warmup."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        self.base_lr = float(base_lr)
        self.total = int(total_steps)
        self.warm = int(warmup_steps)
        self.min_lr = float(min_lr)

    @tf.function
    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        s_int = tf.cast(step, tf.int32)
        s_mod = tf.math.floormod(s_int, tf.cast(self.total, tf.int32))
        step = tf.cast(s_mod, tf.float32)
        warm = tf.cast(tf.maximum(self.warm, 1), tf.float32)

        lr_warm = (
            self.base_lr * tf.minimum(step / warm, 1.0)
            if self.warm > 0
            else self.base_lr
        )
        t = tf.clip_by_value(
            (step - warm) / tf.maximum(1.0, tf.cast(self.total - self.warm, tf.float32)),
            0.0,
            1.0,
        )
        cosine = 0.5 * (1.0 + tf.cos(3.1415926535 * t))
        lr_cos = self.min_lr + (lr_warm - self.min_lr) * cosine
        return tf.where(step < warm, self.base_lr * step / warm, lr_cos)


class OneCycleLR:
    """Simplified OneCycle learning rate schedule."""

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        base_lr: Optional[float] = None,
        min_lr: Optional[float] = None,
        final_div_factor: float = 1e2,
    ) -> None:
        self.total = int(total_steps)
        self.pct_start = float(pct_start)
        self.up_steps = max(1, int(self.total * self.pct_start))
        self.down_steps = max(1, self.total - self.up_steps)
        self.max_lr = float(max_lr)
        self.base_lr = float(base_lr if base_lr is not None else max_lr / 25.0)
        self.min_lr = float(min_lr if min_lr is not None else self.base_lr / 25.0)
        self.final_lr = self.min_lr / float(final_div_factor)

    @tf.function
    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        s_int = tf.cast(step, tf.int32)
        s_mod = tf.math.floormod(s_int, tf.cast(self.total, tf.int32))
        s = tf.cast(s_mod, tf.float32)
        up = tf.cast(self.up_steps, tf.float32)
        down = tf.cast(self.down_steps, tf.float32)

        def _cos(a: float, b: float, t: tf.Tensor) -> tf.Tensor:
            return b + 0.5 * (a - b) * (1 + tf.cos(3.1415926535 * t))

        lr_up = _cos(self.base_lr, self.max_lr, 1.0 - tf.minimum(s / up, 1.0))
        s_down = tf.maximum(0.0, s - up)
        lr_down = _cos(self.final_lr, self.min_lr, tf.minimum(s_down / down, 1.0))
        return tf.where(s < up, lr_up, lr_down)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_batch(batch: Any) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None]:
    """Unpack batches produced by :func:`dataset_builder.build_tf_dataset`.

    The dataset can provide ``((X, M), Y)`` or ``((X, M), (Y, W, R))`` or
    ``((X, M), (Y, W, R, SW))``.  The function returns ``X, M, Y, W, R, SW``
    where ``W``, ``R`` and ``SW`` may be ``None``.
    """

    (x, m), y = batch
    W = R = SW = None
    if isinstance(y, (tuple, list)):
        y, *extra = y
        if len(extra) == 1:
            t = extra[0]
            if t.shape.rank == 2 and t.shape[-1] == NUM_CLASSES:
                W = t
            else:
                SW = t
        elif len(extra) == 2:
            a, b = extra
            if a.shape.rank == 2 and a.shape[-1] == NUM_CLASSES:
                W = a
                if b.shape.rank == 1:
                    R = b
                else:
                    SW = b
            else:
                R = a
                SW = b
        elif len(extra) >= 3:
            W, R, SW = extra[:3]
    return x, m, y, W, R, SW


def expected_return_metric(
    logits: tf.Tensor,
    mask: tf.Tensor,
    W: tf.Tensor,
    sample_w: tf.Tensor | None = None,
) -> tf.Tensor:
    """Expected return under the model's policy."""

    probs = tf.nn.softmax(apply_action_mask(logits, mask), axis=-1)
    per_ex = tf.reduce_sum(probs * W * mask, axis=-1)
    has_valid = tf.reduce_any(mask > 0.0, axis=-1)
    per_ex = tf.where(has_valid, per_ex, tf.zeros_like(per_ex))
    if sample_w is not None:
        sw = tf.cast(tf.reshape(sample_w, [-1]), tf.float32)
        num = tf.reduce_sum(per_ex * sw)
        den = tf.reduce_sum(sw * tf.cast(has_valid, tf.float32)) + 1e-8
        return num / den
    else:
        num = tf.reduce_sum(per_ex)
        den = tf.reduce_sum(tf.cast(has_valid, tf.float32)) + 1e-8
        return num / den


def f1_per_class(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    mask: tf.Tensor,
    sample_w: tf.Tensor | None = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute macro F1 and per-class F1 scores."""

    masked_logits = apply_action_mask(logits, mask)
    pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    true = tf.argmax(y_true * mask, axis=-1, output_type=tf.int32)
    has_label = tf.reduce_sum(y_true * mask, axis=-1) > 0.0
    pred = tf.boolean_mask(pred, has_label)
    true = tf.boolean_mask(true, has_label)
    if sample_w is not None:
        sw = tf.boolean_mask(tf.cast(tf.reshape(sample_w, [-1]), tf.float32), has_label)
    else:
        sw = tf.ones_like(tf.cast(pred, tf.float32))
    cm = tf.math.confusion_matrix(true, pred, num_classes=NUM_CLASSES, weights=sw, dtype=tf.float32)
    tp = tf.linalg.diag_part(cm)
    fp = tf.reduce_sum(cm, axis=0) - tp
    fn = tf.reduce_sum(cm, axis=1) - tp
    precision = tf.math.divide_no_nan(tp, tp + fp)
    recall = tf.math.divide_no_nan(tp, tp + fn)
    f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1, f1


def information_coefficient(
    logits: tf.Tensor,
    mask: tf.Tensor,
    W: tf.Tensor,
    R: tf.Tensor,
    sample_w: tf.Tensor | None = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return (Pearson, Spearman) IC between predicted and true returns."""

    probs = tf.nn.softmax(apply_action_mask(logits, mask), axis=-1)
    pred = tf.reduce_sum(probs * W * mask, axis=-1)
    y = tf.cast(R, tf.float32)
    x = tf.cast(pred, tf.float32)
    if sample_w is not None:
        w = tf.cast(tf.reshape(sample_w, [-1]), tf.float32)
        w = w / (tf.reduce_sum(w) + 1e-8)
    else:
        w = None

    def _weighted_stats(a, b):
        if w is not None:
            a_mean = tf.reduce_sum(a * w)
            b_mean = tf.reduce_sum(b * w)
            cov = tf.reduce_sum((a - a_mean) * (b - b_mean) * w)
            var_a = tf.reduce_sum((a - a_mean) ** 2 * w)
            var_b = tf.reduce_sum((b - b_mean) ** 2 * w)
        else:
            a_mean = tf.reduce_mean(a)
            b_mean = tf.reduce_mean(b)
            cov = tf.reduce_mean((a - a_mean) * (b - b_mean))
            var_a = tf.reduce_mean((a - a_mean) ** 2)
            var_b = tf.reduce_mean((b - b_mean) ** 2)
        return cov, var_a, var_b

    cov, var_x, var_y = _weighted_stats(x, y)
    pearson = cov / tf.sqrt(var_x * var_y + 1e-8)

    x_rank = tf.cast(tf.argsort(tf.argsort(x)), tf.float32)
    y_rank = tf.cast(tf.argsort(tf.argsort(y)), tf.float32)
    cov_r, var_xr, var_yr = _weighted_stats(x_rank, y_rank)
    spearman = cov_r / tf.sqrt(var_xr * var_yr + 1e-8)
    return pearson, spearman


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


@tf.function
def train_one_epoch(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    global_step: tf.Variable,
    lr_schedule: Optional[callable] = None,
):
    total_loss = tf.zeros((), tf.float32)
    total_acc = tf.zeros((), tf.float32)
    total_cnt = tf.zeros((), tf.float32)

    total_er = tf.zeros((), tf.float32)
    er_cnt = tf.zeros((), tf.float32)

    for batch in train_ds:
        xb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)

        if lr_schedule is not None:
            optimizer.learning_rate.assign(lr_schedule(global_step))

        with tf.GradientTape() as tape:
            logits = model([xb, mb], training=True)
            loss = masked_categorical_crossentropy(yb, logits, mb, sample_w=SWb)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc = masked_accuracy(yb, logits, mb, sample_w=SWb)

        bs = tf.cast(tf.shape(xb)[0], tf.float32)
        total_loss += loss * bs
        total_acc += acc * bs
        total_cnt += bs

        if Wb is not None:
            er = expected_return_metric(logits, mb, Wb, sample_w=SWb)
            total_er += er * bs
            er_cnt += bs

        global_step.assign_add(1)

    out = {
        "loss": total_loss / tf.maximum(total_cnt, 1.0),
        "acc": total_acc / tf.maximum(total_cnt, 1.0),
    }
    if er_cnt > 0:
        out["exp_return"] = total_er / tf.maximum(er_cnt, 1.0)
    else:
        out["exp_return"] = tf.constant(0.0, dtype=tf.float32)
    return out


@tf.function
def validate_one_epoch(model: keras.Model, val_ds: tf.data.Dataset):
    total_loss = tf.zeros((), tf.float32)
    total_acc = tf.zeros((), tf.float32)
    total_cnt = tf.zeros((), tf.float32)

    model_er_sum = tf.zeros((), tf.float32)
    model_er_cnt = tf.zeros((), tf.float32)
    oracle_er_sum = tf.zeros((), tf.float32)
    oracle_er_cnt = tf.zeros((), tf.float32)

    f1_sum = tf.zeros([NUM_CLASSES], tf.float32)
    f1_batches = tf.zeros((), tf.float32)

    icp_sum = tf.zeros((), tf.float32)
    ics_sum = tf.zeros((), tf.float32)
    ic_batches = tf.zeros((), tf.float32)

    for batch in val_ds:
        xb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)
        logits = model([xb, mb], training=False)

        loss = masked_categorical_crossentropy(yb, logits, mb, sample_w=SWb)
        acc = masked_accuracy(yb, logits, mb, sample_w=SWb)
        bs = tf.cast(tf.shape(xb)[0], tf.float32)

        total_loss += loss * bs
        total_acc += acc * bs
        total_cnt += bs

        if Wb is not None:
            er_b = expected_return_metric(logits, mb, Wb, sample_w=SWb)
            model_er_sum += er_b * bs
            model_er_cnt += bs

            oracle_b = oracle_expected_return_batch(Wb, mb, SWb)
            oracle_er_sum += oracle_b * bs
            oracle_er_cnt += bs

        macro_f1, f1_vec = f1_per_class(yb, logits, mb, sample_w=SWb)
        f1_sum += f1_vec
        f1_batches += 1.0

        if (Wb is not None) and (Rb is not None):
            icp, ics = information_coefficient(logits, mb, Wb, Rb, sample_w=SWb)
            icp_sum += icp
            ics_sum += ics
            ic_batches += 1.0

    model_er = model_er_sum / tf.maximum(model_er_cnt, 1.0)
    oracle_er = oracle_er_sum / tf.maximum(oracle_er_cnt, 1.0)
    er_ratio = model_er / tf.maximum(oracle_er, tf.constant(1e-8, tf.float32))

    out = {
        "loss": total_loss / tf.maximum(total_cnt, 1.0),
        "acc": total_acc / tf.maximum(total_cnt, 1.0),
        "macro_f1": tf.reduce_mean(f1_sum / tf.maximum(f1_batches, 1.0)),
        "f1_per_class": (f1_sum / tf.maximum(f1_batches, 1.0)),
        "exp_return": model_er,
        "oracle_ER": oracle_er,
        "ER_ratio": er_ratio,
    }
    if ic_batches > 0:
        out["IC_pearson"] = icp_sum / tf.maximum(ic_batches, 1.0)
        out["IC_spearman"] = ics_sum / tf.maximum(ic_batches, 1.0)
    else:
        out["IC_pearson"] = tf.constant(0.0, dtype=tf.float32)
        out["IC_spearman"] = tf.constant(0.0, dtype=tf.float32)
    return out


def fit_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 50,
    steps_per_epoch: Optional[int] = None,
    lr: float = 3e-4,
    weight_decay: Optional[float] = None,
    grad_clip_norm: Optional[float] = 1.0,
    early_stopping_patience: int = 7,
    lr_mode: Optional[str] = None,
    cosine_warmup_steps: int = 0,
    cosine_min_lr: float = 0.0,
    onecycle_max_lr: Optional[float] = None,
    onecycle_pct_start: float = 0.3,
    lr_restart_patience: int = 3,
    lr_restart_shrink: float = 0.5,
    best_path: str = "best_lstm_weights.h5",
):
    """Train ``model`` and return history dict."""

    try:
        import tensorflow_addons as tfa
    except Exception:  # pragma: no cover - optional dependency
        tfa = None

    if steps_per_epoch is None:
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch < 0:
            raise ValueError("Cannot determine steps_per_epoch; specify explicitly.")

    if weight_decay and weight_decay > 0 and tfa is not None:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr, weight_decay=weight_decay, clipnorm=grad_clip_norm
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=grad_clip_norm)

    def make_schedule(factor: float = 1.0):
        base = lr * factor
        if lr_mode is None:
            return None
        if lr_mode.lower() == "cosine":
            return CosineWithWarmup(
                base_lr=base,
                total_steps=steps_per_epoch,
                warmup_steps=cosine_warmup_steps,
                min_lr=cosine_min_lr * factor,
            )
        if lr_mode.lower() == "onecycle":
            return OneCycleLR(
                max_lr=(onecycle_max_lr or base) * factor,
                total_steps=steps_per_epoch,
                pct_start=onecycle_pct_start,
            )
        raise ValueError("lr_mode must be None|'cosine'|'onecycle'")

    schedule = make_schedule(factor=1.0)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    best_val = float("inf")
    no_improve = 0
    since_restart = 0
    shrink_factor = 1.0

    history = {
        "train": {"loss": [], "acc": [], "exp_return": []},
        "val": {
            "loss": [],
            "acc": [],
            "macro_f1": [],
            "exp_return": [],
            "IC_pearson": [],
            "IC_spearman": [],
            "oracle_ER": [],
            "ER_ratio": [],
        },
    }

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_ds, global_step, lr_schedule=schedule)
        va = validate_one_epoch(model, val_ds)

        msg = (
            f"Epoch {epoch:02d} | train: loss {float(tr['loss']):.4f}, acc {float(tr['acc']):.4f}"
        )
        if 'exp_return' in tr:
            msg += f", ER {float(tr['exp_return']):.6f} | "
        else:
            msg += " | "
        msg += (
            f"val: loss {float(va['loss']):.4f}, acc {float(va['acc']):.4f}, macroF1 {float(va['macro_f1']):.4f}"
        )
        if 'exp_return' in va:
            msg += f", ER {float(va['exp_return']):.6f}"
        if 'IC_pearson' in va:
            msg += f", ICp {float(va['IC_pearson']):.4f}, ICs {float(va['IC_spearman']):.4f}"
        print(msg)

        history["train"]["loss"].append(float(tr["loss"]))
        history["train"]["acc"].append(float(tr["acc"]))
        history["train"]["exp_return"].append(float(tr.get("exp_return", float("nan"))))

        history["val"]["loss"].append(float(va["loss"]))
        history["val"]["acc"].append(float(va["acc"]))
        history["val"]["macro_f1"].append(float(va["macro_f1"]))
        history["val"]["exp_return"].append(float(va["exp_return"]))
        history["val"]["IC_pearson"].append(float(va.get("IC_pearson", float("nan"))))
        history["val"]["IC_spearman"].append(float(va.get("IC_spearman", float("nan"))))
        history["val"]["oracle_ER"].append(float(va["oracle_ER"]))
        history["val"]["ER_ratio"].append(float(va["ER_ratio"]))

        if float(va['loss']) < best_val - 1e-6:
            best_val = float(va['loss'])
            no_improve = 0
            since_restart = 0
            model.save_weights(best_path)
        else:
            no_improve += 1
            since_restart += 1

            if lr_mode is not None and since_restart >= lr_restart_patience:
                shrink_factor *= lr_restart_shrink
                schedule = make_schedule(factor=shrink_factor)
                global_step.assign(0)
                since_restart = 0
                print(f"[LR-RESTART] New cycle with factor={shrink_factor:.3f}")

            if no_improve >= early_stopping_patience:
                print(
                    f"Early stopping: no improvement for {early_stopping_patience} epochs."
                )
                break

    try:
        model.load_weights(best_path)
        print(f"Restored best weights from {best_path}")
    except Exception as e:  # pragma: no cover - best effort
        print(f"Warning: could not restore best weights: {e}")

    return history


# ---------------------------------------------------------------------------
# Evaluation / visualisation utilities
# ---------------------------------------------------------------------------


def plot_history_curves(history: dict) -> None:
    ep = np.arange(1, len(history['train']['loss']) + 1)

    plt.figure()
    plt.plot(ep, history['train']['loss'], label='train')
    plt.plot(ep, history['val']['loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ep, history['train']['acc'], label='train')
    plt.plot(ep, history['val']['acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ep, history['val']['macro_f1'], marker='o', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.title('Validation Macro-F1')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ep, history['val']['exp_return'], label='ER (model)')
    if 'oracle_ER' in history['val']:
        plt.plot(ep, history['val']['oracle_ER'], label='ER (oracle)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Expected Return')
    plt.title('ER on validation')
    plt.legend()
    plt.show()

    if 'ER_ratio' in history['val']:
        plt.figure()
        plt.plot(ep, history['val']['ER_ratio'], marker='o', label='ER_ratio')
        plt.xlabel('Epoch')
        plt.ylabel('ER / oracle_ER')
        plt.title('ER ratio (validation)')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.show()

    if 'IC_pearson' in history['val']:
        plt.figure()
        plt.plot(ep, history['val']['IC_pearson'], label='IC Pearson (val)')
        plt.plot(ep, history['val']['IC_spearman'], label='IC Spearman (val)')
        plt.xlabel('Epoch')
        plt.ylabel('IC')
        plt.title('Information Coefficient (val)')
        plt.legend()
        plt.show()


def confusion_and_f1_on_dataset(model: keras.Model, ds: tf.data.Dataset):
    from sklearn.metrics import confusion_matrix  # local import

    y_true = []
    y_pred = []
    for (xb, mb), (yb, *rest) in ds:
        logits = model([xb, mb], training=False)
        masked_logits = apply_action_mask(logits, mb)
        y_true.append(np.argmax(yb.numpy(), axis=1))
        y_pred.append(np.argmax(masked_logits.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(ACTIONS)))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS, rotation=45, ha='right')
    ax.set_yticklabels(ACTIONS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (test)')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    f1s = []
    for c in range(len(ACTIONS)):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ACTIONS)), f1s)
    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS, rotation=30, ha='right')
    ax.set_ylabel('F1')
    ax.set_title('F1 per class (test)')
    plt.tight_layout()
    plt.show()
    return cm, np.array(f1s)


@tf.function
def oracle_expected_return_batch(
    W: tf.Tensor, M: tf.Tensor, sample_w: tf.Tensor | None
) -> tf.Tensor:
    """Average oracle expected return for a batch."""

    M = tf.cast(M, W.dtype)
    very_neg = tf.constant(-1e9, dtype=W.dtype)
    masked_W = tf.where(M > 0.0, W, very_neg)
    per_ex = tf.reduce_max(masked_W, axis=-1)
    has_valid = tf.reduce_any(M > 0.0, axis=-1)
    per_ex = tf.where(has_valid, per_ex, tf.zeros_like(per_ex))
    if sample_w is not None:
        sw = tf.cast(tf.reshape(sample_w, [-1]), W.dtype)
        num = tf.reduce_sum(per_ex * sw)
        den = tf.reduce_sum(sw * tf.cast(has_valid, W.dtype)) + 1e-8
        return num / den
    num = tf.reduce_sum(per_ex)
    den = tf.reduce_sum(tf.cast(has_valid, W.dtype)) + 1e-8
    return num / den


@tf.function
def evaluate_dataset(model: keras.Model, ds: tf.data.Dataset):
    total_loss = tf.zeros((), tf.float32)
    total_acc = tf.zeros((), tf.float32)
    total_cnt = tf.zeros((), tf.float32)

    model_er_sum = tf.zeros((), tf.float32)
    model_er_cnt = tf.zeros((), tf.float32)
    oracle_er_sum = tf.zeros((), tf.float32)
    oracle_er_cnt = tf.zeros((), tf.float32)

    f1_sum = tf.zeros([NUM_CLASSES], tf.float32)
    f1_batches = tf.zeros((), tf.float32)

    icp_sum = tf.zeros((), tf.float32)
    ics_sum = tf.zeros((), tf.float32)
    ic_batches = tf.zeros((), tf.float32)

    for batch in ds:
        xb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)
        logits = model([xb, mb], training=False)

        loss = masked_categorical_crossentropy(yb, logits, mb, sample_w=SWb)
        acc = masked_accuracy(yb, logits, mb, sample_w=SWb)
        bs = tf.cast(tf.shape(xb)[0], tf.float32)

        total_loss += loss * bs
        total_acc += acc * bs
        total_cnt += bs

        if Wb is not None:
            er_b = expected_return_metric(logits, mb, Wb, sample_w=SWb)
            model_er_sum += er_b * bs
            model_er_cnt += bs

            oracle_b = oracle_expected_return_batch(Wb, mb, SWb)
            oracle_er_sum += oracle_b * bs
            oracle_er_cnt += bs

        macro_f1, f1_vec = f1_per_class(yb, logits, mb, sample_w=SWb)
        f1_sum += f1_vec
        f1_batches += 1.0

        if (Wb is not None) and (Rb is not None):
            icp, ics = information_coefficient(logits, mb, Wb, Rb, sample_w=SWb)
            icp_sum += icp
            ics_sum += ics
            ic_batches += 1.0

    out = {
        "loss": total_loss / tf.maximum(total_cnt, 1.0),
        "acc": total_acc / tf.maximum(total_cnt, 1.0),
        "macro_f1": tf.reduce_mean(f1_sum / tf.maximum(f1_batches, 1.0)),
        "f1_per_class": (f1_sum / tf.maximum(f1_batches, 1.0)),
        "exp_return": model_er_sum / tf.maximum(model_er_cnt, 1.0),
        "oracle_ER": oracle_er_sum / tf.maximum(oracle_er_cnt, 1.0),
    }
    out["ER_ratio"] = out["exp_return"] / tf.maximum(out["oracle_ER"], tf.constant(1e-8, tf.float32))

    if ic_batches > 0:
        out["IC_pearson"] = icp_sum / tf.maximum(ic_batches, 1.0)
        out["IC_spearman"] = ics_sum / tf.maximum(ic_batches, 1.0)
    else:
        out["IC_pearson"] = tf.constant(0.0, dtype=tf.float32)
        out["IC_spearman"] = tf.constant(0.0, dtype=tf.float32)
    return out


def materialize_metrics(d: dict) -> dict:
    """Convert a dict of tensors/arrays to plain Python types."""

    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, tf.Tensor):
            v = v.numpy()
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = float(v.item())
            else:
                out[k] = v.tolist()
        elif isinstance(v, (float, int)):
            out[k] = v
        else:
            try:
                out[k] = float(v)  # type: ignore[arg-type]
            except Exception:
                out[k] = v
    return out


__all__ = [
    "CosineWithWarmup",
    "OneCycleLR",
    "train_one_epoch",
    "validate_one_epoch",
    "fit_model",
    "plot_history_curves",
    "confusion_and_f1_on_dataset",
    "oracle_expected_return_batch",
    "evaluate_dataset",
    "materialize_metrics",
]

