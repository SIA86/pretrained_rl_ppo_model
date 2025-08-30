"""Metric helpers for masked action spaces.

This module provides TensorFlow implementations of several metrics used for
training and evaluating policies with action masks.  All metrics are aware of
invalid actions and optional sample weights.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from .residual_lstm import NUM_CLASSES, VERY_NEG, masked_logits_and_probs


@tf.function
def expected_return_metric(
    logits: tf.Tensor,
    mask: tf.Tensor,
    action_weights: tf.Tensor,
    sample_w: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Compute expected return of actions under the policy.

    The metric averages ``probs * action_weights`` over the batch, optionally
    weighted by ``sample_w`` and ignoring samples without any valid actions.
    """

    mask = tf.cast(mask, tf.float32)
    _, probs = masked_logits_and_probs(logits, mask)
    er = tf.reduce_sum(probs * action_weights, axis=-1)
    has_valid = tf.reduce_sum(mask, axis=-1) > 0.0
    er = tf.where(has_valid, er, tf.zeros_like(er))

    if sample_w is not None:
        sw = tf.cast(tf.reshape(sample_w, [-1]), tf.float32)
        num = tf.reduce_sum(er * sw)
        den = tf.reduce_sum(sw * tf.cast(has_valid, tf.float32)) + 1e-8
        return num / den
    else:
        num = tf.reduce_sum(er)
        den = tf.reduce_sum(tf.cast(has_valid, tf.float32)) + 1e-8
        return num / den


@tf.function
def f1_per_class(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    mask: tf.Tensor,
    sample_w: Optional[tf.Tensor] = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return mean and per-class F1 scores.

    The function uses the mask to ignore invalid actions and supports sample
    weights at the example level.
    """

    mask = tf.cast(mask, tf.float32)
    masked_logits = tf.where(mask > 0.0, logits, VERY_NEG * tf.ones_like(logits))
    pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    true_cls = tf.argmax(y_true * mask, axis=-1, output_type=tf.int32)
    has_valid = tf.reduce_sum(mask, axis=-1) > 0.0

    if sample_w is None:
        sw = tf.ones_like(tf.cast(has_valid, tf.float32))
    else:
        sw = tf.cast(tf.reshape(sample_w, [-1]), tf.float32)
    sw = tf.where(has_valid, sw, tf.zeros_like(sw))

    f1s = []
    for c in range(NUM_CLASSES):
        pred_c = tf.cast(tf.equal(pred, c), tf.float32)
        true_c = tf.cast(tf.equal(true_cls, c), tf.float32)
        tp = tf.reduce_sum(sw * pred_c * true_c)
        fp = tf.reduce_sum(sw * pred_c * (1.0 - true_c))
        fn = tf.reduce_sum(sw * (1.0 - pred_c) * true_c)
        precision = tf.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = tf.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = tf.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), 0.0)
        f1s.append(f1)
    f1_vec = tf.stack(f1s)
    return tf.reduce_mean(f1_vec), f1_vec


@tf.function
def pearson_corr(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Compute Pearson correlation coefficient."""

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    xm = x - tf.reduce_mean(x)
    ym = y - tf.reduce_mean(y)
    num = tf.reduce_sum(xm * ym)
    den = tf.sqrt(tf.reduce_sum(xm * xm) * tf.reduce_sum(ym * ym)) + eps
    return num / den


@tf.function
def spearman_corr(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Compute Spearman correlation coefficient via ranks."""

    def _ranks(v: tf.Tensor) -> tf.Tensor:
        order = tf.argsort(v, axis=0, stable=True)
        inv = tf.zeros_like(order)
        inv = tf.tensor_scatter_nd_update(
            inv, tf.expand_dims(order, 1), tf.range(tf.shape(v)[0], dtype=order.dtype)
        )
        return tf.cast(inv, tf.float32)

    return pearson_corr(_ranks(x), _ranks(y))


@tf.function
def information_coefficient(
    logits: tf.Tensor,
    mask: tf.Tensor,
    action_weights: tf.Tensor,
    realized_return: tf.Tensor,
    sample_w: Optional[tf.Tensor] = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return Pearson and Spearman information coefficients.

    Correlates expected returns from the policy with realized returns. Samples
    without any valid actions are ignored.  If fewer than two valid samples are
    present, zero is returned for both coefficients.
    """

    mask = tf.cast(mask, tf.float32)
    _, probs = masked_logits_and_probs(logits, mask)
    has_valid = tf.reduce_sum(mask, axis=-1) > 0.0
    er = tf.reduce_sum(probs * action_weights, axis=-1)
    rr = tf.cast(realized_return, tf.float32)
    er = tf.boolean_mask(er, has_valid)
    rr = tf.boolean_mask(rr, has_valid)
    cond = tf.shape(er)[0] > 1
    icp = tf.cond(cond, lambda: pearson_corr(er, rr), lambda: tf.constant(0.0))
    ics = tf.cond(cond, lambda: spearman_corr(er, rr), lambda: tf.constant(0.0))
    return icp, ics

