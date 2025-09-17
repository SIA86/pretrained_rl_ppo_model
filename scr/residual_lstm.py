"""Модель residual LSTM и утилиты, учитывающие маску действий.

Модуль реализует остаточную стековую LSTM, выдающую логиты для выбора
действия. Последовательность обрабатывается стеком LSTM, а маска валидности
применяется снаружи через :func:`apply_action_mask` перед softmax в тренировке.
Также приведены функции потерь и точности с учётом маски.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 4
VERY_NEG = -1e9


def build_backbone(
    seq_len: int,
    feature_dim: int,
    *,
    units: int = 128,
    dropout: float = 0.2,
    ln_eps: float = 1e-5,
) -> keras.Model:
    """Построить компактный однослойный LSTM-бэкбон."""

    inputs = keras.Input(shape=(seq_len, feature_dim), name="features")
    x = layers.LSTM(units, return_sequences=False, name="feat_lstm")(inputs)
    x = layers.LayerNormalization(epsilon=ln_eps, name="feat_ln")(x)
    x = layers.Dropout(dropout, name="feat_dropout")(x)
    return keras.Model(inputs=inputs, outputs=x, name="ResidualLSTMBackbone")


def build_head(
    backbone: keras.Model, num_classes: int, units: Optional[int] = None
) -> keras.Model:
    """Добавить голову к бэкбону и вернуть полную модель."""
    x = backbone.output
    if units is None:
        x = layers.Dense(
            backbone.output_shape[-1], activation="relu", name="head_dense"
        )(x)
    else:
        x = layers.Dense(units, activation="relu", name="head_dense")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs=backbone.input, outputs=logits, name="ResidualLSTM")


def apply_action_mask(
    logits: tf.Tensor, mask: tf.Tensor, very_neg: float = VERY_NEG
) -> tf.Tensor:
    """Применить маску валидности к логитам.

    Недопустимые действия (``mask == 0``) получают большое отрицательное
    смещение перед применением softmax.
    """

    # dtype-aware very negative value to avoid ``-inf`` in fp16
    logits_dtype = logits.dtype
    if logits_dtype == tf.float16:
        vneg = tf.constant(-6e4, dtype=logits_dtype)
    else:
        vneg = tf.constant(very_neg, dtype=logits_dtype)
    m = tf.cast(mask, logits_dtype)
    # replace invalid positions with a large negative constant
    return tf.where(m > 0.0, logits, vneg)


def masked_logits_and_probs(
    logits: tf.Tensor, mask: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Вернуть маскированные логиты и соответствующие вероятности."""

    masked = apply_action_mask(logits, mask)
    has_valid = tf.reduce_any(tf.cast(mask, tf.bool), axis=-1, keepdims=True)
    probs = tf.nn.softmax(masked, axis=-1)
    # if all actions invalid, return zeros to avoid NaNs
    probs = tf.where(has_valid, probs, tf.zeros_like(probs, dtype=logits.dtype))
    return masked, probs


def masked_categorical_crossentropy(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    mask: tf.Tensor,
    sample_w: Optional[tf.Tensor] = None,
    eps: float = 1e-8,
) -> tf.Tensor:
    """Кросс‑энтропия с маской действий и опциональными весами."""

    dtype = logits.dtype
    mask = tf.cast(mask, dtype)
    y_true = tf.cast(y_true, dtype)

    masked_logits = apply_action_mask(logits, mask)
    log_probs = tf.nn.log_softmax(masked_logits, axis=-1)

    y_masked = y_true * mask
    y_sum = tf.reduce_sum(y_masked, axis=-1, keepdims=True)
    y_norm = tf.where(y_sum > 0.0, y_masked / (y_sum + eps), y_masked)
    per_sample = -tf.reduce_sum(y_norm * log_probs, axis=-1)

    has_label = tf.squeeze(y_sum, axis=-1) > 0.0
    per_sample = tf.where(has_label, per_sample, tf.zeros_like(per_sample))

    if sample_w is not None:
        sw = tf.cast(tf.reshape(sample_w, [-1]), dtype)
        per_sample = per_sample * sw
        denom = tf.reduce_sum(sw * tf.cast(has_label, dtype)) + eps
    else:
        denom = tf.reduce_sum(tf.cast(has_label, dtype)) + eps

    return tf.reduce_sum(per_sample) / denom


def masked_accuracy(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    mask: tf.Tensor,
    sample_w: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Метрика точности, учитывающая маску действий и веса."""

    dtype = logits.dtype
    mask = tf.cast(mask, dtype)
    y_true = tf.cast(y_true, dtype)
    masked_logits = apply_action_mask(logits, mask)
    pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    true_cls = tf.argmax(y_true * mask, axis=-1, output_type=tf.int32)
    has_label = tf.reduce_sum(y_true * mask, axis=-1) > 0.0
    correct = tf.cast(tf.equal(pred, true_cls), tf.float32)
    correct = tf.where(has_label, correct, tf.zeros_like(correct))

    if sample_w is not None:
        sw = tf.cast(tf.reshape(sample_w, [-1]), tf.float32)
        num = tf.reduce_sum(correct * sw)
        den = tf.reduce_sum(sw * tf.cast(has_label, tf.float32)) + 1e-8
        return num / den
    else:
        num = tf.reduce_sum(correct)
        den = tf.reduce_sum(tf.cast(has_label, tf.float32)) + 1e-8

        return num / den
