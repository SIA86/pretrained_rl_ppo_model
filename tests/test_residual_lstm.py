import numpy as np
import tensorflow as tf
import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.residual_lstm import (
    build_stacked_residual_lstm,
    masked_logits_and_probs,
    masked_categorical_crossentropy,
    masked_accuracy,
)


def test_model_output_shape_and_inputs():
    model = build_stacked_residual_lstm(seq_len=5, feature_dim=3, units_per_layer=(4, 4))
    assert len(model.inputs) == 1
    x = tf.random.normal((2, 5, 3))
    logits = model(x)
    assert logits.shape == (2, 4)


def test_apply_action_mask_and_softmax():
    logits = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    mask = tf.constant([[1.0, 0.0, 1.0]], dtype=tf.float32)
    masked, probs = masked_logits_and_probs(logits, mask)
    assert masked[0, 1] < -1e8
    assert tf.reduce_sum(probs).numpy() == pytest.approx(1.0)
    assert probs[0, 1].numpy() == pytest.approx(0.0, abs=1e-6)


def test_masked_loss_and_accuracy_with_sample_weights():
    logits = tf.math.log(tf.constant([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32))
    mask = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    sw = tf.constant([1.0, 2.0], dtype=tf.float32)

    loss = masked_categorical_crossentropy(y_true, logits, mask, sample_w=sw)
    acc = masked_accuracy(y_true, logits, mask, sample_w=sw)

    # manual computation
    VERY_NEG = -1e9
    logits_np = logits.numpy()
    mask_np = mask.numpy()
    y_np = y_true.numpy()
    sw_np = sw.numpy()

    masked_logits = np.where(mask_np > 0, logits_np, VERY_NEG)
    expv = np.exp(masked_logits)
    expv *= mask_np
    probs = expv / np.maximum(expv.sum(axis=1, keepdims=True), 1e-8)
    y_masked = y_np * mask_np
    y_sum = y_masked.sum(axis=1, keepdims=True)
    y_norm = np.where(y_sum > 0, y_masked / np.maximum(y_sum, 1e-8), y_masked)
    per_sample = -(y_norm * np.log(np.maximum(probs, 1e-8))).sum(axis=1)
    has_valid = mask_np.sum(axis=1) > 0
    per_sample = np.where(has_valid, per_sample, 0.0)
    loss_exp = (per_sample * sw_np).sum() / (sw_np * has_valid).sum()

    masked_logits_np = np.where(mask_np > 0, logits_np, VERY_NEG)
    pred = masked_logits_np.argmax(axis=1)
    true_cls = (y_np * mask_np).argmax(axis=1)
    correct = (pred == true_cls).astype(np.float32)
    correct = np.where(has_valid, correct, 0.0)
    acc_exp = (correct * sw_np).sum() / (sw_np * has_valid).sum()

    assert loss.numpy() == pytest.approx(loss_exp, rel=1e-5)
    assert acc.numpy() == pytest.approx(acc_exp, rel=1e-5)
