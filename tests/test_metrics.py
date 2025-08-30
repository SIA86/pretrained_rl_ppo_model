import pathlib
import sys

import tensorflow as tf
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.metrics import (
    expected_return_metric,
    f1_per_class,
    pearson_corr,
    spearman_corr,
    information_coefficient,
)
from scr.residual_lstm import NUM_CLASSES


def test_expected_return_metric_with_and_without_weights():
    logits = tf.math.log(
        tf.constant(
            [
                [0.6, 0.4, 1.0, 1.0],
                [0.1, 0.9, 1.0, 1.0],
            ],
            dtype=tf.float32,
        )
    )
    mask = tf.constant([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=tf.float32)
    action_weights = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=tf.float32)

    metric = expected_return_metric(logits, mask, action_weights)
    assert metric.numpy() == pytest.approx(0.75, rel=1e-6)

    sw = tf.constant([1.0, 2.0], dtype=tf.float32)
    metric_sw = expected_return_metric(logits, mask, action_weights, sample_w=sw)
    assert metric_sw.numpy() == pytest.approx(0.8, rel=1e-6)


def test_f1_per_class():
    logits = tf.constant(
        [
            [5.0, 1.0, 1.0, 1.0],
            [1.0, 5.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 5.0],
        ],
        dtype=tf.float32,
    )
    y_true = tf.one_hot([0, 1, 2], NUM_CLASSES, dtype=tf.float32)
    mask = tf.ones_like(logits, dtype=tf.float32)

    mean_f1, f1_vec = f1_per_class(y_true, logits, mask)
    assert mean_f1.numpy() == pytest.approx(0.5, rel=1e-6)
    assert f1_vec.numpy() == pytest.approx([1.0, 1.0, 0.0, 0.0], rel=1e-6)


def test_correlations_perfect_alignment():
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    y = tf.constant([2.0, 4.0, 6.0], dtype=tf.float32)
    assert pearson_corr(x, y).numpy() == pytest.approx(1.0, rel=1e-6)
    assert spearman_corr(x, y).numpy() == pytest.approx(1.0, rel=1e-6)


def test_information_coefficient_perfect_corr():
    logits = tf.math.log(
        tf.constant(
            [
                [0.6, 0.4, 1.0, 1.0],
                [0.1, 0.9, 1.0, 1.0],
            ],
            dtype=tf.float32,
        )
    )
    mask = tf.constant([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=tf.float32)
    action_weights = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=tf.float32)
    realized = tf.constant([0.6, 0.9], dtype=tf.float32)

    icp, ics = information_coefficient(logits, mask, action_weights, realized)
    assert icp.numpy() == pytest.approx(1.0, rel=1e-6)
    assert ics.numpy() == pytest.approx(1.0, rel=1e-6)

