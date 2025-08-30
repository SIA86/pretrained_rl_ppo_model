import pathlib
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.train_eval import (
    CosineWithWarmup,
    OneCycleLR,
    _unpack_batch,
    expected_return_metric,
    materialize_metrics,
)
from scr.dataset_builder import NUM_CLASSES


def test_unpack_batch_with_sw():
    x = tf.zeros((2, 5, 3))
    m = tf.ones((2, NUM_CLASSES))
    y = tf.zeros((2, NUM_CLASSES))
    W = tf.ones((2, NUM_CLASSES))
    R = tf.ones((2,))
    SW = tf.constant([0.5, 1.0], dtype=tf.float32)
    batch = ((x, m), (y, W, R, SW))
    xb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)
    assert xb is x and mb is m and yb is y
    assert Wb is W and Rb is R and SWb is SW


def test_cosine_warmup_schedule():
    sched = CosineWithWarmup(base_lr=1.0, total_steps=10, warmup_steps=2, min_lr=0.1)
    l0 = float(sched(tf.constant(0)))
    l1 = float(sched(tf.constant(1)))
    l2 = float(sched(tf.constant(2)))
    assert l0 == pytest.approx(0.0)
    assert l1 == pytest.approx(0.5)
    assert l2 <= 1.0 and l2 >= 0.1


def test_onecycle_lr_schedule():
    sched = OneCycleLR(max_lr=1.0, total_steps=10, pct_start=0.3)
    lrs = [float(sched(tf.constant(i))) for i in range(10)]
    assert max(lrs) == pytest.approx(1.0, rel=1e-6)


def test_expected_return_metric_simple():
    logits = tf.math.log(tf.constant([[0.5, 0.5]], dtype=tf.float32))
    mask = tf.constant([[1.0, 1.0]], dtype=tf.float32)
    W = tf.constant([[1.0, 0.0]], dtype=tf.float32)
    er = expected_return_metric(logits, mask, W)
    assert er.numpy() == pytest.approx(0.5)


def test_materialize_metrics():
    metrics = {"a": tf.constant(1.0), "b": np.array([1, 2])}
    out = materialize_metrics(metrics)
    assert out["a"] == 1.0
    assert out["b"] == [1, 2]

