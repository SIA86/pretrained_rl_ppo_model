import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scr.train_eval import (
    CosineWithWarmup,
    OneCycleLR,
    _unpack_batch,
    confusion_and_f1_on_dataset,
    evaluate_dataset,
    expected_return_metric,
    materialize_metrics,
    validate_one_epoch,
    predict_logits_dataset,
)
from scr.dataset_builder import NUM_CLASSES


class DummyModel(tf.keras.Model):
    def call(self, inputs, training=False):  # pragma: no cover - simple model
        x, acc = inputs
        batch = tf.shape(x)[0]
        logits = tf.reshape(tf.constant([1.0, 0.5, -2.0]), [1, 3])
        return tf.tile(logits, [batch, 1])


def _ds_no_w(batch_size=4, batches=3, num_classes=3):
    for _ in range(batches):
        x = tf.zeros([batch_size, 5])
        acc = tf.zeros([batch_size, 2])
        m = tf.ones([batch_size, num_classes])
        y = tf.one_hot(np.random.randint(0, num_classes, size=(batch_size,)), num_classes)
        yield (x, acc), (y, m)


def _ds_sw(batch_size=4, batches=3, num_classes=3):
    for _ in range(batches):
        x = tf.zeros([batch_size, 5])
        acc = tf.zeros([batch_size, 2])
        m = tf.ones([batch_size, num_classes])
        y = tf.one_hot(np.random.randint(0, num_classes, size=(batch_size,)), num_classes)
        sw = tf.random.uniform([batch_size])
        yield (x, acc), (y, m, sw)


def _ds_masked():
    x = tf.zeros([4, 5])
    acc = tf.zeros([4, 2])
    m = tf.concat([tf.zeros([4, 1]), tf.ones([4, 2])], axis=1)
    y = tf.one_hot([1, 1, 2, 2], 3)
    yield (x, acc), (y, m)

    
def test_unpack_batch_with_sw():
    x = tf.zeros((2, 5, 3))
    acc = tf.zeros((2, 5, 2))
    m = tf.ones((2, NUM_CLASSES))
    y = tf.zeros((2, NUM_CLASSES))
    W = tf.ones((2, NUM_CLASSES))
    R = tf.ones((2,))
    SW = tf.constant([0.5, 1.0], dtype=tf.float32)
    batch = ((x, acc), (y, m, W, R, SW))
    xb, accb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)
    assert xb is x and accb is acc and mb is m and yb is y
    assert Wb is W and Rb is R and SWb is SW


def test_unpack_batch_sw_only():
    x = tf.zeros((2, 5, 3))
    acc = tf.zeros((2, 5, 2))
    m = tf.ones((2, NUM_CLASSES))
    y = tf.zeros((2, NUM_CLASSES))
    SW = tf.constant([0.5, 1.0], dtype=tf.float32)
    batch = ((x, acc), (y, m, SW))
    xb, accb, mb, yb, Wb, Rb, SWb = _unpack_batch(batch)
    assert Wb is None and Rb is None and SWb is SW


def test_lr_schedules_repeat():
    cos = CosineWithWarmup(base_lr=1.0, total_steps=5, warmup_steps=1, min_lr=0.1)
    steps = tf.range(0, 10)
    vals = tf.stack([cos(s) for s in steps])
    assert tf.reduce_all(tf.abs(vals[:5] - vals[5:]) < 1e-6)

    oc = OneCycleLR(max_lr=1.0, total_steps=6, pct_start=0.5)
    steps = tf.range(0, 12)
    vals = tf.stack([oc(s) for s in steps])
    assert tf.reduce_all(tf.abs(vals[:6] - vals[6:]) < 1e-6)


def test_expected_return_metric_simple():
    logits = tf.math.log(tf.constant([[0.5, 0.5]], dtype=tf.float32))
    mask = tf.constant([[1.0, 1.0]], dtype=tf.float32)
    W = tf.constant([[1.0, 0.0]], dtype=tf.float32)
    er = expected_return_metric(logits, mask, W)
    assert er.numpy() == pytest.approx(0.5)


def test_validate_and_evaluate_no_w():
    model = DummyModel()
    sig = (
        (
            tf.TensorSpec([None, 5], tf.float32),
            tf.TensorSpec([None, 2], tf.float32),
        ),
        (
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
        ),
    )
    ds = tf.data.Dataset.from_generator(_ds_no_w, output_signature=sig)
    out_val = validate_one_epoch(model, ds)
    out_eval = evaluate_dataset(model, ds)
    assert float(out_val["exp_return"]) == 0.0
    assert float(out_eval["exp_return"]) == 0.0


def test_validate_and_evaluate_sw_only():
    model = DummyModel()
    sig = (
        (
            tf.TensorSpec([None, 5], tf.float32),
            tf.TensorSpec([None, 2], tf.float32),
        ),
        (
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None], tf.float32),
        ),
    )
    ds = tf.data.Dataset.from_generator(_ds_sw, output_signature=sig)
    out_val = validate_one_epoch(model, ds)
    out_eval = evaluate_dataset(model, ds)
    assert float(out_val["exp_return"]) == 0.0
    assert float(out_eval["exp_return"]) == 0.0


def test_confusion_f1_respects_mask(monkeypatch):
    model = DummyModel()
    sig = (
        (
            tf.TensorSpec([None, 5], tf.float32),
            tf.TensorSpec([None, 2], tf.float32),
        ),
        (
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
        ),
    )
    ds = tf.data.Dataset.from_generator(_ds_masked, output_signature=sig)
    monkeypatch.setattr(plt, "show", lambda: None)
    cm, f1s = confusion_and_f1_on_dataset(model, ds)
    assert cm[0].sum() == 0
    assert cm.sum() == 4


def test_materialize_metrics():
    metrics = {"a": tf.constant(1.0), "b": np.array([1, 2])}
    out = materialize_metrics(metrics)
    assert out["a"] == 1.0
    assert out["b"] == [1, 2]


def test_predict_logits_dataset():
    model = DummyModel()
    sig = (
        (
            tf.TensorSpec([None, 5], tf.float32),
            tf.TensorSpec([None, 2], tf.float32),
        ),
        (
            tf.TensorSpec([None, 3], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
        ),
    )
    ds = tf.data.Dataset.from_generator(lambda: _ds_no_w(num_classes=3), output_signature=sig)
    logits = predict_logits_dataset(model, ds)
    assert logits.shape[1] == 3

