import collections
import os
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scr.backtest_env import EnvConfig
import scr.ppo_training as ppo_training
from scr.ppo_training import (
    _prepare_validation_windows,
    collect_trajectories,
    ppo_update,
    train,
)
from scr.residual_lstm import build_backbone, build_head, VERY_NEG


def make_df():
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    return pd.DataFrame({"Open": np.ones(20), "feat": np.zeros(20)}, index=idx)


def make_cfg():
    return EnvConfig(
        mode=1,
        fee=0.0,
        spread=0.0,
        leverage=1.0,
        max_steps=1,
        reward_scale=1.0,
        use_log_reward=False,
        time_penalty=0.0,
        hold_penalty=0.0,
    )


def build_models(seq_len, feature_dim, num_actions=4):
    actor_backbone = build_backbone(seq_len, feature_dim)
    critic_backbone = build_backbone(seq_len, feature_dim)
    actor = build_head(actor_backbone, num_actions)
    critic = build_head(critic_backbone, 1)
    return actor, critic


def test_get_actor_logits_fn_caches():
    seq_len = 2
    feature_dim = 7
    actor, _ = build_models(seq_len, feature_dim)

    fn1, dtype1 = ppo_training._get_actor_logits_fn(actor, seq_len, feature_dim)
    fn2, dtype2 = ppo_training._get_actor_logits_fn(actor, seq_len, feature_dim)

    assert fn1 is fn2
    assert dtype1 == dtype2 == tf.as_dtype(actor.compute_dtype)


def test_build_and_collect():
    train_df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    traj = collect_trajectories(
        train_df,
        actor,
        critic,
        cfg,
        feat_cols,
        n_env=2,
        rollout=2,
        seq_len=seq_len,
        num_actions=4,
    )
    assert traj.obs.shape == (4, seq_len, feature_dim)
    assert traj.actions.shape == (4,)
    assert traj.returns.shape == (4,)
    assert traj.advantages.shape == (4,)


def test_collect_trajectories_parallel():
    train_df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    traj = collect_trajectories(
        train_df,
        actor,
        critic,
        cfg,
        feat_cols,
        n_env=2,
        rollout=2,
        seq_len=seq_len,
        num_actions=4,
    )
    assert traj.obs.shape == (4, seq_len, feature_dim)


def test_ppo_update_kl_decay():
    train_df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    teacher_backbone = build_backbone(seq_len, feature_dim)
    teacher = build_head(teacher_backbone, 4)
    traj = collect_trajectories(
        train_df,
        actor,
        critic,
        cfg,
        feat_cols,
        n_env=1,
        rollout=2,
        seq_len=seq_len,
        num_actions=4,
    )
    opt_a = keras.optimizers.Adam(1e-3)
    opt_c = keras.optimizers.Adam(1e-3)
    new_kl, new_c2, metrics = ppo_update(
        actor,
        critic,
        traj,
        opt_a,
        opt_c,
        num_actions=4,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher=teacher,
        kl_coef=0.1,
        kl_decay=0.5,
        entropy_decay=0.1,
    )
    assert np.isclose(new_kl, 0.05)
    assert np.isclose(new_c2, 0.001)
    assert "teacher_kl" in metrics


def test_collect_trajectories_nan_probs():
    class DummyActor(tf.keras.Model):
        def call(self, inputs, training=False):
            batch = tf.shape(inputs)[0]
            return tf.tile(
                tf.constant([[np.nan, VERY_NEG, np.nan, VERY_NEG]], dtype=tf.float32),
                [batch, 1],
            )

    train_df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    critic = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, feature_dim)),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ]
    )
    traj = collect_trajectories(
        train_df,
        DummyActor(),
        critic,
        cfg,
        feat_cols,
        n_env=1,
        rollout=1,
        seq_len=seq_len,
        num_actions=4,
    )
    assert np.all(np.isfinite(traj.old_logp))


def test_collect_trajectories_index_ranges():
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    df = pd.DataFrame({"Open": np.ones(20), "feat": np.arange(20)}, index=idx)
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    rollout = 2
    needed = max(cfg.max_steps + seq_len, rollout)
    interval = pd.DatetimeIndex(df.index[5 : 5 + needed])
    traj = collect_trajectories(
        df,
        actor,
        critic,
        cfg,
        feat_cols,
        index_ranges=[interval],
        n_env=1,
        rollout=rollout,
        seq_len=seq_len,
        num_actions=4,
    )
    assert traj.obs.shape == (rollout, seq_len, feature_dim)
    assert traj.obs[0, 0, 0] == 5


def test_collect_trajectories_invalid_range():
    df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    interval = pd.DatetimeIndex([df.index[0]])
    with pytest.raises(ValueError):
        collect_trajectories(
            df,
            actor,
            critic,
            cfg,
            feat_cols,
            index_ranges=[interval],
            n_env=1,
            rollout=2,
            seq_len=seq_len,
            num_actions=4,
        )


def test_prepare_validation_windows_short_interval():
    df = make_df()
    cfg = make_cfg()
    short_interval = pd.DatetimeIndex(df.index[:2])
    windows = _prepare_validation_windows(df, [short_interval], cfg.max_steps + 1)
    assert len(windows) == 1
    assert len(windows[0]) == len(short_interval)


def test_prepare_validation_windows_truncate_long_interval():
    df = make_df()
    cfg = make_cfg()
    long_interval = pd.DatetimeIndex(df.index[:5])
    windows = _prepare_validation_windows(df, [long_interval], cfg.max_steps + 1)
    assert len(windows) == 1
    assert len(windows[0]) == cfg.max_steps + 1


def test_train_validation_windows_unique_index_ranges(monkeypatch, tmp_path):
    df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    index_ranges = [
        pd.DatetimeIndex(df.index[start : start + cfg.max_steps + 1])
        for start in (0, 3, 6)
    ]

    original_choice = ppo_training.np.random.choice
    calls = []

    def tracking_choice(a, size=None, replace=True, p=None):
        result = original_choice(a, size=size, replace=replace, p=p)
        calls.append((a, size, replace, np.array(result, copy=True)))
        return result

    monkeypatch.setattr(ppo_training.np.random, "choice", tracking_choice)

    train(
        df,
        df,
        cfg,
        cfg,
        feature_dim,
        feat_cols,
        seq_len,
        teacher_weights="",
        critic_weights="",
        backbone_weights="",
        save_path=str(tmp_path),
        num_actions=4,
        units=16,
        dropout=0.0,
        updates=0,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        entropy_decay=1.0,
        max_grad_norm=1.0,
        target_kl=1.0,
        val_interval=1,
        n_validations=len(index_ranges),
        index_ranges=index_ranges,
    )

    assert len(calls) == 1
    a, size, replace, result = calls[0]
    assert a == len(index_ranges)
    assert size == len(index_ranges)
    assert replace is False
    assert np.unique(result).size == len(index_ranges)


def test_train_validation_windows_not_enough_candidates(monkeypatch, tmp_path):
    df = make_df().iloc[:2]
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    original_choice = ppo_training.np.random.choice
    calls = []

    def tracking_choice(a, size=None, replace=True, p=None):
        calls.append((a, size, replace))
        return original_choice(a, size=size, replace=replace, p=p)

    monkeypatch.setattr(ppo_training.np.random, "choice", tracking_choice)

    train(
        df,
        df,
        cfg,
        cfg,
        feature_dim,
        feat_cols,
        seq_len,
        teacher_weights="",
        critic_weights="",
        backbone_weights="",
        save_path=str(tmp_path),
        num_actions=4,
        units=16,
        dropout=0.0,
        updates=0,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        entropy_decay=1.0,
        max_grad_norm=1.0,
        target_kl=1.0,
        val_interval=1,
        n_validations=2,
    )

    assert len(calls) == 1
    a, size, replace = calls[0]
    assert a == 1
    assert size == 1
    assert replace is False


def test_train_uses_all_validation_windows(monkeypatch, tmp_path):
    df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    window_counts = collections.Counter()
    lock = threading.Lock()

    def fake_eval(window_df, **kwargs):
        window_id = (window_df.index[0], window_df.index[-1])
        with lock:
            window_counts[window_id] += 1
        return ppo_training.EvalCacheEntry(metrics={"Realized PnL": 0.0})

    monkeypatch.setattr(ppo_training, "_evaluate_window", fake_eval)

    def fake_collect(*args, **kwargs):
        return ppo_training.Trajectory(
            obs=np.zeros((1, seq_len, feature_dim), dtype=np.float32),
            actions=np.zeros(1, dtype=np.int32),
            advantages=np.zeros(1, dtype=np.float32),
            returns=np.zeros(1, dtype=np.float32),
            old_logp=np.zeros(1, dtype=np.float32),
            masks=np.ones(1, dtype=np.float32),
        )

    monkeypatch.setattr(ppo_training, "collect_trajectories", fake_collect)

    def fake_update(*args, **kwargs):
        return (
            kwargs["kl_coef"],
            kwargs["c2"],
            {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "teacher_kl": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            },
        )

    monkeypatch.setattr(ppo_training, "ppo_update", fake_update)

    train(
        df,
        df,
        cfg,
        cfg,
        feature_dim,
        feat_cols,
        seq_len,
        teacher_weights="",
        critic_weights="",
        backbone_weights="",
        save_path=str(tmp_path),
        num_actions=4,
        units=16,
        dropout=0.0,
        updates=1,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        entropy_decay=1.0,
        max_grad_norm=1.0,
        target_kl=1.0,
        val_interval=1,
        n_validations=0,
    )

    val_required = cfg.max_steps + 1
    expected_windows = len(df) - val_required + 1
    assert len(window_counts) == expected_windows
    assert all(count == 2 for count in window_counts.values())


def test_train_logs_to_pdf(monkeypatch, tmp_path):
    df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5

    class DummyEnv:
        def __init__(self, idx: int):
            self.idx = idx

        def plot(self, title: str, show: bool = True):
            fig, ax = plt.subplots()
            ax.set_title(f"{title}-{self.idx}")
            ax.plot([0, 1], [0, 1])
            if show:
                plt.show()
            return fig

    def fake_eval(window_df, **kwargs):
        keep_env = kwargs.get("keep_env", False)
        window_id = hash((window_df.index[0], window_df.index[-1]))
        entry = ppo_training.EvalCacheEntry(
            metrics={"Realized PnL": float(window_id % 5)}
        )
        if keep_env:
            entry.env = DummyEnv(window_id)
        return entry

    monkeypatch.setattr(ppo_training, "_evaluate_window", fake_eval)

    def fake_collect(*args, **kwargs):
        return ppo_training.Trajectory(
            obs=np.zeros((1, seq_len, feature_dim), dtype=np.float32),
            actions=np.zeros(1, dtype=np.int32),
            advantages=np.zeros(1, dtype=np.float32),
            returns=np.zeros(1, dtype=np.float32),
            old_logp=np.zeros(1, dtype=np.float32),
            masks=np.ones(1, dtype=np.float32),
        )

    monkeypatch.setattr(ppo_training, "collect_trajectories", fake_collect)

    def fake_update(*args, **kwargs):
        return (
            kwargs["kl_coef"],
            kwargs["c2"],
            {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "teacher_kl": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            },
        )

    monkeypatch.setattr(ppo_training, "ppo_update", fake_update)

    log_dir = tmp_path / "pdf_logs"

    train(
        df,
        df,
        cfg,
        cfg,
        feature_dim,
        feat_cols,
        seq_len,
        teacher_weights="",
        critic_weights="",
        backbone_weights="",
        save_path=str(tmp_path),
        num_actions=4,
        units=16,
        dropout=0.0,
        updates=1,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        entropy_decay=1.0,
        max_grad_norm=1.0,
        target_kl=1.0,
        val_interval=1,
        n_validations=2,
        log_to_pdf_path=str(log_dir),
    )

    pdf_files = list(log_dir.glob("validation_step_*.pdf"))
    assert len(pdf_files) == 1
    assert pdf_files[0].stat().st_size > 0


def test_train_freeze_backbones(tmp_path):
    df = make_df()
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    weight_path = tmp_path / "weights.weights.h5"
    backbone = build_backbone(seq_len, feature_dim, units=64, dropout=0.5)
    model = build_head(backbone, 4)
    model.save_weights(weight_path)
    actor, critic, _, _ = train(
        df,
        df,
        cfg,
        cfg,
        feature_dim,
        feat_cols,
        seq_len,
        teacher_weights=str(weight_path),
        critic_weights="",
        backbone_weights=str(weight_path),
        save_path=str(tmp_path),
        num_actions=4,
        units=64,
        dropout=0.5,
        updates=1,
        n_env=1,
        rollout=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        clip_ratio=0.2,
        c1=0.5,
        c2=0.01,
        epochs=1,
        batch_size=1,
        teacher_kl=0.1,
        kl_decay=0.5,
        entropy_decay=1.0,
        max_grad_norm=1.0,
        target_kl=1.0,
        val_interval=1,
        fine_tune=True,
    )
    actor_layers = [l for l in actor.layers if l.name.startswith("feat_")]
    critic_layers = [l for l in critic.layers if l.name.startswith("feat_")]
    assert actor_layers and critic_layers
    assert not any(layer.trainable for layer in actor_layers)
    assert not any(layer.trainable for layer in critic_layers)
