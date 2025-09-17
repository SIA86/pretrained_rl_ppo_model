import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scr.backtest_env import EnvConfig
from scr.ppo_training import collect_trajectories, ppo_update, train
from scr.residual_lstm import build_backbone, build_head, VERY_NEG


def make_df():
    return pd.DataFrame({"Open": np.ones(20), "feat": np.zeros(20)})


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
    new_coef, metrics = ppo_update(
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
    )
    assert np.isclose(new_coef, 0.05)
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
    df = pd.DataFrame({"Open": np.ones(20), "feat": np.arange(20)})
    feat_cols = ["feat"]
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_models(seq_len, feature_dim)
    rollout = 2
    needed = max(cfg.max_steps + seq_len, rollout)
    traj = collect_trajectories(
        df,
        actor,
        critic,
        cfg,
        feat_cols,
        index_ranges=[(5, needed)],
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
    with pytest.raises(ValueError):
        collect_trajectories(
            df,
            actor,
            critic,
            cfg,
            feat_cols,
            index_ranges=[(0, 1)],
            n_env=1,
            rollout=2,
            seq_len=seq_len,
            num_actions=4,
        )


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
