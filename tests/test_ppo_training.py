import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scr.backtest_env import EnvConfig
from scr.ppo_training import (
    build_actor_critic,
    collect_trajectories,
    ppo_update,
    prepare_datasets,
)
from scr.residual_lstm import build_backbone, build_head, VERY_NEG


def make_df():
    return pd.DataFrame({"close": np.ones(20), "feat": np.zeros(20)})


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


def test_build_and_collect():
    df = make_df()
    train_df, _, _, feat_cols, state_stats = prepare_datasets(df)
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_actor_critic(seq_len, feature_dim)
    traj = collect_trajectories(
        train_df,
        actor,
        critic,
        cfg,
        feat_cols,
        state_stats,
        n_env=2,
        rollout=2,
        seq_len=seq_len,
    )
    assert traj.obs.shape == (4, seq_len, feature_dim)
    assert traj.actions.shape == (4,)
    assert traj.returns.shape == (4,)
    assert traj.advantages.shape == (4,)


def test_ppo_update_kl_decay():
    df = make_df()
    train_df, _, _, feat_cols, state_stats = prepare_datasets(df)
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    actor, critic = build_actor_critic(seq_len, feature_dim)
    teacher_backbone = build_backbone(seq_len, feature_dim)
    teacher = build_head(teacher_backbone, 4)
    traj = collect_trajectories(
        train_df,
        actor,
        critic,
        cfg,
        feat_cols,
        state_stats,
        n_env=1,
        rollout=2,
        seq_len=seq_len,
    )
    opt_a = keras.optimizers.Adam(1e-3)
    opt_c = keras.optimizers.Adam(1e-3)
    new_coef, metrics = ppo_update(
        actor,
        critic,
        traj,
        opt_a,
        opt_c,
        teacher=teacher,
        kl_coef=0.1,
        kl_decay=0.5,
        epochs=1,
        batch_size=1,
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

    df = make_df()
    train_df, _, _, feat_cols, state_stats = prepare_datasets(df)
    cfg = make_cfg()
    seq_len = 1
    feature_dim = len(feat_cols) + 5
    critic = keras.Sequential(
        [keras.layers.Input(shape=(seq_len, feature_dim)), keras.layers.Flatten(), keras.layers.Dense(1)]
    )
    traj = collect_trajectories(
        train_df,
        DummyActor(),
        critic,
        cfg,
        feat_cols,
        state_stats,
        n_env=1,
        rollout=1,
        seq_len=seq_len,
    )
    assert np.all(np.isfinite(traj.old_logp))
