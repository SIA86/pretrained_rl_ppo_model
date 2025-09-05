import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from scr.backtest_env import BacktestEnv, EnvConfig
from scr.ppo_training import (
    build_actor_critic,
    collect_trajectories,
    ppo_update,
)
from scr.residual_lstm import build_stacked_residual_lstm, VERY_NEG


def make_env():
    df = pd.DataFrame({"close": [1.0, 1.0], "feat": [0.0, 0.0]})
    cfg = EnvConfig(
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
    return BacktestEnv(df, feature_cols=["feat"], cfg=cfg)


def test_build_and_collect():
    env = make_env()
    env.reset()
    feature_dim = env.features.shape[1] + env._get_state().shape[0]
    actor, critic = build_actor_critic(seq_len=1, feature_dim=feature_dim)
    traj = collect_trajectories(
        env, actor, critic, batch_size=1, seq_len=1, feature_dim=feature_dim
    )
    assert traj.obs.shape == (1, 1, feature_dim)
    assert traj.actions.shape == (1,)
    assert traj.returns.shape == (1,)
    assert traj.advantages.shape == (1,)


def test_ppo_update_kl_decay():
    env = make_env()
    env.reset()
    feature_dim = env.features.shape[1] + env._get_state().shape[0]
    actor, critic = build_actor_critic(seq_len=1, feature_dim=feature_dim)
    teacher = build_stacked_residual_lstm(1, feature_dim, num_classes=4)
    traj = collect_trajectories(
        env, actor, critic, batch_size=1, seq_len=1, feature_dim=feature_dim
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
            return tf.constant([[np.nan, VERY_NEG, np.nan, VERY_NEG]], dtype=tf.float32)

    class DummyEnv:
        def __init__(self):
            self.features = np.zeros((2, 1), dtype=np.float32)
            self.t = 1
            self.state = np.zeros(1, dtype=np.float32)
            self.done = False

        def reset(self):
            self.t = 1
            self.done = False
            return {"state": self.state.copy()}

        def step(self, action):
            self.done = True
            return {"state": self.state.copy()}, 0.0, True, {}

        def action_mask(self):
            return np.array([1, 0, 1, 0], dtype=np.float32)

    env = DummyEnv()
    critic = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    traj = collect_trajectories(
        env, DummyActor(), critic, batch_size=1, seq_len=2, feature_dim=2
    )
    assert np.all(np.isfinite(traj.old_logp))
