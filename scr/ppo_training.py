"""PPO training loop built on top of the residual LSTM model.

The module wires together the pretrained residual LSTM policy from
``residual_lstm.py`` with the ``BacktestEnv`` environment.  It adds
teacher‑guided KL regularisation with a decaying coefficient and profit‑based
early stopping to fine‑tune the supervised model via reinforcement learning.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .backtest_env import BacktestEnv
from .residual_lstm import apply_action_mask, build_stacked_residual_lstm

NUM_ACTIONS = 4


def build_actor_critic(
    seq_len: int, feature_dim: int, num_actions: int = NUM_ACTIONS
) -> Tuple[keras.Model, keras.Model]:
    """Create actor and critic networks sharing the same architecture."""

    actor = build_stacked_residual_lstm(seq_len, feature_dim, num_classes=num_actions)
    critic = build_stacked_residual_lstm(seq_len, feature_dim, num_classes=1)
    return actor, critic


@dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    old_logp: np.ndarray
    masks: np.ndarray


def collect_trajectories(
    env: BacktestEnv,
    actor: keras.Model,
    critic: keras.Model,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Trajectory:
    """Roll out the current policy and compute advantages via GAE."""

    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    rew_buf: List[float] = []
    val_buf: List[float] = []
    logp_buf: List[float] = []
    mask_buf: List[np.ndarray] = []

    obs_dict = env.reset()
    for _ in range(batch_size):
        feat = obs_dict["features"].reshape(seq_len, feature_dim)
        mask = env.action_mask()
        logits = actor(feat[None, ...], training=False)
        masked = apply_action_mask(logits, mask[None, :])
        probs = tf.nn.softmax(masked, axis=-1).numpy()[0]
        action = int(np.random.choice(NUM_ACTIONS, p=probs))
        logp = float(np.log(probs[action] + 1e-8))
        value = float(critic(feat[None, ...], training=False).numpy()[0, 0])

        next_obs, reward, done, _ = env.step(action)
        obs_buf.append(feat)
        act_buf.append(action)
        rew_buf.append(reward)
        val_buf.append(value)
        logp_buf.append(logp)
        mask_buf.append(mask.astype(np.float32))

        obs_dict = next_obs
        if done:
            break

    # bootstrap value
    last_val = (
        critic(obs_dict["features"].reshape(seq_len, feature_dim)[None, ...], training=False)
        .numpy()[0, 0]
        if not getattr(env, "done", False)
        else 0.0
    )
    vals = np.append(val_buf, last_val)

    # Generalised Advantage Estimation
    adv = np.zeros_like(rew_buf, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rew_buf))):
        delta = rew_buf[t] + gamma * vals[t + 1] - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + vals[:-1]

    return Trajectory(
        obs=np.array(obs_buf, dtype=np.float32),
        actions=np.array(act_buf, dtype=np.int32),
        advantages=adv,
        returns=ret.astype(np.float32),
        old_logp=np.array(logp_buf, dtype=np.float32),
        masks=np.array(mask_buf, dtype=np.float32),
    )


def ppo_update(
    actor: keras.Model,
    critic: keras.Model,
    traj: Trajectory,
    actor_opt: keras.optimizers.Optimizer,
    critic_opt: keras.optimizers.Optimizer,
    clip_ratio: float = 0.2,
    c1: float = 0.5,
    c2: float = 0.01,
    epochs: int = 5,
    batch_size: int = 32,
    teacher: Optional[keras.Model] = None,
    kl_coef: float = 0.1,
    kl_decay: float = 0.99,
) -> Tuple[float, Dict[str, float]]:
    """Perform several epochs of PPO updates.
    Returns the updated KL coefficient and a dictionary of metrics."""

    obs = tf.convert_to_tensor(traj.obs)
    acts = tf.convert_to_tensor(traj.actions)
    adv = tf.convert_to_tensor(traj.advantages)
    ret = tf.convert_to_tensor(traj.returns)
    old_logp = tf.convert_to_tensor(traj.old_logp)
    masks = tf.convert_to_tensor(traj.masks)

    dataset = tf.data.Dataset.from_tensor_slices((obs, acts, adv, ret, old_logp, masks))
    dataset = dataset.shuffle(len(traj.actions)).batch(batch_size)
    pi_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    teacher_kls: List[float] = []
    approx_kls: List[float] = []
    clipfracs: List[float] = []

    for _ in range(epochs):
        for batch in dataset:
            b_obs, b_act, b_adv, b_ret, b_old, b_mask = batch
            b_adv = (b_adv - tf.reduce_mean(b_adv)) / (
                tf.math.reduce_std(b_adv) + 1e-8
            )

            with tf.GradientTape(persistent=True) as tape:
                logits = actor(b_obs, training=True)
                masked = apply_action_mask(logits, b_mask)
                logp_all = tf.nn.log_softmax(masked, axis=-1)
                logp_act = tf.reduce_sum(
                    tf.one_hot(b_act, NUM_ACTIONS) * logp_all, axis=-1
                )
                ratio = tf.exp(logp_act - b_old)
                clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                policy_loss = -tf.reduce_mean(
                    tf.minimum(ratio * b_adv, clipped * b_adv)
                )

                entropy = -tf.reduce_mean(
                    tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=-1)
                )

                value = critic(b_obs, training=True)[:, 0]
                value_loss = tf.reduce_mean(tf.square(b_ret - value))
                t_kl = tf.constant(0.0)
                if teacher is not None and kl_coef > 0.0:
                    t_logits = teacher(b_obs, training=False)
                    t_masked = apply_action_mask(t_logits, b_mask)
                    t_logp = tf.nn.log_softmax(t_masked, axis=-1)
                    t_kl = tf.reduce_mean(
                        tf.reduce_sum(tf.exp(logp_all) * (logp_all - t_logp), axis=-1)
                    )
                    actor_loss = policy_loss + kl_coef * t_kl - c2 * entropy
                else:
                    actor_loss = policy_loss - c2 * entropy

                critic_loss = c1 * value_loss

            a_grads = tape.gradient(actor_loss, actor.trainable_variables)
            c_grads = tape.gradient(critic_loss, critic.trainable_variables)
            actor_opt.apply_gradients(zip(a_grads, actor.trainable_variables))
            critic_opt.apply_gradients(zip(c_grads, critic.trainable_variables))
            pi_losses.append(float(policy_loss))
            v_losses.append(float(value_loss))
            entropies.append(float(entropy))
            teacher_kls.append(float(t_kl))
            approx_kls.append(float(tf.reduce_mean(b_old - logp_act)))
            clipfracs.append(
                float(
                    tf.reduce_mean(
                        tf.cast(tf.abs(ratio - 1.0) > clip_ratio, tf.float32)
                    )
                )
            )

    metrics = {
        "policy_loss": float(np.mean(pi_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
        "teacher_kl": float(np.mean(teacher_kls)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clipfracs)),
    }
    return kl_coef * kl_decay, metrics


def evaluate_profit(
    env: BacktestEnv, actor: keras.Model, seq_len: int, feature_dim: int
) -> float:
    """Run policy in the environment and return final equity."""

    obs = env.reset()
    while True:
        feat = obs["features"].reshape(seq_len, feature_dim)
        mask = env.action_mask()
        logits = actor(feat[None, ...], training=False)
        masked = apply_action_mask(logits, mask[None, :])
        action = int(tf.argmax(masked, axis=-1)[0])
        obs, _, done, _ = env.step(action)
        if done:
            break
    return float(env.equity)

def train(
    train_env: BacktestEnv,
    test_env: BacktestEnv,
    seq_len: int,
    feature_dim: int,
    actor_weights: str,
    save_path: str = "ppo",
    total_steps: int = 1024,
    teacher_kl: float = 0.1,
    kl_decay: float = 0.99,
    early_stop_patience: int = 10,
):
    """High level training routine."""

    actor, critic = build_actor_critic(seq_len, feature_dim)
    teacher = build_stacked_residual_lstm(seq_len, feature_dim, num_classes=NUM_ACTIONS)
    actor.load_weights(actor_weights)
    teacher.load_weights(actor_weights)
    teacher.trainable = False

    actor_opt = keras.optimizers.Adam(3e-4)
    critic_opt = keras.optimizers.Adam(1e-3)
    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs"))

    steps = 0
    kl_coef = teacher_kl
    best_profit = -np.inf
    wait = 0
    best_actor_path = os.path.join(save_path, "actor_best.h5")
    best_critic_path = os.path.join(save_path, "critic_best.h5")

    while steps < total_steps and wait < early_stop_patience:
        traj = collect_trajectories(
            train_env, actor, critic, batch_size=256, seq_len=seq_len, feature_dim=feature_dim
        )
        kl_coef, metrics = ppo_update(
            actor,
            critic,
            traj,
            actor_opt,
            critic_opt,
            teacher=teacher,
            kl_coef=kl_coef,
            kl_decay=kl_decay,
        )
        steps += len(traj.actions)
        avg_ret = float(np.mean(traj.returns))
        profit = evaluate_profit(test_env, actor, seq_len, feature_dim)
        if profit > best_profit:
            best_profit = profit
            wait = 0
            actor.save_weights(best_actor_path)
            critic.save_weights(best_critic_path)
        else:
            wait += 1
        print(
            f"step={steps} avg_reward={avg_ret:.3f} profit={profit:.3f} kl_coef={kl_coef:.4f}"
        )
        with writer.as_default():
            tf.summary.scalar("avg_return", avg_ret, step=steps)
            tf.summary.scalar("profit", profit, step=steps)
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=steps)
    writer.flush()

    actor.save_weights(os.path.join(save_path, "actor.h5"))
    critic.save_weights(os.path.join(save_path, "critic.h5"))
    actor.load_weights(best_actor_path)
    critic.load_weights(best_critic_path)

    # Inference on test data
    evaluate_profit(test_env, actor, seq_len, feature_dim)
    fig = test_env.plot("PPO inference")
    os.makedirs("results", exist_ok=True)
    fig.savefig(os.path.join("results", "ppo_inference.png"))
    return actor, critic


__all__ = [
    "build_actor_critic",
    "collect_trajectories",
    "ppo_update",
    "train",
]

