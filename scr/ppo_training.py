"""Цикл обучения PPO поверх предобученной модели residual LSTM.

Модуль связывает предварительно обученную политику из ``residual_lstm.py`` с
средой ``BacktestEnv``. Дополнительно используется KL‑регуляризация по
teacher‑модели с затухающим коэффициентом и ранняя остановка по прибыли, что
позволяет дообучить супервизорную модель методом обучения с подкреплением.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .backtest_env import BacktestEnv, EnvConfig
from .dataset_builder import extract_features
from .normalisation import NormalizationStats
from .residual_lstm import (
    apply_action_mask,
    build_backbone,
    build_head,
    masked_logits_and_probs,
)



def build_actor_critic(
    seq_len: int,
    feature_dim: int,
    *,
    num_actions: int,
    units_per_layer: List[int],
    dropout: float,
    backbone_weights: str | None = None,
) -> Tuple[keras.Model, keras.Model]:
    """Создать сети актёра и критика с общей архитектурой и опциональной загрузкой бэкбона."""

    # Создаём две независимые копии бэкбона для актёра и критика
    actor_backbone = build_backbone(
        seq_len, feature_dim, units_per_layer=units_per_layer, dropout=dropout
    )
    critic_backbone = build_backbone(
        seq_len, feature_dim, units_per_layer=units_per_layer, dropout=dropout
    )
    # При необходимости загружаем предобученные веса бэкбона
    if backbone_weights:
        actor_backbone.load_weights(backbone_weights)
        critic_backbone.load_weights(backbone_weights)
    # На выход бэкбона навешиваются головы актёра и критика
    actor = build_head(actor_backbone, num_actions)
    critic = build_head(critic_backbone, 1)
    return actor, critic


@dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    old_logp: np.ndarray
    masks: np.ndarray


def prepare_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    norm_kind: str = "zscore",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    NormalizationStats,
]:
    """Разбить ``df`` на Train/Val/Test и нормализовать признаки."""
    assert abs(sum(splits) - 1.0) < 1e-8
    n = len(df)
    s1 = int(n * splits[0])
    s2 = int(n * (splits[0] + splits[1]))
    train_df = df[feature_cols].iloc[:s1].reset_index(drop=True).copy()
    val_df = df[feature_cols].iloc[s1:s2].reset_index(drop=True).copy()
    test_df = df[feature_cols].iloc[s2:].reset_index(drop=True).copy()

    # Считаем статистики нормализации по тренировочной части
    feat_stats = NormalizationStats(kind=norm_kind).fit(
        train_df[feature_cols].to_numpy(np.float32)
    )
    # Применяем нормализацию к каждому из подмножеств
    for part in (train_df, val_df, test_df):
        part[feature_cols] = feat_stats.transform(
            part[feature_cols].to_numpy(np.float32)
        )

    return train_df, val_df, test_df, feat_stats


def _env_step(args: Tuple[BacktestEnv, int]) -> Tuple[BacktestEnv, Dict[str, np.ndarray], float, bool]:
    """Помощник для параллельного шага среды.

    Принимает пару (env, action), выполняет ``env.step(action)`` и возвращает
    обновлённую среду, наблюдение, награду и флаг завершения.
    """

    env, action = args
    next_obs, reward, done, _ = env.step(int(action))
    return env, next_obs, reward, done


def collect_trajectories(
    train_df: pd.DataFrame,
    actor: keras.Model,
    critic: keras.Model,
    cfg: EnvConfig,
    feature_cols: List[str],
    n_env: int,
    rollout: int,
    seq_len: int,
    num_actions: int,
    gamma: float = 0.99,
    lam: float = 0.95,
    debug: bool = False,
    use_parallel: bool = False,
) -> Trajectory:
    """Собрать батч траекторий из ``n_env`` сред с пересэмплингом окон."""
    L = cfg.max_steps + 1
    if debug:
        print(
            f"collect_trajectories: n_env={n_env} rollout={rollout} seq_len={seq_len}"
        )
    max_start = len(train_df) - L
    starts = np.arange(max_start + 1)

    envs: List[BacktestEnv] = []
    state_hists: List[List[np.ndarray]] = []
    for _ in range(n_env):
        s = int(np.random.choice(starts))
        window_df = train_df.iloc[s : s + L].reset_index(drop=True)
        env = BacktestEnv(
            window_df,
            feature_cols=feature_cols,
            price_col="Open",
            cfg=cfg,
            ppo_true=True
        )
        obs = env.reset()
        hist = [obs["state"]]
        for _ in range(seq_len - 1):
            obs, _, done, _ = env.step(3)
            hist.append(obs["state"])
            if done:
                break
        envs.append(env)
        state_hists.append(hist)

    obs_buf = [[] for _ in range(n_env)]
    act_buf = [[] for _ in range(n_env)]
    rew_buf = [[] for _ in range(n_env)]
    val_buf = [[] for _ in range(n_env)]
    next_val_buf = [[] for _ in range(n_env)]
    logp_buf = [[] for _ in range(n_env)]
    mask_buf = [[] for _ in range(n_env)]
    done_buf = [[] for _ in range(n_env)]

    if use_parallel:
        with ProcessPoolExecutor(max_workers=min(n_env, os.cpu_count() or 1)) as pool:
            for _ in range(rollout):
                feats: List[np.ndarray] = []
                masks: List[np.ndarray] = []
                actions: List[int] = []
                logps: List[float] = []
                values: List[float] = []
                for i, env in enumerate(envs):
                    if env.done:
                        s = int(np.random.choice(starts))
                        window_df = train_df.iloc[s : s + L].reset_index(drop=True)
                        env = BacktestEnv(
                            window_df,
                            feature_cols=feature_cols,
                            price_col="Open",
                            cfg=cfg,
                            ppo_true=True
                        )
                        obs = env.reset()
                        hist = [obs["state"]]
                        for _ in range(seq_len - 1):
                            obs, _, done, _ = env.step(3)
                            hist.append(obs["state"])
                            if done:
                                break
                        envs[i] = env
                        state_hists[i] = hist

                    t = env.t
                    window = env.features[t - seq_len + 1 : t + 1]
                    state_window = np.stack(state_hists[i][t - seq_len + 1 : t + 1])
                    feat = np.concatenate([window, state_window], axis=1)
                    mask = env.action_mask()
                    logits = actor(feat[None, ...], training=False)
                    _, probs_tf = masked_logits_and_probs(logits, mask[None, :])
                    probs = probs_tf.numpy()[0]
                    if not np.isfinite(probs).all() or probs.sum() <= 0.0:
                        valid_actions = np.flatnonzero(mask)
                        probs = np.zeros(num_actions, dtype=np.float32)
                        probs[valid_actions] = 1.0 / len(valid_actions)
                    action = int(np.random.choice(num_actions, p=probs))
                    logp = float(np.log(probs[action] + 1e-8))
                    value = float(critic(feat[None, ...], training=False).numpy()[0, 0])
                    feats.append(feat)
                    masks.append(mask.astype(np.float32))
                    actions.append(action)
                    logps.append(logp)
                    values.append(value)

                results = list(pool.map(_env_step, zip(envs, actions)))
                for i, (env, next_obs, reward, done) in enumerate(results):
                    envs[i] = env
                    state_hists[i].append(next_obs["state"])
                    if done:
                        next_value = 0.0
                    else:
                        t2 = env.t
                        window2 = env.features[t2 - seq_len + 1 : t2 + 1]
                        state_window2 = np.stack(
                            state_hists[i][t2 - seq_len + 1 : t2 + 1]
                        )
                        feat2 = np.concatenate([window2, state_window2], axis=1)
                        next_value = float(
                            critic(feat2[None, ...], training=False).numpy()[0, 0]
                        )

                    obs_buf[i].append(feats[i])
                    act_buf[i].append(actions[i])
                    rew_buf[i].append(reward)
                    val_buf[i].append(values[i])
                    next_val_buf[i].append(next_value)
                    logp_buf[i].append(logps[i])
                    mask_buf[i].append(masks[i])
                    done_buf[i].append(done)

                    if done:
                        s = int(np.random.choice(starts))
                        window_df = train_df.iloc[s : s + L].reset_index(drop=True)
                        env = BacktestEnv(
                            window_df,
                            feature_cols=feature_cols,
                            price_col="Open",
                            cfg=cfg,
                            ppo_true=True
                        )
                        obs = env.reset()
                        hist = [obs["state"]]
                        for _ in range(seq_len - 1):
                            obs, _, done, _ = env.step(3)
                            hist.append(obs["state"])
                            if done:
                                break
                        envs[i] = env
                        state_hists[i] = hist
    else:
        for _ in range(rollout):
            for i, env in enumerate(envs):
                if env.done:
                    s = int(np.random.choice(starts))
                    window_df = train_df.iloc[s : s + L].reset_index(drop=True)
                    env = BacktestEnv(
                        window_df,
                        feature_cols=feature_cols,
                        price_col="Open",
                        cfg=cfg,
                        ppo_true=True
                    )
                    obs = env.reset()
                    hist = [obs["state"]]
                    for _ in range(seq_len - 1):
                        obs, _, done, _ = env.step(3)
                        hist.append(obs["state"])
                        if done:
                            break
                    envs[i] = env
                    state_hists[i] = hist

                t = env.t
                window = env.features[t - seq_len + 1 : t + 1]
                state_window = np.stack(state_hists[i][t - seq_len + 1 : t + 1])
                feat = np.concatenate([window, state_window], axis=1)
                mask = env.action_mask()
                logits = actor(feat[None, ...], training=False)
                _, probs_tf = masked_logits_and_probs(logits, mask[None, :])
                probs = probs_tf.numpy()[0]
                if not np.isfinite(probs).all() or probs.sum() <= 0.0:
                    valid_actions = np.flatnonzero(mask)
                    probs = np.zeros(num_actions, dtype=np.float32)
                    probs[valid_actions] = 1.0 / len(valid_actions)
                action = int(np.random.choice(num_actions, p=probs))
                logp = float(np.log(probs[action] + 1e-8))
                value = float(critic(feat[None, ...], training=False).numpy()[0, 0])
                next_obs, reward, done, _ = env.step(action)
                state_hists[i].append(next_obs["state"])
                if debug:
                    print(
                        f"env={i} t={env.t} action={action} reward={reward:.4f} done={done}"
                    )
                if done:
                    next_value = 0.0
                else:
                    t2 = env.t
                    window2 = env.features[t2 - seq_len + 1 : t2 + 1]
                    state_window2 = np.stack(
                        state_hists[i][t2 - seq_len + 1 : t2 + 1]
                    )
                    feat2 = np.concatenate([window2, state_window2], axis=1)
                    next_value = float(
                        critic(feat2[None, ...], training=False).numpy()[0, 0]
                    )

                obs_buf[i].append(feat)
                act_buf[i].append(action)
                rew_buf[i].append(reward)
                val_buf[i].append(value)
                next_val_buf[i].append(next_value)
                logp_buf[i].append(logp)
                mask_buf[i].append(mask.astype(np.float32))
                done_buf[i].append(done)

    # Объединяем накопленные буферы в единые массивы батча
    obs_arr: List[np.ndarray] = []
    act_arr: List[np.ndarray] = []
    adv_arr: List[np.ndarray] = []
    ret_arr: List[np.ndarray] = []
    logp_arr: List[np.ndarray] = []
    mask_arr: List[np.ndarray] = []
    for i in range(n_env):
        rewards = np.array(rew_buf[i], dtype=np.float32)
        values = np.array(val_buf[i], dtype=np.float32)
        next_values = np.array(next_val_buf[i], dtype=np.float32)
        dones = np.array(done_buf[i], dtype=np.bool_)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterm = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_values[t] * nonterm - values[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + values
        obs_arr.append(np.array(obs_buf[i], dtype=np.float32))
        act_arr.append(np.array(act_buf[i], dtype=np.int32))
        adv_arr.append(adv.astype(np.float32))
        ret_arr.append(ret.astype(np.float32))
        logp_arr.append(np.array(logp_buf[i], dtype=np.float32))
        mask_arr.append(np.array(mask_buf[i], dtype=np.float32))

    return Trajectory(
        obs=np.concatenate(obs_arr, axis=0),
        actions=np.concatenate(act_arr, axis=0),
        advantages=np.concatenate(adv_arr, axis=0),
        returns=np.concatenate(ret_arr, axis=0),
        old_logp=np.concatenate(logp_arr, axis=0),
        masks=np.concatenate(mask_arr, axis=0),
    )


@tf.function
def _ppo_batch_update(
    actor: keras.Model,
    critic: keras.Model,
    actor_opt: keras.optimizers.Optimizer,
    critic_opt: keras.optimizers.Optimizer,
    b_obs: tf.Tensor,
    b_act: tf.Tensor,
    b_adv: tf.Tensor,
    b_ret: tf.Tensor,
    b_old: tf.Tensor,
    b_mask: tf.Tensor,
    *,
    num_actions: int,
    clip_ratio: float,
    c1: float,
    c2: float,
    teacher: Optional[keras.Model],
    kl_coef: float,
    max_grad_norm: Optional[float],
):
    with tf.GradientTape(persistent=True) as tape:
        logits = actor(b_obs, training=True)
        masked = apply_action_mask(logits, b_mask)
        logp_all = tf.nn.log_softmax(masked, axis=-1)
        logp_act = tf.reduce_sum(
            tf.one_hot(b_act, num_actions) * logp_all, axis=-1
        )
        ratio = tf.exp(logp_act - b_old)
        clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * b_adv, clipped * b_adv))
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
    if max_grad_norm is not None:
        a_grads, _ = tf.clip_by_global_norm(a_grads, max_grad_norm)
        c_grads, _ = tf.clip_by_global_norm(c_grads, max_grad_norm)
    actor_opt.apply_gradients(zip(a_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(c_grads, critic.trainable_variables))
    approx_kl = tf.reduce_mean(b_old - logp_act)
    clipfrac = tf.reduce_mean(
        tf.cast(tf.abs(ratio - 1.0) > clip_ratio, tf.float32)
    )
    return policy_loss, value_loss, entropy, t_kl, approx_kl, clipfrac


def ppo_update(
    actor: keras.Model,
    critic: keras.Model,
    traj: Trajectory,
    actor_opt: keras.optimizers.Optimizer,
    critic_opt: keras.optimizers.Optimizer,
    *,
    num_actions: int,
    clip_ratio: float,
    c1: float,
    c2: float,
    epochs: int,
    batch_size: int,
    teacher: Optional[keras.Model] = None,
    kl_coef: float,
    kl_decay: float,
    max_grad_norm: Optional[float] = None,
    target_kl: Optional[float] = None,
    debug: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Выполнить несколько эпох обновления PPO.
    Возвращает обновлённый коэффициент KL и словарь метрик."""

    obs = tf.convert_to_tensor(traj.obs)
    acts = tf.convert_to_tensor(traj.actions)
    adv = tf.convert_to_tensor(traj.advantages)
    ret = tf.convert_to_tensor(traj.returns)
    old_logp = tf.convert_to_tensor(traj.old_logp)
    masks = tf.convert_to_tensor(traj.masks)

    # Создаём датасет TensorFlow и перемешиваем его для SGD
    dataset = tf.data.Dataset.from_tensor_slices((obs, acts, adv, ret, old_logp, masks))
    dataset = dataset.shuffle(len(traj.actions)).batch(batch_size)
    pi_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    teacher_kls: List[float] = []
    approx_kls: List[float] = []
    clipfracs: List[float] = []

    # Запускаем несколько эпох обучения на собранном батче
    for ep in range(epochs):
        if debug:
            print(f"ppo_update: epoch {ep + 1}/{epochs}")
        for batch in dataset:
            b_obs, b_act, b_adv, b_ret, b_old, b_mask = batch
            b_adv = (b_adv - tf.reduce_mean(b_adv)) / (tf.math.reduce_std(b_adv) + 1e-8)
            (
                policy_loss,
                value_loss,
                entropy,
                t_kl,
                approx_kl,
                clipfrac,
            ) = _ppo_batch_update(
                actor,
                critic,
                actor_opt,
                critic_opt,
                b_obs,
                b_act,
                b_adv,
                b_ret,
                b_old,
                b_mask,
                num_actions=num_actions,
                clip_ratio=clip_ratio,
                c1=c1,
                c2=c2,
                teacher=teacher,
                kl_coef=kl_coef,
                max_grad_norm=max_grad_norm,
            )
            pi_losses.append(float(policy_loss))
            v_losses.append(float(value_loss))
            entropies.append(float(entropy))
            teacher_kls.append(float(t_kl))
            approx_kls.append(float(approx_kl))
            clipfracs.append(float(clipfrac))
            if debug:
                print(
                    "batch: policy_loss=%.5f value_loss=%.5f entropy=%.5f approx_kl=%.5f"
                    % (
                        float(policy_loss),
                        float(value_loss),
                        float(entropy),
                        approx_kls[-1],
                    )
                )
            if target_kl is not None and approx_kls[-1] > target_kl:
                break

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
    env: BacktestEnv,
    actor: keras.Model,
    seq_len: int,
    feature_dim: int,
    debug: bool = False,
) -> Dict[str, float]:
    """Запустить политику в среде и вернуть метрики."""

    obs = env.reset()
    state_hist = [obs["state"]]
    for _ in range(seq_len - 1):
        obs, _, done, _ = env.step(3)
        state_hist.append(obs["state"])
        if done:
            break
    # Прогоняем политику до завершения эпизода
    while True:
        t = env.t
        window = env.features[t - seq_len + 1 : t + 1]
        state_window = np.stack(state_hist[t - seq_len + 1 : t + 1])
        feat = np.concatenate([window, state_window], axis=1)
        mask = env.action_mask()
        logits = actor(feat[None, ...], training=False)
        masked = apply_action_mask(logits, mask[None, :])
        action = int(tf.argmax(masked, axis=-1)[0])
        obs, _, done, _ = env.step(action)
        state_hist.append(obs["state"])
        if debug:
            print(
                "evaluate_profit: t=%d action=%d equity=%.4f"
                % (env.t, action, env.equity)
            )
        if done:
            break
    metrics = env.metrics_report()
    if debug:
        print(f"evaluate_profit metrics={metrics}")
    return metrics


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: EnvConfig,
    feature_dim: int,
    feature_cols: List[str],
    seq_len: int,
    teacher_weights: str,
    backbone_weights: str,
    save_path: str,
    num_actions: int,
    units_per_layer: List[int],
    dropout: float,
    updates: int,
    n_env: int,
    rollout: int,
    actor_lr: float,
    critic_lr: float,
    clip_ratio: float,
    c1: float,
    c2: float,
    epochs: int,
    batch_size: int,
    teacher_kl: float,
    kl_decay: float,
    max_grad_norm: float,
    target_kl: float,
    val_interval: int,
    debug: bool = False,
) -> Tuple[keras.Model, keras.Model, List[Dict[str, float]], List[Dict[str, float]]]:
    """Основной цикл обучения PPO поверх табличных данных."""
    # создаем среду для валидации и инфереса feature_dim
    val_env = BacktestEnv(
        val_df,
        feature_cols=feature_cols,
        price_col="Open",
        cfg=cfg,
        ppo_true=True
    )

    # Создаём модели актёра и критика
    actor, critic = build_actor_critic(
        seq_len,
        feature_dim,
        num_actions=num_actions,
        units_per_layer=units_per_layer,
        dropout=dropout,
        backbone_weights=backbone_weights,
    )
    teacher_backbone = build_backbone(
        seq_len, feature_dim, units_per_layer=units_per_layer, dropout=dropout
    )
    teacher = build_head(teacher_backbone, num_actions)
    teacher.load_weights(teacher_weights)
    teacher.trainable = False
    if debug:
        print("train: debug mode enabled")

    actor_opt = keras.optimizers.Adam(actor_lr)
    critic_opt = keras.optimizers.Adam(critic_lr)
    os.makedirs(save_path, exist_ok=True)

    # Инициализируем коэффициент KL и параметры ранней остановки
    kl_coef = teacher_kl
    best_profit = -np.inf
    best_actor_path = os.path.join(save_path, "actor_best.weights.h5")
    best_critic_path = os.path.join(save_path, "critic_best.weights.h5")


    train_log: List[Dict[str, float]] = []
    val_log: List[Dict[str, float]] = []

    for step in range(updates):
        # Сбор траекторий и обновление параметров
        traj = collect_trajectories(
            train_df,
            actor,
            critic,
            cfg,
            feature_cols,
            n_env=n_env,
            rollout=rollout,
            seq_len=seq_len,
            num_actions=num_actions,
            debug=debug,
        )
        kl_coef, metrics = ppo_update(
            actor,
            critic,
            traj,
            actor_opt,
            critic_opt,
            num_actions=num_actions,
            clip_ratio=clip_ratio,
            c1=c1,
            c2=c2,
            epochs=epochs,
            batch_size=batch_size,
            teacher=teacher,
            kl_coef=kl_coef,
            kl_decay=kl_decay,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            debug=debug,
        )
        avg_ret = float(np.mean(traj.returns))
        metrics["avg_return"] = avg_ret
        train_log.append(metrics)
        if debug:
            print(f"update={step} metrics={metrics}")

        # Периодически оцениваем политику на валидационном наборе
        if (step + 1) % val_interval == 0:
            val_metrics = evaluate_profit(
                val_env, actor, seq_len, feature_dim, debug=debug
            )
            val_log.append(val_metrics)
            if debug:
                print(f"validation metrics={val_metrics}")
            profit = val_metrics.get("Equity", 0.0)
            if profit > best_profit:
                best_profit = profit
                actor.save_weights(best_actor_path)
                critic.save_weights(best_critic_path)
            print(
                f"update={step} avg_reward={avg_ret:.3f} profit={profit:.3f} kl_coef={kl_coef:.4f}"
            )

    actor.save_weights(os.path.join(save_path, "actor.weights.h5"))
    critic.save_weights(os.path.join(save_path, "critic.weights.h5"))
    actor.load_weights(best_actor_path)
    critic.load_weights(best_critic_path)

    # Финальная оценка на тестовой выборке и сохранение графика
    test_env = BacktestEnv(
        test_df,
        feature_cols=feature_cols,
        price_col="Close",
        cfg=cfg,
    )
    evaluate_profit(test_env, actor, seq_len, feature_dim, debug=debug)
    fig = test_env.plot("PPO inference")
    os.makedirs("results", exist_ok=True)
    fig.savefig(os.path.join("results", "ppo_inference.png"))
    return actor, critic, train_log, val_log


__all__ = [
    "build_actor_critic",
    "prepare_datasets",
    "collect_trajectories",
    "ppo_update",
    "train",
    "evaluate_profit",
]
