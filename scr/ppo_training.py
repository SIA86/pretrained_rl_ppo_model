"""Цикл обучения PPO поверх предобученной модели residual LSTM.

Модуль связывает предварительно обученную политику из ``residual_lstm.py`` с
средой ``BacktestEnv``. Дополнительно используется KL‑регуляризация по
teacher‑модели с затухающим коэффициентом и ранняя остановка по прибыли, что
позволяет дообучить супервизорную модель методом обучения с подкреплением.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import weakref

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib.backends.backend_pdf import PdfPages

from .backtest_env import BacktestEnv, EnvConfig
from .dataset_builder import extract_features
from .normalisation import NormalizationStats
from .residual_lstm import (
    apply_action_mask,
    build_backbone,
    build_head,
    masked_logits_and_probs,
)


def build_critic(
    seq_len: int,
    feature_dim: int,
    *,
    num_actions: int,
    units: int,
    dropout: float,
    backbone_weights: str | None = None,
    freeze_backbone: bool = False,
) -> keras.Model:
    """Создать критика и при необходимости заморозить его бэкбон."""
    # Создаём бэкбон критика
    critic_backbone = build_backbone(seq_len, feature_dim, units=units, dropout=dropout)
    # При необходимости загружаем предобученные веса бэкбона
    if backbone_weights:
        print('Loading critic backbone weights')
        critic_backbone.load_weights(backbone_weights)
    if freeze_backbone:
        print('Critic backbone training off')
        critic_backbone.trainable = False
    # На выход бэкбона навешиваются головы актёра и критика
    critic = build_head(critic_backbone, 1)
    return critic


@dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    old_logp: np.ndarray
    masks: np.ndarray


@dataclass
class EvalCacheEntry:
    env: Optional[BacktestEnv] = None
    metrics: Dict[str, float] = field(default_factory=dict)


IntervalLike = pd.Index | np.ndarray | Iterable[pd.Timestamp]

_ACTOR_LOGITS_CACHE: "weakref.WeakKeyDictionary[keras.Model, Dict[Tuple[int, int, str], tf.types.experimental.ConcreteFunction]]" = weakref.WeakKeyDictionary()


def _get_actor_logits_fn(
    actor: keras.Model, seq_len: int, feature_dim: int
) -> Tuple[tf.types.experimental.ConcreteFunction, tf.dtypes.DType]:
    """Return cached tf.function for actor inference."""

    cache = _ACTOR_LOGITS_CACHE.setdefault(actor, {})
    actor_dtype = getattr(actor, "compute_dtype", None) or getattr(actor, "dtype", None)
    dtype = tf.as_dtype(actor_dtype or tf.float32)
    key = (int(seq_len), int(feature_dim), dtype.name)
    fn = cache.get(key)
    if fn is None:
        input_spec = tf.TensorSpec(shape=(None, seq_len, feature_dim), dtype=dtype)

        @tf.function(input_signature=(input_spec,), reduce_retracing=True)
        def _actor_logits(batch_features: tf.Tensor) -> tf.Tensor:
            return actor(batch_features, training=False)

        cache[key] = _actor_logits
        fn = _actor_logits
    return fn, dtype


def _normalize_interval(
    index: pd.DatetimeIndex, interval: IntervalLike
) -> pd.DatetimeIndex:
    """Привести интервал к DatetimeIndex исходного индекса данных."""
    seg = pd.DatetimeIndex(interval)
    if index.tz is not None:
        if seg.tz is None:
            seg = seg.tz_localize(index.tz)
        elif str(seg.tz) != str(index.tz):
            seg = seg.tz_convert(index.tz)
    elif seg.tz is not None:
        seg = seg.tz_convert("UTC").tz_localize(None)
    return seg.intersection(index)


def _collect_valid_starts(
    index: pd.DatetimeIndex,
    intervals: Sequence[IntervalLike],
    required: int,
) -> np.ndarray:
    """Собрать допустимые стартовые позиции для окон заданной длины."""
    if required <= 0:
        raise ValueError("required must be positive")
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be DatetimeIndex when using intervals")

    starts: List[np.ndarray] = []
    for interval in intervals:
        seg = _normalize_interval(index, interval)
        if len(seg) == 0:
            continue
        seg = seg.unique().sort_values()
        positions = index.get_indexer(seg)
        if len(positions) == 0 or np.any(positions < 0):
            continue
        positions.sort()
        splits = np.where(np.diff(positions) != 1)[0] + 1
        for run in np.split(positions, splits):
            if len(run) < required:
                continue
            limit = len(run) - required + 1
            starts.append(run[:limit])
    if not starts:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(starts))


def _prepare_validation_windows(
    val_df: pd.DataFrame,
    intervals: Sequence[IntervalLike],
    required: int,
) -> List[pd.DataFrame]:
    """Сформировать кандидаты для валидационных окон из произвольных интервалов."""
    if required <= 0:
        raise ValueError("required must be positive")
    if not isinstance(val_df.index, pd.DatetimeIndex):
        raise TypeError("val_df must have DatetimeIndex when using index_ranges")

    windows: List[pd.DataFrame] = []
    for interval in intervals:
        seg = _normalize_interval(val_df.index, interval)
        if len(seg) == 0:
            continue
        seg = seg.unique().sort_values()
        window = val_df.loc[seg]
        if len(window) == 0:
            continue
        if len(window) > required:
            window = window.iloc[:required]
        windows.append(window)
    return windows


def _format_metric_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def prepare_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: Tuple[float, float, float] = None,
    test_from: str = None,
    norm_kind: str = "zscore",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    NormalizationStats,
]:
    """Разбить ``df`` на Train/Val/Test и нормализовать признаки."""
    if splits is not None:
        assert abs(sum(splits) - 1.0) < 1e-8
        n = len(df)
        s1 = int(n * splits[0])
        s2 = int(n * (splits[0] + splits[1]))
        train_df = df[feature_cols].iloc[:s1].copy()
        val_df = df[feature_cols].iloc[s1:s2].copy()
        test_df = df[feature_cols].iloc[s2:].copy()
    elif test_from is not None:
        if isinstance(test_from, str) and test_from:
            train_df = df[feature_cols].loc[:test_from].copy()
            val_df = None
            test_df = df[feature_cols].loc[test_from:].copy()
    else:
        raise ValueError('No "splits" or "test_from" params given')

    # Считаем статистики нормализации по тренировочной части
    feat_stats = NormalizationStats(kind=norm_kind).fit(
        train_df[feature_cols].to_numpy(dtype=np.float32)
    )
    # Применяем нормализацию к каждому из подмножеств
    for part in (train_df, val_df, test_df):
        if part is not None:
            part[feature_cols] = feat_stats.transform(
                part[feature_cols].to_numpy(dtype=np.float32)
            )

    return train_df, val_df, test_df, feat_stats


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
    index_ranges: Optional[Sequence[IntervalLike]] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    debug: bool = False,
) -> Trajectory:
    """Собрать батч траекторий из ``n_env`` сред с пересэмплингом окон."""

    L = cfg.max_steps + 1
    needed = max(cfg.max_steps + seq_len, rollout)
    if debug:
        print(
            f"\ncollect_trajectories: n_env={n_env} rollout={rollout} seq_len={seq_len}"
        )
    if index_ranges:
        starts = _collect_valid_starts(train_df.index, index_ranges, needed)
        if starts.size == 0:
            raise ValueError("No valid start indices in index_ranges")
    else:
        max_start = len(train_df) - needed
        if max_start < 0:
            raise ValueError("DataFrame too short for given parameters")
        starts = np.arange(max_start + 1)

    if debug:
        print(f"Valid Indexes length :{len(starts)}")

    envs: List[BacktestEnv] = []
    state_dim: Optional[int] = None
    feature_dim_total: Optional[int] = None
    for _ in range(n_env):
        s = int(np.random.choice(starts))
        window_df = train_df.iloc[s : s + L].reset_index(drop=True)
        env = BacktestEnv(
            window_df,
            feature_cols=feature_cols,
            price_col="Open",
            cfg=cfg,
            ppo_true=True,
            record_history=False,
        )
        obs = env.reset()
        if state_dim is None:
            state_dim = int(obs["state"].shape[0])
        if feature_dim_total is None:
            feature_dim_total = int(env.features.shape[1])
        for _ in range(seq_len - 1):
            obs, _, done, _ = env.step(3)
            if done:
                break
        envs.append(env)

    if state_dim is None or feature_dim_total is None:
        raise RuntimeError("Failed to initialise PPO environments")

    state_buf = np.zeros((n_env, seq_len, state_dim), dtype=np.float32)
    feat_buf = np.zeros(
        (n_env, seq_len, feature_dim_total + state_dim), dtype=np.float32
    )
    mask_batch = np.zeros((n_env, num_actions), dtype=np.float32)

    obs_buf = [[] for _ in range(n_env)]
    act_buf = [[] for _ in range(n_env)]
    rew_buf = [[] for _ in range(n_env)]
    val_buf = [[] for _ in range(n_env)]
    next_val_buf = [[] for _ in range(n_env)]
    logp_buf = [[] for _ in range(n_env)]
    mask_buf = [[] for _ in range(n_env)]
    done_buf = [[] for _ in range(n_env)]

    @tf.function
    def _infer(obs_batch: tf.Tensor, mask_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        logits = actor(obs_batch, training=False)
        values = critic(obs_batch, training=False)[:, 0]
        _, probs = masked_logits_and_probs(logits, mask_batch)
        return probs, values

    @tf.function
    def _critic(obs_batch: tf.Tensor) -> tf.Tensor:
        values = critic(obs_batch, training=False)[:, 0]
        return values

    def _reset_env(idx: int):
        s = int(np.random.choice(starts))
        window_df = train_df.iloc[s : s + L].reset_index(drop=True)
        env = BacktestEnv(
            window_df,
            feature_cols=feature_cols,
            price_col="Open",
            cfg=cfg,
            ppo_true=True,
            record_history=False,
        )
        obs = env.reset()
        states = [obs["state"].copy()]
        for _ in range(seq_len - 1):
            obs, _, done, _ = env.step(3)
            states.append(obs["state"].copy())
            if done:
                break
        envs[idx] = env
        state_buf[idx].fill(0.0)
        if states:
            state_buf[idx, -len(states) :] = states

    for i in range(n_env):
        _reset_env(i)

    executor = ThreadPoolExecutor(max_workers=min(n_env, os.cpu_count() or 1))
    try:
        for _ in range(rollout):
            for i, env in enumerate(envs):
                if env.done:
                    _reset_env(i)
                t = env.t
                window = env.features[t - seq_len + 1 : t + 1]
                feat_buf[i, :, :feature_dim_total] = window
                feat_buf[i, :, feature_dim_total:] = state_buf[i]
                mask_batch[i] = env.action_mask().astype(np.float32)

            probs_tf, values_tf = _infer(
                tf.convert_to_tensor(feat_buf), tf.convert_to_tensor(mask_batch)
            )
            probs_np = probs_tf.numpy()
            values_np = values_tf.numpy()

            actions: List[int] = []
            logps: List[float] = []
            for i in range(n_env):
                probs = probs_np[i]
                mask = mask_batch[i]
                if not np.isfinite(probs).all() or probs.sum() <= 0.0:
                    valid_actions = np.flatnonzero(mask)
                    probs = np.zeros(num_actions, dtype=np.float32)
                    probs[valid_actions] = 1.0 / len(valid_actions)
                action = int(np.random.choice(num_actions, p=probs))
                actions.append(action)
                logps.append(float(np.log(probs[action] + 1e-8)))

            futures = [
                executor.submit(env.step, int(action))
                for env, action in zip(envs, actions)
            ]

            next_indices: List[int] = []
            next_feats: List[np.ndarray] = []
            for i, fut in enumerate(futures):
                next_obs, reward, done, _ = fut.result()
                obs_buf[i].append(feat_buf[i].copy())
                act_buf[i].append(actions[i])
                rew_buf[i].append(reward)
                val_buf[i].append(float(values_np[i]))
                logp_buf[i].append(logps[i])
                mask_buf[i].append(mask_batch[i].copy())
                done_buf[i].append(done)

                if done:
                    next_val_buf[i].append(0.0)
                    _reset_env(i)
                else:
                    state_buf[i] = np.roll(state_buf[i], -1, axis=0)
                    state_buf[i, -1] = next_obs["state"]
                    t_next = envs[i].t
                    window_next = envs[i].features[t_next - seq_len + 1 : t_next + 1]
                    feat_next = np.zeros_like(feat_buf[i])
                    feat_next[:, :feature_dim_total] = window_next
                    feat_next[:, feature_dim_total:] = state_buf[i]
                    next_indices.append(i)
                    next_feats.append(feat_next)

            if next_indices:
                next_vals = _critic(tf.convert_to_tensor(next_feats)).numpy()
                for idx, value in zip(next_indices, next_vals):
                    next_val_buf[idx].append(float(value))
    finally:
        executor.shutdown(wait=True)

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
    entropy_decay: float,
    max_grad_norm: Optional[float] = None,
    target_kl: Optional[float] = None,
    debug: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Выполнить несколько эпох обновления PPO.
    Возвращает обновлённые коэффициенты KL и энтропии вместе со словарём метрик."""

    obs = tf.convert_to_tensor(traj.obs)
    acts = tf.convert_to_tensor(traj.actions)
    adv = tf.convert_to_tensor(traj.advantages)
    ret = tf.convert_to_tensor(traj.returns)
    old_logp = tf.convert_to_tensor(traj.old_logp)
    masks = tf.convert_to_tensor(traj.masks)

    if not actor_opt.built:
        actor_opt.build(actor.trainable_variables)
    if not critic_opt.built:
        critic_opt.build(critic.trainable_variables)

    # Создаём датасет TensorFlow и перемешиваем его для SGD
    dataset = tf.data.Dataset.from_tensor_slices((obs, acts, adv, ret, old_logp, masks))
    dataset = dataset.shuffle(len(traj.actions)).batch(batch_size)
    pi_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    teacher_kls: List[float] = []
    approx_kls: List[float] = []
    clipfracs: List[float] = []

    if debug:
        print("\nUpdating PPO:")

    # Запускаем несколько эпох обучения на собранном батче
    for ep in range(epochs):
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
                    f"Epoch {ep+1}/{epochs}: batch: policy_loss={policy_loss:.5f} value_loss={value_loss:.5f} entropy={entropy:.5f} approx_kl={approx_kls[-1]:.5f}"
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
    return kl_coef * kl_decay, c2 * entropy_decay, metrics


def evaluate_profit(
    env: BacktestEnv,
    actor: keras.Model,
    seq_len: int,
    feature_dim: int,
    debug: bool = False,
) -> Dict[str, float]:
    """Запустить политику в среде и вернуть метрики."""

    logits_fn, input_dtype = _get_actor_logits_fn(actor, seq_len, feature_dim)
    obs = env.reset()
    state_hist = [obs["state"].copy()]
    for _ in range(seq_len - 1):
        obs, _, done, _ = env.step(3)
        state_hist.append(obs["state"].copy())
        if done:
            break
    # Прогоняем политику до завершения эпизода
    while True:
        t = env.t
        window = env.features[t - seq_len + 1 : t + 1]
        state_window = np.stack(state_hist[t - seq_len + 1 : t + 1])
        feat = np.concatenate([window, state_window], axis=1)
        mask = env.action_mask()
        feat_batch = tf.convert_to_tensor(
            feat[None, ...], dtype=input_dtype, name="eval_features"
        )
        logits = logits_fn(feat_batch)
        masked = apply_action_mask(logits, mask[None, :])
        action = int(tf.argmax(masked, axis=-1)[0])
        obs, _, done, _ = env.step(action)
        state_hist.append(obs["state"].copy())
        if done:
            break
    metrics = env.metrics_report()

    return metrics


def _evaluate_window(
    window_df: pd.DataFrame,
    *,
    model: keras.Model,
    feature_cols: List[str],
    cfg: EnvConfig,
    seq_len: int,
    feature_dim: int,
    price_col: str,
    keep_env: bool,
    debug: bool = False,
) -> EvalCacheEntry:
    """Оценить модель на одном окне валидации."""

    env = BacktestEnv(
        window_df,
        feature_cols=feature_cols,
        cfg=cfg,
        price_col=price_col,
        ppo_true=True,
        record_history=True,
    )
    metrics = evaluate_profit(env, model, seq_len, feature_dim, debug=debug)
    return EvalCacheEntry(env=env if keep_env else None, metrics=metrics)


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: EnvConfig,
    val_cfg: EnvConfig,
    feature_dim: int,
    feature_cols: List[str],
    seq_len: int,
    teacher_weights: str,
    critic_weights: str,
    backbone_weights: str,
    save_path: str,
    num_actions: int,
    units: int,
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
    entropy_decay: float,
    max_grad_norm: float,
    target_kl: float,
    val_interval: int,
    n_validations: int = 1,
    gamma: float = 0.99,
    lam: float = 0.95,
    index_ranges: Optional[Sequence[IntervalLike]] = None,
    log_to_pdf_path: bool = True,
    fine_tune: bool = False,
    debug: bool = False,
) -> Tuple[keras.Model, keras.Model, List[Dict[str, float]], List[Dict[str, float]]]:
    """Основной цикл обучения PPO поверх табличных данных."""
    if debug:
        print(f"CPU available: {os.cpu_count()}")

    # создаем среду для валидации и инфереса feature_dim
    val_required = val_cfg.max_steps + 1
    val_candidates: List[pd.DataFrame] | None = None
    if index_ranges:
        val_candidates = _prepare_validation_windows(val_df, index_ranges, val_required)
        if not val_candidates:
            raise ValueError("No valid validation intervals in index_ranges")
    else:
        max_start = len(val_df) - val_required
        if max_start < 0:
            raise ValueError("DataFrame too short for given parameters")
        val_starts = np.arange(max_start + 1)

    # Создаём модели актёра и критика
    critic = build_critic(
        seq_len,
        feature_dim,
        num_actions=num_actions,
        units=units,
        dropout=dropout,
        backbone_weights=backbone_weights,
        freeze_backbone=fine_tune,
    )
    if critic_weights:
        print('Loading full critic model weights')
        critic.load_weights(critic_weights)

    teacher_backbone = build_backbone(seq_len, feature_dim, units=units, dropout=dropout)
    actor_backbone = build_backbone(seq_len, feature_dim, units=units, dropout=dropout)
    if fine_tune:
        print('Actor backbone training off')
        actor_backbone.trainable = False
    teacher = build_head(teacher_backbone, num_actions)
    actor = build_head(actor_backbone, num_actions)
    if teacher_weights:
        print('Loading full actor model weights')
        teacher.load_weights(teacher_weights)
        actor.load_weights(teacher_weights)
    teacher.trainable = False

    actor_opt = keras.optimizers.Adam(actor_lr)
    critic_opt = keras.optimizers.Adam(critic_lr)
    os.makedirs(save_path, exist_ok=True)

    # Инициализируем коэффициент KL и параметры ранней остановки
    kl_coef = teacher_kl
    entropy_coef = c2
    best_profit: Optional[float] = None

    train_log: List[Dict[str, float]] = []
    val_log: List[Dict[str, float]] = []

    if val_candidates is not None:
        val_windows_all = [window.copy() for window in val_candidates]
    else:
        val_windows_all = [
            val_df.iloc[int(start) : int(start) + val_required].copy()
            for start in val_starts
        ]

    if not val_windows_all:
        raise ValueError("No validation windows available")

    total_windows = len(val_windows_all)
    display_count = min(n_validations, total_windows)
    if display_count > 0:
        display_indices = np.random.choice(
            total_windows, size=display_count, replace=False
        )
        display_indices = np.sort(display_indices.astype(int))
    else:
        display_indices = np.empty(0, dtype=int)
    display_set = {int(idx) for idx in display_indices.tolist()}

    baseline_name = "Teacher"
    baseline_cache: List[EvalCacheEntry] = []

    if log_to_pdf_path:
        if isinstance(log_to_pdf_path, (str, os.PathLike)):
            path_to_log_pdf = os.fspath(log_to_pdf_path)
        else:
            path_to_log_pdf = os.path.join(save_path, "validation")
        os.makedirs(path_to_log_pdf, exist_ok=True)

    for step in range(updates):
        # Сбор траекторий и обновление параметров
        traj = collect_trajectories(
            train_df,
            actor,
            critic,
            cfg,
            feature_cols,
            index_ranges=index_ranges,
            n_env=n_env,
            rollout=rollout,
            seq_len=seq_len,
            num_actions=num_actions,
            gamma=gamma,
            lam=lam,
            debug=debug,
        )
        current_c2 = entropy_coef
        kl_coef, entropy_coef, metrics = ppo_update(
            actor,
            critic,
            traj,
            actor_opt,
            critic_opt,
            num_actions=num_actions,
            clip_ratio=clip_ratio,
            c1=c1,
            c2=current_c2,
            epochs=epochs,
            batch_size=batch_size,
            teacher=teacher,
            kl_coef=kl_coef,
            kl_decay=kl_decay,
            entropy_decay=entropy_decay,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            debug=debug,
        )
        avg_ret = float(np.mean(traj.returns))
        metrics["avg_return"] = avg_ret
        metrics["entropy_coef"] = current_c2
        train_log.append(metrics)
        if debug:
            print(f"\nUpdate={step} \nTraining average metrics:")
            print(f"KL_teacher: {kl_coef:.4f}")
            for k,v in metrics.items():
                print(f"{k}: {v:.5f}")

        # Периодически оцениваем политику на валидационном наборе
        if (step + 1) % val_interval == 0:
            current_baseline_name = baseline_name
            keep_env_flags = [
                (idx in display_set) or bool(log_to_pdf_path)
                for idx in range(total_windows)
            ]
            max_workers = os.cpu_count() or 1

            def _run_validation(model: keras.Model) -> List[EvalCacheEntry]:
                results: List[EvalCacheEntry] = [None] * total_windows
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _evaluate_window,
                            val_windows_all[idx],
                            model=model,
                            feature_cols=feature_cols,
                            cfg=val_cfg,
                            seq_len=seq_len,
                            feature_dim=feature_dim,
                            price_col="Open",
                            keep_env=keep_env_flags[idx],
                            debug=debug,
                        ): idx
                        for idx in range(total_windows)
                    }
                    for future, idx in future_to_idx.items():
                        results[idx] = future.result()
                return results

            baseline_entries = _run_validation(teacher)
            actor_entries = _run_validation(actor)

            baseline_total_pnl = float(
                sum(entry.metrics.get("Realized PnL", 0.0) for entry in baseline_entries)
            )
            actor_total_pnl = float(
                sum(entry.metrics.get("Realized PnL", 0.0) for entry in actor_entries)
            )

            if best_profit is None:
                best_profit = baseline_total_pnl

            if log_to_pdf_path:
                pdf_file = os.path.join(
                    path_to_log_pdf, f"validation_step_{step+1:04d}.pdf"
                )
                with PdfPages(pdf_file) as pdf:
                    for idx in range(total_windows):
                        base_entry = baseline_entries[idx]
                        actor_entry = actor_entries[idx]
                        if base_entry.env is not None:
                            fig = base_entry.env.plot(
                                f"PPO {current_baseline_name.lower()} window {idx}",
                                show=False,
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                        if actor_entry.env is not None:
                            fig = actor_entry.env.plot(
                                f"PPO validation window {idx}", show=False
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                        text_fig = plt.figure(figsize=(8.27, 11.69))
                        text_fig.suptitle(f"Window {idx} metrics", fontsize=14)
                        ax = text_fig.add_subplot(111)
                        ax.axis("off")
                        lines = [f"Baseline ({current_baseline_name}) vs Current"]
                        metric_keys = sorted(
                            set(base_entry.metrics) | set(actor_entry.metrics)
                        )
                        for key in metric_keys:
                            base_val = _format_metric_value(
                                base_entry.metrics.get(key)
                            )
                            model_val = _format_metric_value(
                                actor_entry.metrics.get(key)
                            )
                            lines.append(f"{key}: {base_val} / {model_val}")
                        ax.text(
                            0.01,
                            0.99,
                            "\n".join(lines),
                            va="top",
                            ha="left",
                            family="monospace",
                        )
                        pdf.savefig(text_fig)
                        plt.close(text_fig)

            if log_to_pdf_path:
                for idx, entry in enumerate(baseline_entries):
                    if idx not in display_set:
                        entry.env = None
                for idx, entry in enumerate(actor_entries):
                    if idx not in display_set:
                        entry.env = None

            for raw_idx in display_indices:
                idx = int(raw_idx)
                baseline_entry = baseline_entries[idx]
                actor_entry = actor_entries[idx]
                if baseline_entry.env is not None:
                    fig = baseline_entry.env.plot(
                        f"PPO {current_baseline_name.lower()} window {idx}",
                        show=True,
                    )
                    plt.close(fig)
                if actor_entry.env is not None:
                    fig = actor_entry.env.plot(
                        f"PPO validation window {idx}",
                        show=True,
                    )
                    plt.close(fig)
                print("\nMetrics (Baseline / New):")
                metric_keys = sorted(
                    set(baseline_entry.metrics) | set(actor_entry.metrics)
                )
                for key in metric_keys:
                    base_val = baseline_entry.metrics.get(key)
                    model_val = actor_entry.metrics.get(key)
                    print(
                        f"{key}: {_format_metric_value(base_val)} / {_format_metric_value(model_val)}"
                    )

            for entry in actor_entries:
                val_log.append(entry.metrics)

            print(
                f"\nValidation summary ({current_baseline_name} vs Current):"
            )
            print(
                f"Total Realized PnL: {baseline_total_pnl:.4f} / {actor_total_pnl:.4f}"
            )

            best_actor_path = os.path.join(
                save_path,
                f"actor_best_ep:{step+1}_pnl:{actor_total_pnl:.4f}.weights.h5",
            )
            best_critic_path = os.path.join(
                save_path,
                f"critic_best_ep:{step+1}_pnl:{actor_total_pnl:.4f}.weights.h5",
            )

            if actor_total_pnl > baseline_total_pnl:
                print(
                    "\nНайден новый чемпион. Суммарный профит на валидации: "
                    f"{actor_total_pnl:.5f}"
                )
                best_profit = actor_total_pnl
                actor.save_weights(best_actor_path)
                critic.save_weights(best_critic_path)
                if teacher is not None:
                    teacher.set_weights(actor.get_weights())
                    teacher.trainable = False
                    kl_coef = teacher_kl
                    entropy_coef = c2
                baseline_name = "Champion"
                baseline_cache = actor_entries
            else:
                baseline_cache = baseline_entries
                if best_profit is not None:
                    best_profit = max(best_profit, baseline_total_pnl)


    actor.save_weights(os.path.join(save_path, "actor_final.weights.h5"))
    critic.save_weights(os.path.join(save_path, "critic_final.weights.h5"))

    return actor, critic, train_log, val_log

def testing_simulation(
    test_df,
    actor_weights,
    seq_len,
    feature_dim,
    units,
    dropout,
    num_actions,
    feature_cols,
    test_cfg,
    debug=False
):
    actor_backbone = build_backbone(seq_len, feature_dim, units=units, dropout=dropout)
    actor = build_head(actor_backbone, num_actions)
    actor.load_weights(actor_weights)

    test_env = BacktestEnv(
        test_df,
        feature_cols=feature_cols,
        price_col="Open",
        cfg=test_cfg,
        ppo_true=True,
        record_history=True,
    )
    evaluate_profit(test_env, actor, seq_len, feature_dim, debug=debug)
    fig = test_env.plot("PPO testing")
    metrics = test_env.metrics_report()
    print("\nMetrics Report")
    for k,v in metrics.items():
      print(f"{k}: {v}")


__all__ = [
    "prepare_datasets",
    "collect_trajectories",
    "ppo_update",
    "train",
    "evaluate_profit",
    "testing_simulation"
]
