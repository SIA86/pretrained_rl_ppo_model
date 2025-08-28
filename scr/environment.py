# =========================
# Векторная версия (батч PPO)
# =========================

# trading_env_gym.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, NamedTuple
from numba import njit, prange, boolean, int64, float64


# =========================
# Конфиг (Numba-friendly)
# =========================
class EnvConfig(NamedTuple):
    mode: int64                # +1 long-only, -1 short-only
    fee: float64               # комиссия от notional (например, 0.0005)
    spread: float64            # половина спрэда (доля цены)
    leverage: float64          # плечо
    max_steps: int64           # максимум шагов в эпизоде
    reward_scale: float64      # множитель награды
    use_log_reward: boolean    # лог-доходность
    time_penalty: float64      # штраф за удержание позиции (за шаг)
    hold_penalty: float64      # штраф за пустой шаг во flat

# =========================
# Награда (отдельной функцией, njit)
# =========================
@njit(cache=True, fastmath=True)
def compute_reward(
    position: int64,
    ret_step: float64,          # ((p_{t+1}-p_t)/p_t) * leverage
    opened: boolean, closed: boolean,
    fees_paid: float64,         # комиссии/штрафы, начисленные на шаге
    time_penalty: float64,
    reward_scale: float64,
    use_log_reward: boolean
) -> float64:
    pnl_step = position * ret_step
    core = np.log1p(pnl_step) if use_log_reward else pnl_step
    r = core - fees_paid - time_penalty * (position != 0)
    return reward_scale * r

# =========================
# Исполнение и метрики (njit)
# =========================
@njit(cache=True, fastmath=True)
def _exec_price(next_price: float64, side: int64, spread: float64) -> float64:
    # side: +1 покупка (ask выше), -1 продажа (bid ниже)
    return next_price * (1.0 + spread * side)

@njit(cache=True, fastmath=True)
def _fee_notional(price_exec: float64, leverage: float64, fee: float64) -> float64:
    # простая модель: комиссия как доля notional с учётом плеча
    return fee * leverage

@njit(cache=True, fastmath=True, parallel=True)
def step_batch(
    actions: np.ndarray,       # (N,) int {0,1,2}
    t_idx: np.ndarray,         # (N,)
    start_idx: np.ndarray,     # (N,)
    end_idx: np.ndarray,       # (N,)
    position: np.ndarray,      # (N,) {0, mode}
    entry_price: np.ndarray,   # (N,)
    equity: np.ndarray,        # (N,)
    peak_equity: np.ndarray,   # (N,)
    realized_pnl: np.ndarray,  # (N,)
    open_steps: np.ndarray,    # (N,)
    prices: np.ndarray,        # (T,)
    cfg: EnvConfig,
    trade_pnl_sum: np.ndarray, # (N,)
    trade_cnt: np.ndarray,     # (N,)
    win_cnt: np.ndarray,       # (N,)
    max_dd: np.ndarray         # (N,)
) -> Tuple[np.ndarray, np.ndarray]:
    N = actions.shape[0]
    rewards = np.zeros(N, dtype=np.float64)
    dones = np.zeros(N, dtype=np.bool_)

    for i in prange(N):
        if dones[i]:
            continue

        t = t_idx[i]
        if t >= end_idx[i]:
            dones[i] = True
            continue

        a = actions[i]
        pos = position[i]  # {0, cfg.mode}
        this_price = prices[t]
        next_price_raw = prices[t + 1]

        ret = ((next_price_raw - this_price) / this_price) * cfg.leverage

        opened = False
        closed = False
        fees_paid = 0.0
        allowed_side = cfg.mode  # +1 long-only / -1 short-only

        if a == 1:
            # OPEN/HOLD
            if pos == 0:
                px = _exec_price(next_price_raw, allowed_side, cfg.spread)
                entry_price[i] = px
                position[i] = allowed_side
                pos = allowed_side
                opened = True
                fees_paid += _fee_notional(px, cfg.leverage, cfg.fee)
                open_steps[i] = 0
            # иначе просто держим
        elif a == 2:
            # CLOSE
            if pos != 0:
                px = _exec_price(next_price_raw, -pos, cfg.spread)
                pnl = pos * ((px - entry_price[i]) / entry_price[i]) * cfg.leverage
                realized_pnl[i] += pnl
                trade_pnl_sum[i] += pnl
                trade_cnt[i] += 1
                if pnl > 0.0:
                    win_cnt[i] += 1
                position[i] = 0
                entry_price[i] = 0.0
                open_steps[i] = 0
                closed = True
                fees_paid += _fee_notional(px, cfg.leverage, cfg.fee)
        else:
            # HOLD (включая flat)
            if pos == 0 and cfg.hold_penalty > 0.0:
                fees_paid += cfg.hold_penalty

        # пошаговая награда: приращение богатства + издержки
        rewards[i] = compute_reward(
            position[i], ret, opened, closed, fees_paid, cfg.time_penalty,
            cfg.reward_scale, cfg.use_log_reward
        )

        # обновление equity и просадки
        equity[i] += rewards[i]
        if equity[i] > peak_equity[i]:
            peak_equity[i] = equity[i]
        if peak_equity[i] > 1e-16:
            dd = (peak_equity[i] - equity[i]) / (peak_equity[i] + 1e-16)
            if dd > max_dd[i]:
                max_dd[i] = dd

        if position[i] != 0:
            open_steps[i] += 1

        t_idx[i] = t + 1

        # условия окончания эпизода
        if t_idx[i] >= end_idx[i] or (t_idx[i] - start_idx[i]) >= cfg.max_steps:
            # форс-закрытие, если есть позиция
            if position[i] != 0:
                px = _exec_price(prices[t_idx[i]], -position[i], cfg.spread)
                pnl = position[i] * ((px - entry_price[i]) / entry_price[i]) * cfg.leverage
                realized_pnl[i] += pnl
                trade_pnl_sum[i] += pnl
                trade_cnt[i] += 1
                if pnl > 0.0:
                    win_cnt[i] += 1
                position[i] = 0
                entry_price[i] = 0.0
                open_steps[i] = 0
            dones[i] = True

    return rewards, dones

# =========================
# Подготовка данных как в SL
# =========================
def prepare_dataset_like_sl(
    raw_df,
    sl_preprocess_fn,
    feature_cols: Optional[list] = None,
    price_col: str = "close",
    as_float32: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    df = sl_preprocess_fn(raw_df)  # та же функция, что использовалась в SL
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c != price_col and np.issubdtype(df[c].dtype, np.number)]
    features = df[feature_cols].to_numpy(copy=True)
    prices = df[price_col].to_numpy(copy=True).astype(np.float64)
    if as_float32:
        features = features.astype(np.float32, copy=False)
    return features, prices


class VecTradingEnv:
    """
    Минималистичная векторная среда (не зависит от gym VectorEnv).
    Интерфейс:
      reset() -> obs (N,F), step(actions[N]) -> (obs, rewards[N], dones[N], info_dict_of_arrays)
    Все тяжёлые операции — в njit(step_batch).
    """
    def __init__(
        self,
        features: np.ndarray, prices: np.ndarray, cfg: EnvConfig,
        n_envs: int = 16, min_window: int = 128, seed: int = 42
    ):
        assert features.shape[0] == prices.shape[0]
        assert cfg.mode in (+1, -1)
        self.features, self.prices, self.cfg = features, prices, cfg
        self.n_envs = n_envs
        self.min_window = min_window
        self._rng = np.random.RandomState(seed)
        T = features.shape[0]
        self._valid_start = np.arange(0, T - (min_window + 1), dtype=np.int64)

        # батч-состояния
        N = n_envs
        self.t_idx      = np.zeros(N, dtype=np.int64)
        self.start_idx  = np.zeros(N, dtype=np.int64)
        self.end_idx    = np.zeros(N, dtype=np.int64)
        self.position   = np.zeros(N, dtype=np.int64)
        self.entry_price= np.zeros(N, dtype=np.float64)
        self.equity     = np.zeros(N, dtype=np.float64)
        self.peak_equity= np.zeros(N, dtype=np.float64)
        self.realized_pnl = np.zeros(N, dtype=np.float64)
        self.open_steps = np.zeros(N, dtype=np.int64)

        self.trade_pnl_sum = np.zeros(N, dtype=np.float64)
        self.trade_cnt     = np.zeros(N, dtype=np.int64)
        self.win_cnt       = np.zeros(N, dtype=np.int64)
        self.max_dd        = np.zeros(N, dtype=np.float64)

    def _sample_windows(self):
        T = self.features.shape[0]
        for i in range(self.n_envs):
            s = int(self._rng.choice(self._valid_start))
            e = s + int(self.cfg.max_steps) + 1
            if e >= T:
                e = T - 1
            if (e - s) < self.min_window:
                e = s + self.min_window
            self.start_idx[i] = s
            self.end_idx[i] = e
            self.t_idx[i] = s

    def reset(self) -> np.ndarray:
        self._sample_windows()
        self.position[:] = 0
        self.entry_price[:] = 0.0
        self.equity[:] = 0.0
        self.peak_equity[:] = 0.0
        self.realized_pnl[:] = 0.0
        self.open_steps[:] = 0
        self.trade_pnl_sum[:] = 0.0
        self.trade_cnt[:] = 0
        self.win_cnt[:] = 0
        self.max_dd[:] = 0.0
        return self.features[self.t_idx]

    def step(self, actions: np.ndarray):
        actions = actions.astype(np.int64, copy=False)
        rewards, dones = step_batch(
            actions, self.t_idx, self.start_idx, self.end_idx,
            self.position, self.entry_price,
            self.equity, self.peak_equity, self.realized_pnl,
            self.open_steps, self.prices, self.cfg,
            self.trade_pnl_sum, self.trade_cnt, self.win_cnt, self.max_dd
        )
        obs = self.features[self.t_idx]
        info = {
            "position": self.position.copy(),
            "equity": self.equity.copy(),
            "realized_pnl": self.realized_pnl.copy(),
            "trade_cnt": self.trade_cnt.copy(),
            "win_cnt": self.win_cnt.copy(),
            "max_dd": self.max_dd.copy(),
            "start_idx": self.start_idx.copy(),
            "end_idx": self.end_idx.copy(),
            "t_idx": self.t_idx.copy(),
        }
        return obs, rewards, dones, info