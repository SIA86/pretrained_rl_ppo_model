from __future__ import annotations

# Стандартные библиотечные и внешние зависимости
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, NamedTuple, Any
from numba import njit, boolean, int64, float64
import matplotlib.pyplot as plt
from .normalisation import NormalizationStats
import tensorflow as tf

try:
    from tqdm import trange
except Exception:  # pragma: no cover
    trange = range


# =============================================================
# Конфигурация среды
# =============================================================
class EnvConfig(NamedTuple):
    """Настройки торговой среды."""

    # Режим торговли: +1 только лонг, -1 только шорт
    mode: int
    # Комиссия как доля от номинала сделки
    fee: float
    # Половина спреда как доля от цены (спред симметричный)
    spread: float
    # Используемое кредитное плечо
    leverage: float
    # Максимальное количество шагов в эпизоде
    max_steps: int
    # Множитель вознаграждения за шаг
    reward_scale: float
    # Использовать ли логарифмическую доходность
    use_log_reward: bool
    # Штраф за каждый шаг удержания позиции
    time_penalty: float
    # Штраф за бездействие без открытой позиции
    hold_penalty: float


DEFAULT_CONFIG = EnvConfig(
    mode=1,  # работаем только от длинной позиции
    fee=0.0,  # без комиссии
    spread=0.0,  # без спреда
    leverage=1.0,  # без плеча
    max_steps=10**9,  # практически бесконечный эпизод
    reward_scale=1.0,  # без масштабирования вознаграждения
    use_log_reward=False,  # линейная доходность
    time_penalty=0.0,  # нет штрафа за удержание
    hold_penalty=0.0,  # нет штрафа за бездействие
)


# =============================================================
# Numba helpers
# =============================================================
@njit(cache=False, fastmath=False)
def _exec_price(next_price: float64, side: int64, spread: float64) -> float64:
    """Расчёт цены исполнения с учётом спреда."""
    return next_price * (1.0 + spread * side)


@njit(cache=False, fastmath=False)
def _fee_notional(price_exec: float64, leverage: float64, fee: float64) -> float64:
    """Расчёт комиссии, уплачиваемой за сделку."""
    # Комиссия пропорциональна номиналу позиции: цена * плечо * ставка
    return price_exec * leverage * fee


@njit(cache=False, fastmath=False)
def _step_single(
    action: int64,
    t: int64,
    position: int64,
    entry_price: float64,
    realized_pnl: float64,
    prices: np.ndarray,
    cfg: EnvConfig,
) -> tuple:
    """Векторизованное выполнение одного шага симуляции.

    Параметр ``action`` принимает значения:
    0=Open, 1=Close, 2=Hold, 3=Wait.

    Возвращает кортеж со всеми обновлёнными состояниями и вспомогательной
    информацией, которая используется для логирования и расчёта метрик.

    Порядок элементов в кортеже:
    new_t, position, entry_price, equity, realized_pnl, reward,
    opened, closed, exec_price, pnl_trade, done
    """

    next_t = t + 1  # следующий индекс времени
    done = False
    # Проверяем ограничение по максимальному числу шагов
    if cfg.max_steps is not None and next_t >= cfg.max_steps:
        done = True

    # Текущая и следующая цены
    this_price = prices[t]
    next_price = prices[next_t]

    prev_position = position  # позиция до совершения действия
    prev_realized = realized_pnl
    prev_unrealized = 0.0
    if prev_position != 0:
        prev_unrealized = (
            prev_position * ((this_price - entry_price) / entry_price) * cfg.leverage
        )

    # Флаги и накопители, используемые ниже
    opened = False
    closed = False
    exec_price = 0.0
    pnl_trade = 0.0

    allowed_side = cfg.mode  # допустимое направление торговли

    if action == 0:
        # Открыть позицию
        if position == 0:
            exec_price = _exec_price(next_price, allowed_side, cfg.spread)
            entry_price = exec_price
            position = allowed_side
            opened = True
            fee = _fee_notional(exec_price, cfg.leverage, cfg.fee)
            realized_pnl -= fee
    elif action == 1:
        # Закрыть имеющуюся позицию
        if position != 0:
            exec_price = _exec_price(next_price, -position, cfg.spread)
            pnl_trade = (
                position * ((exec_price - entry_price) / entry_price) * cfg.leverage
            )
            fee = _fee_notional(exec_price, cfg.leverage, cfg.fee)
            realized_pnl += pnl_trade - fee
            position = 0
            entry_price = 0.0
            closed = True
    elif action == 3:
        # Оставаться вне позиции (Wait)
        if cfg.hold_penalty > 0.0:
            realized_pnl -= cfg.hold_penalty
    # action == 2 -> Hold: ничего не делаем

    if prev_position != 0 and cfg.time_penalty > 0.0:
        realized_pnl -= cfg.time_penalty

    # Нереализованный PnL после совершения действия
    unrealized = 0.0
    if position != 0:
        unrealized = (
            position * ((next_price - entry_price) / entry_price) * cfg.leverage
        )

    equity = realized_pnl + unrealized
    prev_equity = prev_realized + prev_unrealized

    # Возможность использовать логарифмическую доходность
    pnl_step = equity - prev_equity
    # Лог-вознаграждение: защищаемся от домена log1p (x > -1)
    if cfg.use_log_reward:
        # Жёстко клипуем шаговую доходность снизу чуть выше -1
        clipped = pnl_step if pnl_step > -0.999999 else -0.999999
        core = np.log1p(clipped)
    else:
        core = pnl_step

    reward = cfg.reward_scale * core

    return (
        next_t,
        position,
        entry_price,
        equity,
        realized_pnl,
        reward,
        opened,
        closed,
        exec_price,
        pnl_trade,
        done,
    )


# =============================================================
# Environment class
# =============================================================
class BacktestEnv:
    """Простая среда для тестирования однонаправленных стратегий."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        price_col: str = "close",
        cfg: EnvConfig = DEFAULT_CONFIG,
        state_stats: Optional[NormalizationStats] = None,
    ):
        """Подготовка данных и настройка параметров среды.

        Parameters
        ----------
        df : pd.DataFrame
            Таблица с ценами и индикаторами.
        feature_cols : list of str, optional
            Колонки, используемые в качестве наблюдений. Если None, берутся
            все числовые признаки кроме цены.
        price_col : str
            Имя колонки с ценой, по которой рассчитывается PnL.
        cfg : EnvConfig
            Объект конфигурации среды.
        state_stats : NormalizationStats, optional
            Статистика для нормализации вектора состояния портфеля.
        """

        # Сбрасываем индекс, чтобы шаги шли от 0
        self.df = df.reset_index(drop=True)
        if feature_cols is None:
            # По умолчанию используем все числовые признаки, кроме цены
            feature_cols = [
                c
                for c in df.columns
                if c != price_col and np.issubdtype(df[c].dtype, np.number)
            ]
        # Массив признаков и цен для быстрого доступа
        self.features = self.df[feature_cols].to_numpy(dtype=np.float32)
        self.prices = self.df[price_col].to_numpy(dtype=np.float64)
        self.highs = (
            self.df["High"].to_numpy(dtype=np.float64)
            if "High" in self.df.columns
            else self.prices
        )
        self.lows = (
            self.df["Low"].to_numpy(dtype=np.float64)
            if "Low" in self.df.columns
            else self.prices
        )
        self.state_stats = state_stats
        # Ограничиваем количество шагов размером датасета
        max_steps = min(cfg.max_steps, len(self.prices) - 1)
        # Создаём копию конфигурации с поправленным max_steps
        self.cfg = EnvConfig(
            cfg.mode,
            cfg.fee,
            cfg.spread,
            cfg.leverage,
            max_steps,
            cfg.reward_scale,
            cfg.use_log_reward,
            cfg.time_penalty,
            cfg.hold_penalty,
        )
        # Инициализация состояния
        self.reset()

    def reset(self):
        """Сброс среды к начальному состоянию."""
        self.t = 0  # текущий шаг
        self.position = 0  # открытая позиция: 0 нет, ±1 направление
        self.entry_price = 0.0  # цена входа в позицию
        self.equity = 0.0  # накопленная доходность
        self.realized_pnl = 0.0  # реализованный PnL
        self.unrealized_pnl = 0.0  # нереализованный PnL
        self.flat_steps = 1  # количество шагов вне позиции
        self.hold_steps = 0  # количество шагов в позиции
        self.drawdown = 0.0  # текущая просадка
        self.worst_price = self.prices[0]
        self.done = False  # флаг завершения эпизода
        self.history: List[Dict] = []  # журнал событий
        return self._get_obs()

    def action_mask(self) -> np.ndarray:
        """Маска допустимых действий.

        Порядок действий: 0=Open, 1=Close, 2=Hold, 3=Wait."""
        if self.position == 0:
            return np.array([1, 0, 0, 1], dtype=np.int8)
        return np.array([0, 1, 1, 0], dtype=np.int8)

    def step(self, action) -> tuple:
        # Запрещаем шаги после завершения эпизода
        if getattr(self, "done", False):
            # Возвращаем текущее наблюдение и нулевую награду
            obs = self._get_obs()
            info = {
                "equity": self.equity,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "position": self.position,
                "flat_steps": self.flat_steps,
                "hold_steps": self.hold_steps,
                "drawdown": self.drawdown,
                "action_mask": self.action_mask(),
            }
            return obs, 0.0, True, info

        mask = self.action_mask()
        # Поддержка как целочисленного действия, так и вектора логитов/оценок длиной 4
        if isinstance(action, (list, tuple, np.ndarray)):
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            if a.size == 4:
                masked = np.where(mask.astype(bool), a, -np.inf)
                if np.all(~np.isfinite(masked)):
                    action = 3 if self.position == 0 else 2
                else:
                    action = int(np.nanargmax(masked))
            else:
                action = 3 if self.position == 0 else 2
        # Валидация для скалярного действия
        if (
            not isinstance(action, (int, np.integer))
            or action < 0
            or action >= len(mask)
            or not mask[action]
        ):
            action = 3 if self.position == 0 else 2

        (
            self.t,
            self.position,
            self.entry_price,
            self.equity,
            self.realized_pnl,
            reward,
            opened,
            closed,
            exec_price,
            pnl_trade,
            done,
        ) = _step_single(
            int64(action),
            int64(self.t),
            int64(self.position),
            float64(self.entry_price),
            float64(self.realized_pnl),
            self.prices,
            self.cfg,
        )
        # Обновляем флаг завершения
        if self.t >= self.cfg.max_steps:
            self.done = True

        price = self.prices[self.t]
        high = self.highs[self.t]
        low = self.lows[self.t]

        if self.position == 0:
            self.flat_steps += 1
            self.hold_steps = 0
            self.unrealized_pnl = 0.0
            self.drawdown = 0.0
            self.worst_price = price
        else:
            if opened:
                self.hold_steps = 1
                self.flat_steps = 0
                self.worst_price = low if self.position > 0 else high
            else:
                self.hold_steps += 1
                self.flat_steps = 0
                if self.position > 0:
                    self.worst_price = min(self.worst_price, low)
                else:
                    self.worst_price = max(self.worst_price, high)
            if self.position > 0:
                self.unrealized_pnl = (
                    (price - self.entry_price) / self.entry_price * self.cfg.leverage
                )
                self.drawdown = self.worst_price / self.entry_price - 1.0
            else:
                self.unrealized_pnl = (
                    (self.entry_price - price) / self.entry_price * self.cfg.leverage
                )
                self.drawdown = self.entry_price / self.worst_price - 1.0

        self.history.append(
            {
                "t": self.t,
                "price": price,
                "position": self.position,
                "entry_price": self.entry_price,
                "reward": reward,
                "equity": self.equity,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "flat_steps": self.flat_steps,
                "hold_steps": self.hold_steps,
                "drawdown": self.drawdown,
                "opened": opened,
                "closed": closed,
                "exec_price": exec_price,
                "pnl_trade": pnl_trade,
            }
        )

        obs = self._get_obs()
        info = {
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position,
            "flat_steps": self.flat_steps,
            "hold_steps": self.hold_steps,
            "drawdown": self.drawdown,
            "action_mask": mask,
        }
        return obs, reward, done, info

    def _get_state(self) -> np.ndarray:
        state = np.array(
            [
                float(self.position),
                float(self.unrealized_pnl),
                float(self.flat_steps) / 1000.0,
                float(self.hold_steps) / 1000.0,
                float(self.drawdown),
            ],
            dtype=np.float32,
        )
        if self.state_stats is not None:
            state = self.state_stats.transform(state[None, :])[0]
        return state

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"features": self.features[self.t], "state": self._get_state()}

    # ---------------------------------------------------------
    # Вспомогательные методы
    # ---------------------------------------------------------
    def logs(self) -> pd.DataFrame:
        """Преобразовать журнал событий в DataFrame."""
        return pd.DataFrame(self.history)

    def save_logs(self, path: str):
        """Сохранить журнал событий на диск."""
        self.logs().to_csv(path, index=False)

    def plot(self, title: Optional[str] = None):
        """Построить набор диагностических графиков по результатам симуляции."""

        # Получаем журнал событий в виде DataFrame
        log = self.logs()

        # Создаём четыре подграфика: цена, индикатор позиции, реализованный PnL и equity
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        if title:
            fig.suptitle(title)

        # -----------------------------------------------------
        # 1) График цены с пометками точек входа/выхода
        # -----------------------------------------------------
        ax[0].plot(log["t"], log["price"], label="price")
        ax[0].set_ylabel("Price")

        # Находим моменты открытия и закрытия позиции
        opens = log[log["opened"]]
        closes = log[log["closed"]]

        # В зависимости от режима торговли выбираем направление стрелок
        open_marker = "^" if self.cfg.mode == 1 else "v"
        close_marker = "v" if self.cfg.mode == 1 else "^"

        if not opens.empty:
            # Рисуем зелёные стрелки входа
            ax[0].scatter(
                opens["t"],
                opens["exec_price"],
                marker=open_marker,
                color="green",
                label="entry",
            )
        if not closes.empty:
            # Рисуем красные стрелки выхода
            ax[0].scatter(
                closes["t"],
                closes["exec_price"],
                marker=close_marker,
                color="red",
                label="exit",
            )

        # -----------------------------------------------------
        # 2) График показывающий, находимся ли мы в позиции
        # -----------------------------------------------------
        in_position = (log["position"] != 0).astype(int)
        ax[1].step(log["t"], in_position, where="post", label="in_position")
        ax[1].set_ylabel("Pos")

        # -----------------------------------------------------
        # 3) Реализованный PnL
        # -----------------------------------------------------
        ax[2].plot(log["t"], log["realized_pnl"], label="realized pnl")
        ax[2].set_ylabel("Realized")

        # -----------------------------------------------------
        # 4) Накопленная доходность (equity)
        # -----------------------------------------------------
        ax[3].plot(log["t"], log["equity"], label="equity")
        ax[3].set_ylabel("Equity")
        ax[3].set_xlabel("Step")

        # Отображаем легенду на каждом подграфике
        for a in ax:
            a.legend()

        plt.tight_layout()
        return fig

    def metrics_report(self) -> str:
        """Сформировать отчёт по ключевым метрикам торговой истории.

        Возвращает многострочную строку со следующими показателями:
        equity, реализованный PnL, количество закрытых сделок, win rate,
        средний PnL на сделку, profit factor, максимальная просадка и
        коэффициент Шарпа.
        """
        if not self.history:
            return "История пуста, метрики недоступны."

        log = self.logs()
        equity = log["equity"].iloc[-1]
        realized_pnl = log["realized_pnl"].iloc[-1]

        trades = log[log["closed"]]
        n_trades = len(trades)
        win_rate = (trades["pnl_trade"] > 0).mean() if n_trades else 0.0
        avg_pnl = trades["pnl_trade"].mean() if n_trades else 0.0

        gross_profit = trades[trades["pnl_trade"] > 0]["pnl_trade"].sum()
        gross_loss = trades[trades["pnl_trade"] < 0]["pnl_trade"].sum()
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = np.inf if gross_profit > 0 else 0.0

        equity_curve = log["equity"]
        run_max = equity_curve.cummax()
        drawdown = (run_max - equity_curve) / run_max.replace(0, np.nan)
        max_drawdown = drawdown.max()
        if pd.isna(max_drawdown):
            max_drawdown = 0.0

        returns = log["reward"]
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))
        else:
            sharpe = 0.0

        metrics = [
            ("Equity", equity),
            ("Realized PnL", realized_pnl),
            ("Closed trades", n_trades),
            ("Win rate", win_rate * 100),
            ("Avg PnL per trade", avg_pnl),
            ("Profit factor", profit_factor),
            ("Max drawdown", max_drawdown),
            ("Sharpe ratio", sharpe),
        ]

        lines = []
        for name, value in metrics:
            if name == "Win rate":
                lines.append(f"{name}: {value:.2f}%")
            elif isinstance(value, float):
                lines.append(f"{name}: {value:.4f}")
            else:
                lines.append(f"{name}: {value}")
        return "\n".join(lines)


def run_backtest_with_logits(
    df: pd.DataFrame,
    model,
    feature_stats: Optional[NormalizationStats],
    seq_len: int,
    start: Optional[int] = None,
    feature_cols: Optional[List[str]] = None,
    price_col: str = "close",
    cfg: EnvConfig = DEFAULT_CONFIG,
    state_stats: Optional[NormalizationStats] = None,
    show_progress: bool = True,
) -> BacktestEnv:
    """Run backtest by querying ``model`` on every step.

    Parameters
    ----------
    df:
        Исходный DataFrame с ценами и признаками.
    model:
        Модель, возвращающая логиты действий при вызове
        ``model(features, training=False)``.
    feature_stats:
        Статистика нормализации признаков. Если ``None``, признаки не
        нормализуются.
    seq_len:
        Длина окна признаков, подаваемого в модель.
    start:
        Индекс строки ``df``, с которой начинается тест. По умолчанию
        ``seq_len - 1``.
    feature_cols:
        Список колонок признаков. Если ``None``, будут использованы все
        числовые признаки кроме ``price_col``.
    price_col:
        Имя колонки с ценой.
    cfg:
        Конфигурация среды ``BacktestEnv``.
    state_stats:
        Статистика нормализации состояния аккаунта, передаваемая в среду.
    show_progress:
        Показывать ли индикатор прогресса бэктеста.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if start is None:
        start = seq_len - 1
    if start < seq_len - 1:
        raise ValueError("start must be >= seq_len - 1")
    if start >= len(df) - 1:
        raise ValueError("start index out of range")

    df_slice = df.iloc[start - seq_len + 1 :].copy()
    env = BacktestEnv(
        df_slice,
        feature_cols=feature_cols,
        price_col=price_col,
        cfg=cfg,
        state_stats=state_stats,
    )
    env.reset()
    state_hist = [env._get_state()]
    for _ in range(seq_len - 1):
        env.step(3)
        state_hist.append(env._get_state())

    if isinstance(model, tf.Module):

        @tf.function
        def _predict(x0):
            return model(x0, training=False)

        def predict(inputs):
            return _predict(inputs)

    else:

        def predict(inputs):
            return model(inputs, training=False)

    total_steps = len(env.prices) - seq_len
    iterator = trange(total_steps) if show_progress else range(total_steps)
    for _ in iterator:
        if env.done or env.t >= len(env.prices) - 1:
            break
        t = env.t
        window = env.features[t - seq_len + 1 : t + 1]
        if feature_stats is not None:
            window = feature_stats.transform(window)
        state_window = np.stack(state_hist[t - seq_len + 1 : t + 1])
        inputs = np.concatenate([window, state_window], axis=1)
        logits = predict(inputs[None, :, :])
        env.step(np.asarray(logits).reshape(-1))
        state_hist.append(env._get_state())

    return env
