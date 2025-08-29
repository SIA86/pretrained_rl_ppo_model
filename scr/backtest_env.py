from __future__ import annotations

# Стандартные библиотечные и внешние зависимости
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, NamedTuple
from numba import njit, boolean, int64, float64
import matplotlib.pyplot as plt

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
    mode=1,          # работаем только от длинной позиции
    fee=0.0,         # без комиссии
    spread=0.0,      # без спреда
    leverage=1.0,    # без плеча
    max_steps=10**9, # практически бесконечный эпизод
    reward_scale=1.0,# без масштабирования вознаграждения
    use_log_reward=False, # линейная доходность
    time_penalty=0.0,     # нет штрафа за удержание
    hold_penalty=0.0,     # нет штрафа за бездействие
  
# =============================================================
# Configuration dataclass
# =============================================================
class EnvConfig(NamedTuple):
    """Configuration of trading environment."""
    mode: int                   # +1 long-only, -1 short-only
    fee: float                  # commission fraction of notional
    spread: float               # half spread as fraction of price
    leverage: float             # leverage
    max_steps: int              # maximum steps in episode
    reward_scale: float         # reward multiplier
    use_log_reward: bool        # log returns
    time_penalty: float         # penalty for holding position per step
    hold_penalty: float         # penalty for doing nothing while flat


DEFAULT_CONFIG = EnvConfig(
    mode=1,
    fee=0.0,
    spread=0.0,
    leverage=1.0,
    max_steps=10**9,
    reward_scale=1.0,
    use_log_reward=False,
    time_penalty=0.0,
    hold_penalty=0.0,
)

# =============================================================
# Numba helpers
# =============================================================
@njit(cache=True, fastmath=True)
def _exec_price(next_price: float64, side: int64, spread: float64) -> float64:

    """Расчёт цены исполнения с учётом спреда."""
    return next_price * (1.0 + spread * side)


@njit(cache=True, fastmath=True)
def _fee_notional(price_exec: float64, leverage: float64, fee: float64) -> float64:
    """Расчёт комиссии, уплачиваемой за сделку."""
    # Комиссия пропорциональна номиналу позиции
    return next_price * (1.0 + spread * side)

@njit(cache=True, fastmath=True)
def _fee_notional(price_exec: float64, leverage: float64, fee: float64) -> float64:
    return fee * leverage

@njit(cache=True, fastmath=True)
def _step_single(
    action: int64,
    t: int64,
    position: int64,
    entry_price: float64,
    equity: float64,
    realized_pnl: float64,
    prices: np.ndarray,
    cfg: EnvConfig
) -> tuple:
    """Векторизованное выполнение одного шага симуляции.

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
    # Относительное изменение цены с учётом плеча
    ret = ((next_price - this_price) / this_price) * cfg.leverage

    # Флаги и накопители, используемые ниже

    """Execute one step of environment.

    Returns
    -------
    new_t : int64
    position : int64
    entry_price : float64
    equity : float64
    realized_pnl : float64
    reward : float64
    opened : boolean
    closed : boolean
    exec_price : float64
    pnl_trade : float64
    done : boolean
    """
    next_t = t + 1
    done = False
    if cfg.max_steps is not None and next_t >= cfg.max_steps:
        done = True

    this_price = prices[t]
    next_price = prices[next_t]
    ret = ((next_price - this_price) / this_price) * cfg.leverage


    opened = False
    closed = False
    exec_price = 0.0
    pnl_trade = 0.0
    fees_paid = 0.0

    allowed_side = cfg.mode  # допустимое направление торговли

    if action == 1:
        # Открыть позицию или продолжить удержание текущей
        if position == 0:
            # Входим по цене с учётом спреда

    allowed_side = cfg.mode

    if action == 1:
        # open or hold
        if position == 0:
            exec_price = _exec_price(next_price, allowed_side, cfg.spread)
            entry_price = exec_price
            position = allowed_side
            opened = True
            fees_paid += _fee_notional(exec_price, cfg.leverage, cfg.fee)
    elif action == 2:

        # Закрыть имеющуюся позицию

        # close

        if position != 0:
            exec_price = _exec_price(next_price, -position, cfg.spread)
            pnl_trade = position * ((exec_price - entry_price) / entry_price) * cfg.leverage
            realized_pnl += pnl_trade
            position = 0
            entry_price = 0.0
            closed = True
            fees_paid += _fee_notional(exec_price, cfg.leverage, cfg.fee)
    else:
        # Оставаться вне позиции
        if position == 0 and cfg.hold_penalty > 0.0:
            fees_paid += cfg.hold_penalty

    # Доход/убыток за шаг с учётом направления позиции
    pnl_step = position * ret
    # Возможность использовать логарифмическую доходность
    core = np.log1p(pnl_step) if cfg.use_log_reward else pnl_step
    # Итоговое вознаграждение с вычетом комиссий и штрафов

        # hold flat
        if position == 0 and cfg.hold_penalty > 0.0:
            fees_paid += cfg.hold_penalty

    pnl_step = position * ret
    core = np.log1p(pnl_step) if cfg.use_log_reward else pnl_step
    reward = cfg.reward_scale * (core - fees_paid - cfg.time_penalty * (position != 0))
    equity += reward

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
        done
    )

# =============================================================
# Environment class
# =============================================================
class BacktestEnv:
    """Простая среда для тестирования однонаправленных стратегий."""
    """Simple trading environment for one-sided strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Data with price and indicator columns.
    feature_cols : list of str, optional
        Columns used as observations. All numeric columns except price if None.
    price_col : str
        Column with price used for PnL calculation.
    cfg : EnvConfig
        Configuration object.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        price_col: str = "close",
        cfg: EnvConfig = DEFAULT_CONFIG,
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
        """

        # Сбрасываем индекс, чтобы шаги шли от 0
        self.df = df.reset_index(drop=True)
        if feature_cols is None:
            # По умолчанию используем все числовые признаки, кроме цены
            feature_cols = [c for c in df.columns if c != price_col and np.issubdtype(df[c].dtype, np.number)]
        # Массив признаков и цен для быстрого доступа
        self.features = self.df[feature_cols].to_numpy(dtype=np.float32)
        self.prices = self.df[price_col].to_numpy(dtype=np.float64)
        # Ограничиваем количество шагов размером датасета
        max_steps = min(cfg.max_steps, len(self.prices) - 1)
        # Создаём копию конфигурации с поправленным max_steps
        self.df = df.reset_index(drop=True)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != price_col and np.issubdtype(df[c].dtype, np.number)]
        self.features = self.df[feature_cols].to_numpy(dtype=np.float32)
        self.prices = self.df[price_col].to_numpy(dtype=np.float64)
        max_steps = min(cfg.max_steps, len(self.prices) - 1)
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
        self.history: List[Dict] = []  # журнал событий
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = 0.0
        self.realized_pnl = 0.0
        self.history: List[Dict] = []
        return self.features[self.t]

    def step(self, action: int) -> tuple:
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
            float64(self.equity),
            float64(self.realized_pnl),
            self.prices,
            self.cfg,
        )

        # Цена на текущем шаге
        price = self.prices[self.t]
        # Расчёт нереализованного PnL, если есть позиция
        unrealized = 0.0
        if self.position != 0:
            unrealized = self.position * ((price - self.entry_price) / self.entry_price) * self.cfg.leverage
        # Сохраняем все данные шага в журнал
        price = self.prices[self.t]
        unrealized = 0.0
        if self.position != 0:
            unrealized = self.position * ((price - self.entry_price) / self.entry_price) * self.cfg.leverage
        self.history.append(
            {
                "t": self.t,
                "price": price,
                "position": self.position,
                "entry_price": self.entry_price,
                "reward": reward,
                "equity": self.equity,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized,
                "opened": opened,
                "closed": closed,
                "exec_price": exec_price,
                "pnl_trade": pnl_trade,
            }
        )

        # Наблюдение и дополнительная информация для агента
        obs = self.features[self.t]
        info = {
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "position": self.position,
        }
        return obs, reward, done, info

    # ---------------------------------------------------------
    # Вспомогательные методы
    # ---------------------------------------------------------
    def logs(self) -> pd.DataFrame:
        """Преобразовать журнал событий в DataFrame."""
        return pd.DataFrame(self.history)

    def save_logs(self, path: str):
        """Сохранить журнал событий на диск."""
        self.logs().to_csv(path, index=False)

    def plot(self):
        """Построить набор диагностических графиков по результатам симуляции."""

        # Получаем журнал событий в виде DataFrame
        log = self.logs()

        # Создаём четыре подграфика: цена, индикатор позиции, реализованный PnL и equity
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

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

    # Utility methods
    # ---------------------------------------------------------
    def logs(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_logs(self, path: str):
        self.logs().to_csv(path, index=False)

    def plot(self):
        log = self.logs()
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax[0].plot(log["t"], log["price"], label="price")
        ax[0].set_ylabel("Price")
        ax[1].plot(log["t"], log["equity"], label="equity")
        ax[1].set_ylabel("Equity")
        ax[1].set_xlabel("Step")
        for a in ax:
            a.legend()
        plt.tight_layout()
        return fig
