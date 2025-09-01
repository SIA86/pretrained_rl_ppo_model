import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr.backtest_env import BacktestEnv, EnvConfig
from scr.backtest_env import run_backtest_with_logits
from scr.normalisation import NormalizationStats


def make_env(prices, **cfg_kwargs):
    cfg = EnvConfig(
        mode=cfg_kwargs.get("mode", 1),
        fee=cfg_kwargs.get("fee", 0.0),
        spread=0.0,
        leverage=1.0,
        max_steps=100,
        reward_scale=1.0,
        use_log_reward=cfg_kwargs.get("use_log_reward", False),
        time_penalty=cfg_kwargs.get("time_penalty", 0.0),
        hold_penalty=cfg_kwargs.get("hold_penalty", 0.0),
    )
    df = pd.DataFrame({"close": prices})
    return BacktestEnv(df, cfg=cfg)


def run_actions(env, actions, reset=True):
    if reset:
        env.reset()
    for a in actions:
        env.step(a)
    return env.logs().iloc[-1]


def test_equity_realized_no_fee():
    env = make_env([1, 2, 3])
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(last["realized_pnl"])


def test_equity_realized_with_fee_long():
    env = make_env([1, 2, 3], fee=0.1)
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(0.0)
    assert last["realized_pnl"] == pytest.approx(0.0)


def test_equity_realized_with_fee_short():
    env = make_env([3, 2, 1], mode=-1, fee=0.1)
    last = run_actions(env, [0, 1])
    assert last["equity"] == pytest.approx(0.2)
    assert last["realized_pnl"] == pytest.approx(0.2)


def parse_report(report: str) -> dict:
    result = {}
    for line in report.splitlines():
        name, value = line.split(": ")
        result[name] = value
    return result


def test_metrics_report():
    env = make_env([1, 2, 3, 4, 1])
    run_actions(env, [0, 1, 0, 1])
    metrics = parse_report(env.metrics_report())
    assert float(metrics["Win rate"].rstrip("%")) == pytest.approx(50.0)
    assert float(metrics["Profit factor"]) == pytest.approx(2 / 3, rel=1e-3)
    assert float(metrics["Max drawdown"]) == pytest.approx(1.5)


def test_time_penalty():
    env = make_env([1, 1, 1], time_penalty=0.1)
    env.reset()
    env.step(0)  # open
    last = run_actions(env, [2], reset=False)  # hold one step
    assert last["equity"] == pytest.approx(-0.1)


def test_hold_penalty():
    env = make_env([1, 1], hold_penalty=0.05)
    last = run_actions(env, [3])
    assert last["equity"] == pytest.approx(-0.05)


def test_use_log_reward():
    env_lin = make_env([1, 2, 3])
    last_lin = run_actions(env_lin, [0, 1])
    env_log = make_env([1, 2, 3], use_log_reward=True)
    last_log = run_actions(env_log, [0, 1])
    assert last_lin["equity"] == pytest.approx(0.5)
    assert last_log["equity"] == pytest.approx(0.5)
    assert last_lin["reward"] == pytest.approx(0.5)
    assert last_log["reward"] == pytest.approx(np.log1p(0.5))


# Конструктор тестовых данных
def make_df(n=6, start=100.0, step=1.0):
    prices = np.array([start + i * step for i in range(n)], dtype=float)
    return pd.DataFrame({"close": prices, "feat": np.arange(n)})


def test_no_index_error_after_done():
    df = make_df(5)  # индексы 0..4
    cfg = EnvConfig(
        mode=1,
        fee=0.0,
        spread=0.0,
        leverage=1.0,
        max_steps=3,
        reward_scale=1.0,
        use_log_reward=False,
        time_penalty=0.0,
        hold_penalty=0.0,
    )
    env = BacktestEnv(df, feature_cols=["feat"], cfg=cfg)

    # Совершаем ровно max_steps шагов
    done = False
    for _ in range(cfg.max_steps):
        _, _, done, _ = env.step(3)  # Wait
    assert done is True

    # Доп. шаг после done не должен падать и должен оставаться done=True
    try:
        _, r, d, _ = env.step(3)
    except IndexError:
        pytest.fail("IndexError after done")
    assert d is True
    assert r == 0.0


def test_log_reward_no_nan_on_large_negative():
    # Имитация сильного минуса на шаге (резкий гэп вниз)
    df = pd.DataFrame({"close": [100.0, 100.0, 0.1], "feat": [0, 1, 2]})
    cfg = EnvConfig(
        mode=1,
        fee=0.0,
        spread=0.0,
        leverage=50.0,
        max_steps=10**9,
        reward_scale=1.0,
        use_log_reward=True,
        time_penalty=0.0,
        hold_penalty=0.0,
    )
    env = BacktestEnv(df, feature_cols=["feat"], cfg=cfg)

    env.step(0)  # Open long
    obs, reward, done, info = env.step(2)  # Hold → сильный минус
    assert np.isfinite(reward), "reward should be finite with log reward clipping"


def test_vector_action_with_mask_argmax():
    df = make_df(5)
    env = BacktestEnv(
        df,
        feature_cols=["feat"],
        cfg=EnvConfig(
            mode=1,
            fee=0.0,
            spread=0.0,
            leverage=1.0,
            max_steps=10**9,
            reward_scale=1.0,
            use_log_reward=False,
            time_penalty=0.0,
            hold_penalty=0.0,
        ),
    )
    # На старте позиция 0 → валидны только [Open, Wait]
    logits = [
        5.0,
        -1.0,
        9.0,
        -5.0,
    ]  # max на индексе 2, но он замаскирован → должен выбрать Open (0)
    _, _, _, info = env.step(logits)
    assert info["position"] in (0, 1)


def test_run_backtest_with_logits_executes_trade():
    df = make_df(4, start=1.0, step=1.0)
    logits = np.array([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]])
    indices = np.array([1, 2])
    env = run_backtest_with_logits(df, logits, indices)
    log = env.logs()
    assert log.iloc[-1]["equity"] == pytest.approx((4 - 3) / 3)


def test_observation_state_updates():
    df = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "feat": [0, 1, 2],
        }
    )
    env = BacktestEnv(
        df,
        feature_cols=["feat"],
        cfg=EnvConfig(
            mode=1,
            fee=0.0,
            spread=0.0,
            leverage=1.0,
            max_steps=100,
            reward_scale=1.0,
            use_log_reward=False,
            time_penalty=0.0,
            hold_penalty=0.0,
        ),
    )
    obs = env.reset()
    np.testing.assert_allclose(
        obs["state"], np.array([0, 0, 1, 0, 0], dtype=np.float32)
    )
    env.step(0)  # open
    obs, _, _, _ = env.step(2)  # hold
    state = obs["state"]
    assert state[0] == 1
    assert state[3] == 2
    assert state[2] == 0
    assert state[4] == pytest.approx(0.0)
    assert state[1] == pytest.approx((3.0 - 2.0) / 2.0)


def test_state_normalization():
    df = pd.DataFrame({"close": [1.0, 2.0], "feat": [0, 1]})
    stats = NormalizationStats()
    train_states = np.array([[0, 0, 1, 0, 0], [1, 0.5, 0, 1, -0.2]], dtype=np.float32)
    stats.fit(train_states)
    env = BacktestEnv(
        df,
        feature_cols=["feat"],
        cfg=EnvConfig(
            mode=1,
            fee=0.0,
            spread=0.0,
            leverage=1.0,
            max_steps=10**9,
            reward_scale=1.0,
            use_log_reward=False,
            time_penalty=0.0,
            hold_penalty=0.0,
        ),
        state_stats=stats,
    )
    obs = env.reset()
    expected = stats.transform(np.array([[0, 0, 1, 0, 0]], dtype=np.float32))[0]
    np.testing.assert_allclose(obs["state"], expected)
