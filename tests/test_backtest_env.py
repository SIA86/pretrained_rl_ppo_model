import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scr.backtest_env import BacktestEnv, EnvConfig


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
    last = run_actions(env, [1, 2])
    assert last["equity"] == pytest.approx(last["realized_pnl"])


def test_equity_realized_with_fee_long():
    env = make_env([1, 2, 3], fee=0.1)
    last = run_actions(env, [1, 2])
    assert last["equity"] == pytest.approx(0.0)
    assert last["realized_pnl"] == pytest.approx(0.0)


def test_equity_realized_with_fee_short():
    env = make_env([3, 2, 1], mode=-1, fee=0.1)
    last = run_actions(env, [1, 2])
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
    run_actions(env, [1, 2, 1, 2])
    metrics = parse_report(env.metrics_report())
    assert float(metrics["Win rate"].rstrip("%")) == pytest.approx(50.0)
    assert float(metrics["Profit factor"]) == pytest.approx(2 / 3, rel=1e-3)
    assert float(metrics["Max drawdown"]) == pytest.approx(1.5)


def test_time_penalty():
    env = make_env([1, 1, 1], time_penalty=0.1)
    env.reset()
    env.step(1)  # open
    last = run_actions(env, [3], reset=False)  # hold one step
    assert last["equity"] == pytest.approx(-0.1)


def test_hold_penalty():
    env = make_env([1, 1], hold_penalty=0.05)
    last = run_actions(env, [0])
    assert last["equity"] == pytest.approx(-0.05)


def test_use_log_reward():
    env_lin = make_env([1, 2, 3])
    last_lin = run_actions(env_lin, [1, 2])
    env_log = make_env([1, 2, 3], use_log_reward=True)
    last_log = run_actions(env_log, [1, 2])
    assert last_lin["equity"] == pytest.approx(0.5)
    assert last_log["equity"] == pytest.approx(0.5)
    assert last_lin["reward"] == pytest.approx(0.5)
    assert last_log["reward"] == pytest.approx(np.log1p(0.5))

