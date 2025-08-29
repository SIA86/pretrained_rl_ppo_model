import os
import sys
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scr.visualisation import plot_enriched_actions_one_side


def test_indicators_panels_accepts_series(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Pos": [0, 0, 0, 0],
            "Q_Open": [0, 0, 0, 0],
            "Q_Close": [0, 0, 0, 0],
            "Q_Hold": [0, 0, 0, 0],
            "Q_Wait": [0, 0, 0, 0],
            "ADX_14": [10, 20, 30, 40],
        }
    )
    plot_enriched_actions_one_side(
        df, indicators_panels={"ADX": df["ADX_14"]}, start=0, end=len(df)
    )

