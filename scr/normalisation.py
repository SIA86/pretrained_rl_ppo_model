from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class NormalizationStats:
    """Расчёт статистик и покомпонентное нормирование признаков.

    Параметры
    ----------
    kind:
        Стратегия нормализации. Поддерживаются:
        ``'zscore'`` (среднее/стандартное отклонение), ``'minmax'`` (размах),
        ``'robust'`` (медиана/IQR).
    eps:
        Малое число для избежания деления на ноль.
    """

    kind: Literal["zscore", "minmax", "robust"] = "zscore"
    eps: float = 1e-8
    imputer_mean: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None
    q1: Optional[np.ndarray] = None
    q3: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "NormalizationStats":
        """Подобрать статистики нормализации на обучающих данных."""
        Xf = X_train.astype(np.float32, copy=True)
        Xf[~np.isfinite(Xf)] = np.nan
        self.imputer_mean = np.nanmean(Xf, axis=0)
        self.imputer_mean = np.where(
            np.isfinite(self.imputer_mean), self.imputer_mean, 0.0
        ).astype(np.float32)
        Xc = self._impute(X_train)
        if self.kind == "zscore":
            self.mean = Xc.mean(axis=0)
            self.std = Xc.std(axis=0) + self.eps
        elif self.kind == "minmax":
            self.min_ = Xc.min(axis=0)
            self.max_ = Xc.max(axis=0)
        elif self.kind == "robust":
            self.q1 = np.percentile(Xc, 25, axis=0)
            self.q3 = np.percentile(Xc, 75, axis=0)
            self.median = np.median(Xc, axis=0)
            self.std = (self.q3 - self.q1) + self.eps
        else:
            raise ValueError("Unknown norm kind")
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Заменить нечисловые значения средними по столбцам."""
        X = X.astype(np.float32, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            col = np.nonzero(bad)[1]
            X[bad] = self.imputer_mean[col]
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применить нормализацию к массиву."""
        X = self._impute(X)
        if self.kind == "zscore":
            return (X - self.mean) / self.std
        if self.kind == "minmax":
            return (X - self.min_) / ((self.max_ - self.min_) + self.eps)
        if self.kind == "robust":
            return (X - self.median) / self.std
        raise ValueError("Unknown norm kind")
