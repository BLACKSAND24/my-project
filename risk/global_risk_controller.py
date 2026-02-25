import numpy as np
from collections import deque
import time


class GlobalRiskController:
    """Tracks portfolio-wide exposures, correlations and drawdown.

    This sits above individual strategies and gives a bird's-eye view
    of how allocations move over time.  It is intentionally lightweight
    so that it can be used even on a single-VPS setup.
    """

    def __init__(self, history_len: int = 100):
        # keep last N allocation snapshots (ordered dicts symbol->notional)
        self.history_len = history_len
        self.exposure_history = deque(maxlen=history_len)
        self.strategy_names = set()
        # equity curve for drawdown tracking
        self.equity_curve = []
        # last update timestamp (for daily/weekly rollover elsewhere)
        self.last_update_ts = time.time()

    def update_exposure(self, allocations: dict):
        """Record a new set of allocations.

        Args:
            allocations (dict): symbol -> notional currently committed.
        """
        if allocations is None:
            allocations = {}
        self.exposure_history.append(allocations.copy())
        self.strategy_names.update(allocations.keys())

    def record_equity(self, equity: float):
        """Append current portfolio equity for drawdown monitoring."""
        self.equity_curve.append(equity)
        if len(self.equity_curve) > self.history_len:
            self.equity_curve = self.equity_curve[-self.history_len:]

    def total_exposure(self) -> float:
        """Return sum of absolute exposures in the latest snapshot."""
        if not self.exposure_history:
            return 0.0
        latest = self.exposure_history[-1]
        return sum(abs(v) for v in latest.values())

    def correlation_matrix(self) -> dict:
        """Compute correlation matrix between strategies based on the
        past allocation history.

        Returns:
            dict: {"names": [...], "matrix": numpy.ndarray}
        """
        if not self.exposure_history or len(self.exposure_history) < 2:
            return {"names": [], "matrix": np.array([[]])}
        names = sorted(self.strategy_names)
        # build an array where each row corresponds to a strategy, each
        # column to a historical snapshot
        data = []
        for strategy in names:
            row = [snap.get(strategy, 0.0) for snap in self.exposure_history]
            data.append(row)
        arr = np.array(data)
        corr = np.corrcoef(arr) if arr.size > 0 else np.array([[]])
        return {"names": names, "matrix": corr}

    def max_drawdown(self) -> float:
        """Return maximum drawdown seen on the equity curve."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        maxdd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > maxdd:
                    maxdd = dd
        return maxdd
