"""Simple market regime detector based on recent volatility.

Institutions often switch strategies depending on market mode;
we mimic that with a tiny, dependency-free component.  The thresholds
are configurable via CONFIG so you can tweak behaviour without
re-deploying code.
"""
from financial_organism.config import CONFIG
import numpy as np


class RegimeDetector:
    LOW_VOL = "LOW_VOL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"

    def __init__(self, window: int = None):
        # how many past returns to consider when computing volatility
        self.window = window if window is not None else int(CONFIG.get("REGIME_WINDOW", 20))
        self.low_vol_threshold = float(CONFIG.get("LOW_VOL_THRESHOLD", 0.10))
        self.high_vol_threshold = float(CONFIG.get("HIGH_VOL_THRESHOLD", 0.30))

    def detect(self, market_data: dict) -> str:
        """Return one of LOW_VOL / HIGH_VOL / CRISIS based on recent returns.

        Args:
            market_data (dict): expected to contain 'returns' list
        """
        returns = market_data.get("returns", []) or []
        if not returns:
            return self.UNKNOWN
        # compute volatility over the configured window
        arr = np.array(returns[-self.window :])
        vol = float(np.std(arr))
        if vol < self.low_vol_threshold:
            return self.LOW_VOL
        elif vol < self.high_vol_threshold:
            return self.HIGH_VOL
        else:
            return self.CRISIS
