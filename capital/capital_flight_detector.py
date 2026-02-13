import statistics

class CapitalFlightDetector:
    def __init__(self, lookback=20, zscore_threshold=2.2, consecutive_negatives=4):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.consecutive_negatives = consecutive_negatives

    def detect(self, market_data):
        returns = [float(x) for x in market_data.get("returns", [])]
        if len(returns) < max(5, self.lookback):
            return False
        w = returns[-self.lookback:]
        mu, sigma = statistics.fmean(w), statistics.pstdev(w)
        latest = w[-1]
        z = 0.0 if sigma == 0 else (latest - mu) / sigma
        tail = w[-self.consecutive_negatives:]
        sustained = all(x < 0 for x in tail)
        return z < -self.zscore_threshold or sustained
