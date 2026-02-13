import statistics

class AICrisisPredictor:
    def __init__(self, lookback=10, crisis_return_threshold=-0.05):
        self.lookback = lookback
        self.crisis_return_threshold = crisis_return_threshold
        self._bias = 0.0

    def train(self, returns, epochs=1):
        vals = [float(x) for x in returns]
        if not vals:
            self._bias = 0.0
            return
        hits = [1 for v in vals if v < self.crisis_return_threshold]
        self._bias = max(0.0, min(1.0, len(hits) / len(vals)))

    def predict(self, recent_returns):
        vals = [float(x) for x in recent_returns]
        if len(vals) < self.lookback:
            return 0.0
        sample = vals[-self.lookback:]
        mu = statistics.fmean(sample)
        sigma = statistics.pstdev(sample)
        recent_stress = max(0.0, min(1.0, abs(min(mu, 0.0)) / abs(self.crisis_return_threshold)))
        vol_stress = max(0.0, min(1.0, sigma / 0.03))
        return float(max(0.0, min(1.0, 0.5*recent_stress + 0.3*vol_stress + 0.2*self._bias)))
