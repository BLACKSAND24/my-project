import statistics
from strategies.strategy_base import Strategy

class MomentumStrategy(Strategy):
    def __init__(self): super().__init__("momentum")
    def score(self, market_data):
        r = [float(x) for x in market_data.get("returns",[])]
        return max(0.0, statistics.fmean(r[-5:])*100) if len(r) >= 5 else 0.0

class MeanReversionStrategy(Strategy):
    def __init__(self): super().__init__("mean_revert")
    def score(self, market_data):
        r = [float(x) for x in market_data.get("returns",[])]
        if len(r) < 10: return 0.0
        return max(0.0, (statistics.fmean(r[-10:]) - r[-1])*100)

class VolatilityBreakoutStrategy(Strategy):
    def __init__(self): super().__init__("vol_breakout")
    def score(self, market_data):
        r = [float(x) for x in market_data.get("returns",[])]
        return max(0.0, statistics.pstdev(r[-20:])*10) if len(r) >= 20 else 0.0

class StrategyEnsemble:
    def __init__(self):
        self.strategies = [MomentumStrategy(), MeanReversionStrategy(), VolatilityBreakoutStrategy()]

    def generate_signals(self, market_data, crisis_signals):
        raw = {s.name: s.score(market_data) for s in self.strategies}
        total = sum(raw.values())
        norm = ({k: 0.0 for k in raw} if total <= 0 else {k: v/total for k,v in raw.items()})
        damp = max(0.15, 1.0 - float(crisis_signals.get("severity", 0.0)))
        return {k: v*damp for k,v in norm.items()}

    def performance_metrics(self):
        return {"ensemble_health": 1.0, "active_strategies": len(self.strategies)}
