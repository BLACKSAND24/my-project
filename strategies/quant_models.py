import statistics
from financial_organism.strategies.strategy_base import Strategy
# optional regime support
from financial_organism.macro.regime_detector import RegimeDetector

class MomentumStrategy(Strategy):
    def __init__(self): 
        super().__init__("momentum")
        self.cooldown = 0  # Strategy cooldown counter (4️⃣)
    
    def score(self, market_data):
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0.0  # Soft-mute during cooldown
        r = [float(x) for x in market_data.get("returns",[])]
        return max(0.0, statistics.fmean(r[-5:])*100) if len(r) >= 5 else 0.0

class MeanReversionStrategy(Strategy):
    def __init__(self): 
        super().__init__("mean_revert")
        self.cooldown = 0
    
    def score(self, market_data):
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0.0
        r = [float(x) for x in market_data.get("returns",[])]
        if len(r) < 10: return 0.0
        return max(0.0, (statistics.fmean(r[-10:]) - r[-1])*100)

class VolatilityBreakoutStrategy(Strategy):
    def __init__(self): 
        super().__init__("vol_breakout")
        self.cooldown = 0
    
    def score(self, market_data):
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0.0
        r = [float(x) for x in market_data.get("returns",[])]
        return max(0.0, statistics.pstdev(r[-20:])*10) if len(r) >= 20 else 0.0

class StrategyEnsemble:
    def __init__(self):
        self.strategies = [MomentumStrategy(), MeanReversionStrategy(), VolatilityBreakoutStrategy()]
        self.capital_flight_flag = False
        self.efficiency_penalties = {}  # strategy name -> penalty factor

    def penalize_low_efficiency(self, exec_efficiency_map):
        """Penalize strategies with low execution efficiency (1️⃣)."""
        for strategy_name, efficiency in (exec_efficiency_map or {}).items():
            if efficiency < 0.90:  # Less than 90% fill
                if strategy_name not in self.efficiency_penalties:
                    self.efficiency_penalties[strategy_name] = 0.0
                self.efficiency_penalties[strategy_name] = min(0.5, self.efficiency_penalties[strategy_name] + 0.1)

    def apply_strategy_cooldown(self, cycles=3):
        """Apply cooldown to all strategies after capital flight (4️⃣)."""
        for strategy in self.strategies:
            strategy.cooldown = max(strategy.cooldown, cycles)

    def generate_signals(self, market_data, crisis_signals, exec_efficiency_map=None, regime=None):
        """Return a normalized allocation vector for the ensemble.

        A new optional ``regime`` argument allows strategies or the
        ensemble to behave differently when the market is LOW_VOL /
        HIGH_VOL / CRISIS.  Currently unused by the base strategies but
        available for future experimentation.
        """
        # Penalize low-efficiency strategies (1️⃣)
        if exec_efficiency_map:
            self.penalize_low_efficiency(exec_efficiency_map)

        raw = {s.name: s.score(market_data) for s in self.strategies}

        # Apply efficiency penalties
        for strat_name, penalty in self.efficiency_penalties.items():
            if strat_name in raw:
                raw[strat_name] *= (1.0 - penalty)

        total = sum(raw.values())
        norm = ({k: 0.0 for k in raw} if total <= 0 else {k: v/total for k,v in raw.items()})
        damp = max(0.15, 1.0 - float(crisis_signals.get("severity", 0.0)))
        # further reduce allocations if market is in a crisis regime
        if regime == RegimeDetector.CRISIS:
            damp *= 0.5
        return {k: v*damp for k,v in norm.items()}

    def performance_metrics(self):
        return {"ensemble_health": 1.0, "active_strategies": len(self.strategies)}

