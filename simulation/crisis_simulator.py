import math
import statistics
from financial_organism.config import CONFIG

class CrisisSimulator:
    def __init__(self, vol_threshold=None, drawdown_threshold=None):
        self.volatility_threshold = float(vol_threshold if vol_threshold is not None else CONFIG.get("CRISIS_VOL_THRESHOLD", 0.35))
        self.drawdown_threshold = float(drawdown_threshold if drawdown_threshold is not None else CONFIG.get("CRISIS_DRAWDOWN_THRESHOLD", -0.20))

    def evaluate(self, market_data):
        returns = [float(x) for x in market_data.get("returns", [])]
        if not returns:
            return {"crisis": False, "severity": 0.0, "action": None, "volatility": 0.0, "recent_return": 0.0}
        volatility = statistics.pstdev(returns) * math.sqrt(252)
        cumulative, drawdown = 0.0, 0.0
        for v in returns:
            cumulative += v
            if cumulative < drawdown:
                drawdown = cumulative
        recent_return = returns[-1]
        crisis = volatility > self.volatility_threshold or drawdown < self.drawdown_threshold or recent_return < -0.05
        severity = min(1.0, max(
            volatility / max(self.volatility_threshold, 1e-6),
            abs(drawdown / min(self.drawdown_threshold, -1e-6)),
            abs(recent_return / -0.05),
        ))
        return {"crisis": crisis, "severity": round(severity, 2), "action": "REDUCE" if crisis else None, "volatility": volatility, "recent_return": recent_return}
