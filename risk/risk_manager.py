from financial_organism.config import CONFIG
from financial_organism.risk.kill_switch import KillSwitch

class RiskManager:
    def __init__(self):
        start = float(CONFIG.get("STARTING_CAPITAL", 10000.0))
        self.equity_peak = start
        self.current_equity = start
        self.kill_switch = KillSwitch(
            max_drawdown=float(CONFIG.get("MAX_PORTFOLIO_DRAWDOWN", 0.10)),
            cooldown_seconds=int(CONFIG.get("KILL_SWITCH_COOLDOWN", 300)),
        )
    def evaluate(self, allocations, market_data, crisis_signals):
        returns = market_data.get("returns", [])
        if returns:
            self.current_equity += float(returns[-1]) * 100.0
            self.equity_peak = max(self.equity_peak, self.current_equity)
        if self.kill_switch.is_locked(): return False
        if self.kill_switch.check(self.equity_peak, self.current_equity): return False
        return True
