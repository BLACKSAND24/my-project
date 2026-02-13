import time
class KillSwitch:
    def __init__(self, max_drawdown=0.15, cooldown_seconds=300):
        self.max_drawdown=max_drawdown; self.cooldown_seconds=cooldown_seconds
        self.triggered_at=None; self.active=False
    def check(self, equity_peak, current_equity):
        if equity_peak <= 0: return False
        drawdown = (equity_peak-current_equity)/equity_peak
        if drawdown >= self.max_drawdown:
            self.active=True; self.triggered_at=time.time(); return True
        return False
    def is_locked(self):
        if not self.active: return False
        if time.time()-self.triggered_at > self.cooldown_seconds:
            self.active=False; return False
        return True
