import statistics
from config import CONFIG

class BlackSwanHedger:
    def __init__(self):
        self.enabled = bool(CONFIG.get("AUTO_HEDGE_ENABLED", True))
        self.vol_trigger = float(CONFIG.get("HEDGE_VOL_TRIGGER", 0.18))
        self.max_hedge_ratio = float(CONFIG.get("MAX_HEDGE_RATIO", 0.35))
        self.hedge_symbol = CONFIG.get("HEDGE_SYMBOL", "INDEX_PUT_PROXY")

    def generate_hedges(self, market_data, allocations):
        if not self.enabled: return []
        r = [float(x) for x in market_data.get("returns",[])]
        if len(r) < 5: return []
        vol = statistics.pstdev(r) * (252 ** 0.5)
        if vol <= self.vol_trigger: return []
        gross_long = sum(max(float(v),0.0) for v in (allocations or {}).values())
        if gross_long <= 0: return []
        stress = min(1.0, (vol - self.vol_trigger) / max(self.vol_trigger, 1e-6))
        hedge_notional = gross_long * min(self.max_hedge_ratio, self.max_hedge_ratio * stress)
        return [{"symbol": self.hedge_symbol, "side":"BUY", "type":"HEDGE", "notional": round(hedge_notional,2), "reason": f"volatility={vol:.4f}"}]
