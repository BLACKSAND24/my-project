from financial_organism.config import CONFIG

class CapitalAllocator:
    def __init__(self, total_capital=None):
        self.total_capital = float(total_capital if total_capital is not None else CONFIG.get("STARTING_CAPITAL", 10000.0))
        self.allocations = {}

    def allocate(self, strategies_conf):
        total_conf = sum(float(v) for v in (strategies_conf or {}).values())
        self.allocations = {}
        if total_conf <= 0:
            for k in (strategies_conf or {}):
                self.allocations[k] = 0.0
            return self.allocations
        for k, v in strategies_conf.items():
            self.allocations[k] = self.total_capital * (float(v) / total_conf)
        return self.allocations

    def defensive_mode(self, allocations):
        return self.reduce_risk(allocations, factor=0.5)

    def reduce_risk(self, allocations, factor):
        out = {}
        m = max(0.0, 1.0 - float(factor))
        for k, v in (allocations or {}).items():
            try: out[k] = float(v) * m
            except: out[k] = v
        return out
