class ShadowBook:
    def __init__(self):
        self.positions = {}
        self.order_history = []
    def apply_allocation(self, allocations):
        for symbol, notional in (allocations or {}).items():
            self.positions[symbol] = float(notional)
            self.order_history.append({"symbol": symbol, "notional": float(notional), "kind":"allocation"})
    def apply_hedges(self, hedge_orders):
        for hedge in (hedge_orders or []):
            self.order_history.append(dict(hedge))
    def flatten(self):
        self.positions = {}
        self.order_history.append({"kind":"flatten"})
