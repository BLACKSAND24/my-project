from config import CONFIG
from execution.shadow_book import ShadowBook
from execution.trade_throttle import TradeThrottle
from utils.logger import get_logger

class ExecutionEngine:
    def __init__(self, mode="PAPER"):
        self.mode = mode
        self.logger = get_logger("EXECUTION")
        self.shadow_book = ShadowBook()
        self.trade_throttle = TradeThrottle(float(CONFIG.get("MIN_TRADE_INTERVAL_SECONDS", 0.5)))

    def _emit_order(self, symbol, notional):
        if self.mode == "LIVE":
            self.logger.info(f"[LIVE] submit order symbol={symbol} notional={notional:.2f}")
        elif self.mode == "SHADOW":
            self.logger.info(f"[SHADOW] simulate order symbol={symbol} notional={notional:.2f}")
        else:
            self.logger.info(f"[PAPER] simulate order symbol={symbol} notional={notional:.2f}")

    def execute(self, allocations, hedge_orders):
        for symbol, notional in (allocations or {}).items():
            if self.trade_throttle.allow(f"alloc::{symbol}"):
                self._emit_order(symbol, float(notional))
        for hedge in (hedge_orders or []):
            if self.trade_throttle.allow(f"hedge::{hedge.get('symbol','UNKNOWN')}"):
                self.logger.info(f"[{self.mode}] hedge_order={hedge}")
        self.shadow_book.apply_allocation(allocations)
        self.shadow_book.apply_hedges(hedge_orders)
        self.logger.info(f"[{self.mode}] executing allocations={allocations} hedges={hedge_orders}")

    def flatten_all(self):
        self.shadow_book.flatten()
        self.logger.warning(f"[{self.mode}] flatten_all called")
