import random
import time
from financial_organism.config import CONFIG
from financial_organism.execution.shadow_book import ShadowBook
from financial_organism.execution.trade_throttle import TradeThrottle
from financial_organism.utils.logger import get_logger

class ExecutionEngine:
    # Realistic market microstructure costs (bps)
    SLIPPAGE_PCT = 0.0005       # 5 bps
    FEE_PCT = 0.0004            # 4 bps
    PARTIAL_FILL_MIN = 0.85     # min fill ratio
    PARTIAL_FILL_MAX = 1.0      # max fill ratio
    LATENCY_MIN_MS = 10         # simulate 10-50ms latency per cycle
    LATENCY_MAX_MS = 50
    MAX_UTILIZATION = 0.85      # Capital utilization ceiling (2️⃣)
    DEFENSIVE_DD_THRESHOLD = 0.12  # Drawdown trigger for defensive mode (3️⃣)

    def __init__(self, mode="PAPER"):
        self.mode = mode
        self.logger = get_logger("EXECUTION")
        self.shadow_book = ShadowBook()
        self.trade_throttle = TradeThrottle(float(CONFIG.get("MIN_TRADE_INTERVAL_SECONDS", 0.5)))
        self.starting_capital = float(CONFIG.get("STARTING_CAPITAL", 10000.0))
        # Execution efficiency tracking (1️⃣)
        self.exec_efficiency = {}  # strategy -> [rolling window of efficiencies]
        # Drawdown monitoring (3️⃣)
        self.force_defensive = False
        
        # SHADOW mode safety guard (Critical!)
        if self.mode == "SHADOW":
            assert not self._has_trade_permissions(), \
                "SHADOW mode MUST NEVER have trade permissions!"
            self.logger.warning("⚠️ SHADOW MODE ACTIVE: Simulated execution only, NO REAL TRADES")
    
    def _has_trade_permissions(self) -> bool:
        """Check if this engine could possibly place real trades.
        
        For now, always False in SHADOW == no real trade capability.
        """
        return False

    def _apply_costs(self, notional):
        """Apply transaction costs (slippage + fees) to notional."""
        cost = notional * (self.SLIPPAGE_PCT + self.FEE_PCT)
        return max(0.0, notional - cost)

    def _apply_partial_fill(self, notional):
        """Simulate partial fill (85-100%)."""
        fill_ratio = random.uniform(self.PARTIAL_FILL_MIN, self.PARTIAL_FILL_MAX)
        return notional * fill_ratio

    def _cap_allocations(self, allocations):
        """Guard against over-allocation: scale if total > capital."""
        total = sum((allocations or {}).values())
        if total <= 0:
            return allocations or {}
        if total > self.starting_capital:
            scale = self.starting_capital / total
            return {k: v * scale for k, v in allocations.items()}
        return allocations or {}

    def _apply_capital_ceiling(self, allocations):
        """Apply capital utilization ceiling (2️⃣) to avoid over-leverage."""
        return {k: v * self.MAX_UTILIZATION for k, v in (allocations or {}).items()}

    def _track_execution_efficiency(self, symbol, requested, filled):
        """Track rolling FILL EFFICIENCY per symbol (1️⃣).
        
        Fill efficiency = filled / requested (measures execution quality)
        NOT capital utilization = filled / total_capital (different metric)
        
        Keeps rolling window of 20 observations for statistical significance.
        """
        if symbol not in self.exec_efficiency:
            self.exec_efficiency[symbol] = []
        
        # CORRECT: Fill efficiency (order quality)
        if requested > 0:
            fill_efficiency = filled / requested
        else:
            fill_efficiency = 1.0
        
        self.exec_efficiency[symbol].append(fill_efficiency)
        # Keep last 20 observations for better rolling average (was 5)
        if len(self.exec_efficiency[symbol]) > 20:
            self.exec_efficiency[symbol] = self.exec_efficiency[symbol][-20:]

    def get_avg_execution_efficiency(self, symbol=None):
        """Get average execution efficiency for a symbol or all symbols."""
        if not self.exec_efficiency:
            return 1.0
        if symbol:
            effs = self.exec_efficiency.get(symbol, [])
            return sum(effs) / len(effs) if effs else 1.0
        all_effs = [e for effs in self.exec_efficiency.values() for e in effs]
        return sum(all_effs) / len(all_effs) if all_effs else 1.0

    def check_drawdown_defensive_trigger(self):
        """Check if drawdown exceeded threshold and force defensive mode (3️⃣)."""
        metrics = self.shadow_book.get_metrics()
        # Only trigger if we have equity data and drawdown is significantly negative
        if len(self.shadow_book.equity_curve) > 5 and metrics['max_drawdown'] < -self.DEFENSIVE_DD_THRESHOLD:
            self.force_defensive = True
            self.logger.warning(f"🚨 Drawdown {metrics['max_drawdown']:.2%} < -{self.DEFENSIVE_DD_THRESHOLD:.2%} — forcing defensive mode")
            return True
        return False

    def _emit_order(self, symbol, notional):
        # apply realistic costs
        adjusted = self._apply_costs(float(notional))
        # simulate partial fill
        filled = self._apply_partial_fill(adjusted)
        
        if self.mode == "LIVE":
            self.logger.info(f"[LIVE] submit order symbol={symbol} requested={notional:.2f} filled={filled:.2f}")
        elif self.mode == "SHADOW":
            self.logger.info(f"[SHADOW] simulate order symbol={symbol} requested={notional:.2f} filled={filled:.2f}")
        else:
            self.logger.info(f"[PAPER] simulate order symbol={symbol} requested={notional:.2f} filled={filled:.2f}")
        
        return filled

    def execute(self, allocations, hedge_orders):
        # CRITICAL SAFETY: SHADOW must NEVER place real trades
        if self.mode == "SHADOW":
            assert self._has_trade_permissions() == False, \
                f"SHADOW mode MUST NOT have trade permissions! Aborting order."
        
        # Simulate execution latency (once per cycle, not per order)
        latency_ms = random.uniform(self.LATENCY_MIN_MS, self.LATENCY_MAX_MS)
        time.sleep(latency_ms / 1000.0)
        
        # Check drawdown and set defensive flag (3️⃣)
        self.check_drawdown_defensive_trigger()
        
        # Guard: prevent over-allocation (capital conservation)
        allocations = self._cap_allocations(allocations)
        
        # Apply capital utilization ceiling (2️⃣)
        allocations = self._apply_capital_ceiling(allocations)
        
        # Apply defensive mode reduction if triggered (3️⃣)
        if self.force_defensive:
            allocations = {k: v * 0.5 for k, v in allocations.items()}
            self.logger.info(f"[{self.mode}] applying defensive mode: allocations reduced by 50%")
        
        filled_allocs = {}
        for symbol, notional in (allocations or {}).items():
            if self.trade_throttle.allow(f"alloc::{symbol}"):
                filled = self._emit_order(symbol, float(notional))
                filled_allocs[symbol] = filled
                # Track execution efficiency (1️⃣)
                self._track_execution_efficiency(symbol, float(notional), filled)
            else:
                filled_allocs[symbol] = float(notional)
        
        for hedge in (hedge_orders or []):
            if self.trade_throttle.allow(f"hedge::{hedge.get('symbol','UNKNOWN')}"):
                self.logger.info(f"[{self.mode}] hedge_order={hedge}")
        
        # update shadow book and persist PAPER-mode handoff
        self.shadow_book.apply_allocation(filled_allocs)
        self.shadow_book.apply_hedges(hedge_orders)
        self.shadow_book.record_regime(self.mode, allocations, hedge_orders)
        
        try:
            self.shadow_book.record_handoff(self.mode, filled_allocs, hedge_orders)
        except Exception:
            self.logger.exception("Failed to record handoff snapshot")
        
        # Calculate both metrics (fill efficiency vs capital utilization)
        fill_eff = self.get_avg_execution_efficiency()
        total_filled = sum(filled_allocs.values())
        capital_util = total_filled / max(self.starting_capital, 1.0)
        
        self.logger.info(
            f"[{self.mode}] executing allocations={filled_allocs} "
            f"fill_eff={fill_eff:.2%} "
            f"capital_util={capital_util:.2%} "
            f"latency={latency_ms:.1f}ms" + 
            (" [USING LIVE PRICES, SIMULATED EXECUTION]" if self.mode == "SHADOW" else "")
        )
        
        # Return execution summary
        return {
            'mode': self.mode,
            'execution_efficiency': self.get_avg_execution_efficiency(),
            'filled_allocations': filled_allocs,
            'defensive_active': self.force_defensive,
            'latency_ms': latency_ms
        }

    def flatten_all(self):
        self.shadow_book.flatten()
        self.logger.warning(f"[{self.mode}] flatten_all called")
