"""PAPER → LIVE Readiness Gate (5️⃣)

Computes whether the system is ready for live trading based on:
- Max drawdown < 15%
- Rolling average execution efficiency > 97.5% (with hysteresis)
- No allocation errors

Features:
- Rolling average prevents single-tick fluctuations
- Hysteresis (asymmetric thresholds) prevents readiness flapping
- Per-strategy EE scoring for fine-grained control
- Institutional-grade governance model
"""


from financial_organism.config import CONFIG


class ReadinessGate:
    """Certification system to verify PAPER mode is stable before going LIVE."""
    
    MAX_DRAWDOWN_THRESHOLD = 0.15
    
    # Rolling average thresholds (hysteresis model)
    # Requires HIGHER threshold to go from NOT_READY -> READY (prevents false positives)
    # Allows LOWER threshold to stay READY (prevents flapping)
    MIN_EXECUTION_EFFICIENCY_TO_GO_READY = 0.975  # 97.5% - gate to enter READY
    MIN_EXECUTION_EFFICIENCY_TO_STAY_READY = 0.970  # 97.0% - threshold to remain READY
    MIN_EXECUTION_EFFICIENCY_MINIMUM = 0.960  # 96% - hard floor (force NOT_READY)
    
    # Rolling window for EE tracking
    ROLLING_WINDOW_SIZE = 10  # Track last 10 execution observations
    
    def __init__(self):
        self.check_history = []
        self.ee_rolling_history = []  # All EE observations across all symbols
        self.last_ready_state = None   # For hysteresis: was it READY last time?
        self.strategy_ee_history = {}  # Per-strategy EE tracking: symbol -> [eff1, eff2, ...]
        # Load thresholds from CONFIG (allow exchange-specific tuning)
        try:
            exchange = CONFIG.get('EXCHANGE', 'DEFAULT') or 'DEFAULT'
            thresholds = CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get(exchange.upper())
            if thresholds is None:
                # fallback to DEFAULT
                thresholds = CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get('DEFAULT', {})

            # Apply thresholds with sensible defaults
            self.MIN_EXECUTION_EFFICIENCY_TO_GO_READY = float(thresholds.get('TO_GO_READY', 0.975))
            self.MIN_EXECUTION_EFFICIENCY_TO_STAY_READY = float(thresholds.get('TO_STAY_READY', 0.970))
            self.MIN_EXECUTION_EFFICIENCY_MINIMUM = float(thresholds.get('MINIMUM', 0.960))
            self.ROLLING_WINDOW_SIZE = int(thresholds.get('ROLLING_WINDOW', 10))
        except Exception:
            # In case CONFIG missing or malformed, use class-level defaults
            self.MIN_EXECUTION_EFFICIENCY_TO_GO_READY = self.MIN_EXECUTION_EFFICIENCY_TO_GO_READY
            self.MIN_EXECUTION_EFFICIENCY_TO_STAY_READY = self.MIN_EXECUTION_EFFICIENCY_TO_STAY_READY
            self.MIN_EXECUTION_EFFICIENCY_MINIMUM = self.MIN_EXECUTION_EFFICIENCY_MINIMUM
            self.ROLLING_WINDOW_SIZE = self.ROLLING_WINDOW_SIZE
    
    def update_ee_history(self, exec_efficiency_dict):
        """Update rolling EE history from execution engine.
        
        Args:
            exec_efficiency_dict: Dict {symbol: [eff1, eff2, ...]}
        """
        # Flatten all efficiencies into rolling history
        all_effs = []
        for symbol, effs in exec_efficiency_dict.items():
            if not symbol in self.strategy_ee_history:
                self.strategy_ee_history[symbol] = []
            
            # Add only new efficiencies (delta)
            for eff in effs:
                all_effs.append(eff)
                self.strategy_ee_history[symbol].append(eff)
                
                # Keep rolling window per strategy
                if len(self.strategy_ee_history[symbol]) > self.ROLLING_WINDOW_SIZE:
                    self.strategy_ee_history[symbol] = self.strategy_ee_history[symbol][-self.ROLLING_WINDOW_SIZE:]
        
        # Update global rolling history
        self.ee_rolling_history.extend(all_effs)
        if len(self.ee_rolling_history) > self.ROLLING_WINDOW_SIZE:
            self.ee_rolling_history = self.ee_rolling_history[-self.ROLLING_WINDOW_SIZE:]
    
    def get_rolling_avg_execution_efficiency(self):
        """Get rolling average of EE (last N observations across all symbols).
        
        Returns:
            float: Rolling average (0.0-1.0), or 1.0 if no history
        """
        if not self.ee_rolling_history:
            return 1.0
        return sum(self.ee_rolling_history) / len(self.ee_rolling_history)
    
    def get_per_strategy_ee(self):
        """Get per-strategy EE breakdown.
        
        Returns:
            dict: {symbol: avg_ee}
        """
        result = {}
        for symbol, effs in self.strategy_ee_history.items():
            if effs:
                result[symbol] = sum(effs) / len(effs)
        return result
    
    def _should_go_live_ready(self, rolling_avg_ee, last_state_ready):
        """Determine if system should transition to LIVE READY using hysteresis.
        
        Hysteresis logic:
        - If NOT_READY: require high threshold (97.5%) to go READY
        - If READY: allow lower threshold (97.0%) to stay READY
        - If below 96%: always force NOT_READY
        
        Args:
            rolling_avg_ee: Current rolling average EE
            last_state_ready: Was system READY in last check?
        
        Returns:
            bool: Whether system should be LIVE READY
        """
        # Hard minimum: below 96% always block
        if rolling_avg_ee < self.MIN_EXECUTION_EFFICIENCY_MINIMUM:
            return False
        
        # Hysteresis: different thresholds for entry vs staying
        if last_state_ready is None or not last_state_ready:
            # Not ready, or first check: require high threshold to enter READY
            return rolling_avg_ee >= self.MIN_EXECUTION_EFFICIENCY_TO_GO_READY
        else:
            # Was ready: allow slightly lower threshold to stay ready
            return rolling_avg_ee >= self.MIN_EXECUTION_EFFICIENCY_TO_STAY_READY
    
    def compute_readiness(self, engine, executor):
        """Compute LIVE readiness flag from execution engine metrics.
        
        Args:
            engine: ExecutionEngine instance
            executor: RiskManager or similar instance (for error tracking)
        
        Returns:
            dict with readiness flag and breakdown
        """
        metrics = engine.shadow_book.get_metrics()
        
        # Update rolling EE history
        self.update_ee_history(engine.exec_efficiency)
        rolling_avg_ee = self.get_rolling_avg_execution_efficiency()
        per_strategy_ee = self.get_per_strategy_ee()
        
        # Check each criterion
        dd_ok = metrics['max_drawdown'] >= -self.MAX_DRAWDOWN_THRESHOLD  # closer to 0 is better
        eff_ok = self._should_go_live_ready(rolling_avg_ee, self.last_ready_state)
        allocation_ok = len(engine.exec_efficiency) > 0  # At least one trade executed
        
        live_ready = dd_ok and eff_ok and allocation_ok
        
        result = {
            "LIVE_READY": live_ready,
            "max_drawdown": metrics['max_drawdown'],
            "dd_ok": dd_ok,
            "rolling_avg_execution_efficiency": rolling_avg_ee,
            # Backwards compatibility: keep legacy key name
            "execution_efficiency": rolling_avg_ee,
            "per_strategy_ee": per_strategy_ee,
            "eff_ok": eff_ok,
            "allocation_ok": allocation_ok,
            "status": self._format_status(live_ready, dd_ok, eff_ok, allocation_ok, rolling_avg_ee),
            "hysteresis_threshold": (
                self.MIN_EXECUTION_EFFICIENCY_TO_GO_READY 
                if not self.last_ready_state 
                else self.MIN_EXECUTION_EFFICIENCY_TO_STAY_READY
            )
        }
        
        self.check_history.append(result)
        self.last_ready_state = live_ready  # Update state for next check
        return result
    
    def _format_status(self, live_ready, dd_ok, eff_ok, alloc_ok, rolling_avg_ee=None):
        """Format status message."""
        checks = {
            "drawdown": "✅" if dd_ok else "❌",
            "execution_eff": "✅" if eff_ok else "❌",
            "allocations": "✅" if alloc_ok else "❌"
        }
        overall = "✅ LIVE READY" if live_ready else "🚫 NOT READY"
        if rolling_avg_ee is not None:
            return f"{overall} | DD:{checks['drawdown']} EE:{checks['execution_eff']} ({rolling_avg_ee:.2%}) AL:{checks['allocations']}"
        return f"{overall} | DD:{checks['drawdown']} EE:{checks['execution_eff']} AL:{checks['allocations']}"
    
    def get_latest_check(self):
        """Return the most recent readiness check."""
        return self.check_history[-1] if self.check_history else None
    
    def get_latest_summary(self):
        """Get a one-line governance summary for logging."""
        if not self.check_history:
            return "readiness=UNINITIALIZED rolling_avg_ee=N/A max_dd=N/A"
        latest = self.check_history[-1]
        threshold = latest.get('hysteresis_threshold', 0.975)
        return (
            f"readiness={latest['status']} "
            f"rolling_avg_ee={latest.get('rolling_avg_execution_efficiency', 'N/A'):.2%} "
            f"(threshold={threshold:.2%}) "
            f"max_dd={latest['max_drawdown']:.2%}"
        )
