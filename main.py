"""
Financial Organism — Main Entry Point
Survival > Prediction > Profit

Modes:
- PAPER   : no real orders, full risk + evolution active
- SHADOW  : live data, simulated execution
- LIVE    : real capital (requires checklist approval)
"""

import time
import sys
import os

# Add repo root to path for imports to work from any directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

from financial_organism.data.fetch_data import fetch_market_data
from financial_organism.data.live_feed import LiveMarketFeed
from financial_organism.data.macro_data import simple_rss_sentiment, fetch_fred_series
from financial_organism.strategies.quant_models import StrategyEnsemble
from financial_organism.capital.capital_allocator import CapitalAllocator
from financial_organism.capital.capital_flight_detector import CapitalFlightDetector

from financial_organism.risk.risk_manager import RiskManager
from financial_organism.risk.black_swan_hedger import BlackSwanHedger

from financial_organism.simulation.crisis_simulator import CrisisSimulator
from financial_organism.evolution.genetic_engine import GeneticEngine

from financial_organism.execution.execution_engine import ExecutionEngine
from financial_organism.governance.ai_risk_committee import AIRiskCommittee
from financial_organism.governance.paper_to_live_readiness import ReadinessGate
from financial_organism.monitoring.ee_monitor import EE_MONITOR
from financial_organism.monitoring.telegram_alerts import poll_commands
from financial_organism.macro.regime_detector import RegimeDetector
from financial_organism.monitoring.analytics import report as ee_report

import json

# BurnInMonitor for SHADOW mode cycle tracking
class BurnInMonitor:
    """Tracks SHADOW burn-in cycles for DRY-RUN readiness."""
    def __init__(self, checkpoint_file="burn_in_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self._load()
        self.cycle_count = self.checkpoint.get('cycle_count', 0)
        # Whether we've already emitted the 'ready' notification
        self._ready_emitted = self.is_ready()
    
    def _load(self):
        try:
            with open(self.checkpoint_file) as f:
                return json.load(f)
        except:
            return {'cycle_count': 0, 'cycles': []}
    
    def record_cycle(self, executor_efficiency, max_drawdown):
        """Record one cycle of SHADOW execution."""
        self.cycle_count += 1
        cycle_data = {
            'cycle': self.cycle_count,
            'timestamp': time.time(),
            'fill_efficiency': executor_efficiency,
            'max_drawdown': max_drawdown,
            'ready_for_dryrun': self.cycle_count >= CONFIG.get("DRY_RUN_BURN_IN_CYCLES", 50)
        }
        self.checkpoint['cycles'].append(cycle_data)
        self.checkpoint['cycle_count'] = self.cycle_count
        self._save()
        # Determine if readiness transitioned to True this cycle
        ready_now = self.cycle_count >= CONFIG.get("DRY_RUN_BURN_IN_CYCLES", 50)
        newly_ready = ready_now and (not self._ready_emitted)
        if newly_ready:
            self._ready_emitted = True
        return cycle_data, newly_ready

    def reset(self):
        """Reset the burn-in checkpoint (useful for fresh runs)."""
        self.checkpoint = {'cycle_count': 0, 'cycles': []}
        self.cycle_count = 0
        self._ready_emitted = False
        self._save()
    
    def _save(self):
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_progress_pct(self):
        target = CONFIG.get("DRY_RUN_BURN_IN_CYCLES", 50)
        return (self.cycle_count / target) * 100 if target > 0 else 0
    
    def is_ready(self):
        target = CONFIG.get("DRY_RUN_BURN_IN_CYCLES", 50)
        return self.cycle_count >= target


# -----------------------------
# INITIALIZATION
# -----------------------------

logger = get_logger("MAIN")

MODE = CONFIG.get("MODE", "PAPER")  # PAPER | SHADOW | LIVE
burn_in_monitor = None  # Will be initialized if MODE is SHADOW


def system_banner():
    logger.info("=" * 80)
    logger.info("🧠 FINANCIAL ORGANISM BOOTING")
    logger.info(f"MODE            : {MODE}")
    logger.info("PRIORITY        : Survival > Prediction > Profit")
    logger.info("=" * 80)


# -----------------------------
# MAIN LOOP
# -----------------------------

def main(cycles: int = None, interval: float = None, reset_burn: bool = False):
    global burn_in_monitor

    # Allow runtime override of loop interval
    if interval is not None:
        try:
            CONFIG['LOOP_INTERVAL'] = float(interval)
        except Exception:
            pass

    system_banner()
    
    # Initialize burn_in_monitor for SHADOW mode
    if MODE == "SHADOW":
        burn_in_monitor = BurnInMonitor()
        if reset_burn:
            burn_in_monitor.reset()
            logger.info("🔄 Burn-in checkpoint reset before run")
        logger.info(f"🔥 SHADOW BURN-IN MODE: {burn_in_monitor.cycle_count} / {CONFIG.get('DRY_RUN_BURN_IN_CYCLES', 50)} cycles")

    # --- Core Components ---
    strategy_ensemble = StrategyEnsemble()
    allocator = CapitalAllocator()
    capital_flight = CapitalFlightDetector()

    risk_manager = RiskManager()
    black_swan = BlackSwanHedger()

    # portfolio-level AI uses the same global brain for correlations
    from financial_organism.ai.portfolio_ai import PortfolioAI
    portfolio_ai = PortfolioAI(risk_manager.global_brain)

    crisis_simulator = CrisisSimulator()
    genetic_engine = GeneticEngine()

    executor = ExecutionEngine(mode=MODE)
    ai_committee = AIRiskCommittee()
    readiness_gate = ReadinessGate()  # 5️⃣ PAPER → LIVE readiness checker

    # --- Safety Gate ---
    if MODE == "LIVE":
        ai_committee.assert_go_live_ready()

    # --- Data Source Selection (MODE-aware) ---
    if MODE == "PAPER":
        data_source = "SimulatedMarket"  # Use simulated market data
        logger.info("[PAPER] Data source: Simulated market data")
    elif MODE == "SHADOW":
        market_feed = LiveMarketFeed()  # Read-only live feed
        data_source = "LiveMarketFeed"
        logger.info("[SHADOW] Data source: Live market data (SIMULATED execution)")
    elif MODE == "LIVE":
        market_feed = LiveMarketFeed()  # Real execution with live feed
        data_source = "LiveMarketFeed"
        logger.info("[LIVE] Data source: Live market data (REAL execution)")
    else:
        raise ValueError(f"Invalid MODE: {MODE}")

    logger.info("System initialized successfully.")

    # -----------------------------
    # LIVE LOOP
    # -----------------------------
    cycle_counter = 0
    while True:
        try:
            # 1️⃣ Fetch Market Data (MODE-aware data source)
            # attach lightweight macro signals (free!)
            rss = CONFIG.get("NEWS_RSS_URL", "")
            if rss:
                try:
                    market_data["macro_sentiment"] = simple_rss_sentiment(rss)
                except Exception:
                    market_data["macro_sentiment"] = 0.0
            # FRED series may be expensive, call sparingly
            fred_id = CONFIG.get("FRED_SERIES", None)
            if fred_id and cycle_counter % 60 == 0:  # once per 60 cycles
                market_data["fred"] = fetch_fred_series(fred_id)
            # check for any human-AI commands coming over telegram
            try:
                poll_commands()
            except Exception:
                pass
            if MODE == "PAPER":
                market_data = fetch_market_data()
            else:  # SHADOW or LIVE
                market_data = market_feed.get_market_data()
            
            # Optional: Log divergence between PAPER and SHADOW (if both active)
            if MODE == "SHADOW":
                logger.debug(f"[SHADOW→Live] Data source active. Prices from live feed.")

            # 2️⃣ Run Crisis Injection (synthetic + historical)
            crisis_signals = crisis_simulator.evaluate(market_data)

            # detect market regime (low-vol/high-vol/crisis) for later use
            regime = RegimeDetector().detect(market_data)
            logger.debug(f"Market regime detected: {regime}")

            # 3️⃣ Strategy Decisions (with execution efficiency awareness - 1️⃣)
            # Build efficiency map: translate exec_efficiency by symbol to strategy names
            exec_eff_by_strategy = {}
            for symbol, effs in executor.exec_efficiency.items():
                # Map symbol names to strategy names if available (BTC→"vol_breakout", etc.)
                if symbol in strategy_ensemble.strategies:
                    exec_eff_by_strategy[symbol] = sum(effs) / len(effs) if effs else 1.0
            
            strategy_outputs = strategy_ensemble.generate_signals(
                market_data=market_data,
                crisis_signals=crisis_signals,
                exec_efficiency_map=exec_eff_by_strategy,  # Pass efficiency for penalty logic
                regime=regime
            )

            # apply portfolio-level AI to adjust strategy weights
            try:
                weights = portfolio_ai.get_weights(strategy_outputs)
            except Exception:
                weights = strategy_outputs  # fallback

            # confidence gating: if the leading weight is too small, shrink all
            confidence = max(weights.values()) if weights else 0.0
            min_conf = float(CONFIG.get("MIN_PORTFOLIO_CONFIDENCE", 0.20))
            if confidence < min_conf and confidence > 0:
                shrink = confidence / min_conf
                logger.warning(f"📉 low portfolio confidence {confidence:.2%} < {min_conf:.2%}, shrinking exposures by {shrink:.2%}")
                weights = {k: v * shrink for k, v in weights.items()}
            
            # 4️⃣ Capital Allocation
            allocations = allocator.allocate(weights)
            # 6️⃣ Risk Evaluation (KILL-SWITCH LAYER)
            risk_ok = risk_manager.evaluate(
                allocations=allocations,
                market_data=market_data,
                crisis_signals=crisis_signals
            )

            if not risk_ok:
                logger.critical("🛑 Risk limits breached — flattening book")
                executor.flatten_all()
                continue

            # 7️⃣ Black Swan Auto-Hedging
            hedge_orders = black_swan.generate_hedges(
                market_data=market_data,
                allocations=allocations
            )

            # 8️⃣ Execute Trades
            exec_summary = executor.execute(allocations, hedge_orders, market_data=market_data)
            # Divergence Analytics: Log PAPER vs SHADOW behavior (optional, for research)
            if MODE == "SHADOW":
                # Track how live market data diverges from simulated behavior
                logger.debug(
                    f"[SHADOW] Allocations: {allocations} | "
                    f"Live vol: {market_data.get('volatility', 0):.3f} | "
                    f"Hedges active: {len(hedge_orders) > 0}"
                )

            # Record EE metrics and one-line summary for LIVE mode too
            if MODE == "LIVE":
                try:
                    filled_allocs = exec_summary.get('filled_allocations', {})
                    total_filled = sum(filled_allocs.values())
                    capital_util = total_filled / max(executor.starting_capital, 1.0)
                    latency_ms = exec_summary.get('latency_ms', 0.0)
                    # In LIVE mode, readiness gate already asserted at startup
                    readiness_flag = True
                    for symbol, requested in (allocations or {}).items():
                        filled = filled_allocs.get(symbol, 0.0)
                        EE_MONITOR.record(symbol, requested, filled, latency_ms, capital_util, readiness_flag)

                    # Ensure we have a readiness snapshot for metrics (compute if needed)
                    try:
                        readiness_local = readiness_gate.compute_readiness(executor, risk_manager)
                    except Exception:
                        readiness_local = {}

                    # Aggregated metrics (approximate)
                    rolling_avg_ee = executor.get_avg_execution_efficiency()
                    exch = CONFIG.get('EXCHANGE', 'DEFAULT').upper()
                    thresholds = CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get(exch) or CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get('DEFAULT', {})
                    window = int(thresholds.get('ROLLING_WINDOW', 10))
                    ee_min = float(thresholds.get('TO_GO_READY', 0.975))
                    hysteresis = float(thresholds.get('TO_STAY_READY', 0.970) - ee_min) if thresholds else 0.0
                    per_strategy = {s: (sum(effs) / len(effs) if effs else 1.0) for s, effs in executor.exec_efficiency.items()}
                    cap_pct = capital_util * 100.0
                    metrics = {
                        'rolling_avg_ee': rolling_avg_ee * 100.0 if rolling_avg_ee <= 1.0 else rolling_avg_ee,
                        'window': window,
                        'ee_min': ee_min * 100.0 if ee_min <= 1.0 else ee_min,
                        'hysteresis': hysteresis * 100.0 if abs(hysteresis) <= 1.0 else hysteresis,
                        'per_strategy': per_strategy,
                        'latency_min_ms': float(latency_ms),
                        'latency_max_ms': float(latency_ms),
                        'capital_util_min_pct': cap_pct,
                        'capital_util_max_pct': cap_pct,
                        'max_dd': readiness_local.get('max_drawdown', 0.0) * 100.0 if isinstance(readiness_local, dict) else 0.0
                    }
                    summary = ee_report(metrics)
                    logger.info(f"[EE_SUMMARY] {summary}")
                except Exception:
                    logger.exception("EE monitor failed to record LIVE cycle")

            # 9️⃣ Evolution Pressure (Strategy Darwinism)
            genetic_engine.evolve(
                performance=strategy_ensemble.performance_metrics(),
                crisis_feedback=crisis_signals
            )

            # 🔟 PAPER → LIVE Readiness Check (5️⃣)
            if MODE in ("PAPER", "SHADOW"):
                readiness = readiness_gate.compute_readiness(executor, risk_manager)
                logger.info(f"Readiness: {readiness['status']}")
                # Add governance one-glance safety log (expert suggestion)
                logger.info(f"[GOVERNANCE] {readiness_gate.get_latest_summary()}")
                
                # Track SHADOW burn-in cycles for DRY-RUN readiness
                if MODE == "SHADOW" and burn_in_monitor:
                    # Use new rolling average EE metric (improved readiness gate)
                    exec_eff = readiness.get('rolling_avg_execution_efficiency', readiness.get('execution_efficiency', 0.0))
                    max_dd = readiness.get('max_drawdown', 0.0)
                    _, newly_ready = burn_in_monitor.record_cycle(exec_eff, max_dd)
                    pct = burn_in_monitor.get_progress_pct()
                    logger.info(f"[BURN-IN] {burn_in_monitor.cycle_count}/{CONFIG.get('DRY_RUN_BURN_IN_CYCLES', 50)} ({pct:.0f}%)")
                    if newly_ready:
                        logger.critical("✅ SHADOW BURN-IN COMPLETE! Ready to switch to DRY-RUN mode.")

                # EE Monitoring: record per-strategy metrics to CSV/logs
                try:
                    filled_allocs = exec_summary.get('filled_allocations', {})
                    total_filled = sum(filled_allocs.values())
                    capital_util = total_filled / max(executor.starting_capital, 1.0)
                    readiness_flag = readiness.get('LIVE_READY', False)
                    latency_ms = exec_summary.get('latency_ms', 0.0)
                    for symbol, requested in (allocations or {}).items():
                        filled = filled_allocs.get(symbol, 0.0)
                        EE_MONITOR.record(symbol, requested, filled, latency_ms, capital_util, readiness_flag)
                    # Also emit an aggregated one-line EE summary and persist metrics
                    try:
                        rolling_avg_ee = readiness.get('rolling_avg_execution_efficiency', readiness.get('execution_efficiency', 0.0))
                        # Determine configured rolling window
                        exch = CONFIG.get('EXCHANGE', 'DEFAULT').upper()
                        thresholds = CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get(exch) or CONFIG.get('EXEC_EFFICIENCY_THRESHOLDS', {}).get('DEFAULT', {})
                        window = int(thresholds.get('ROLLING_WINDOW', 10))
                        ee_min = float(thresholds.get('TO_GO_READY', 0.975))
                        hysteresis = float(thresholds.get('TO_STAY_READY', 0.970) - ee_min) if thresholds else 0.0
                        # per_strategy efficiencies can be approximated from executor.exec_efficiency
                        per_strategy = {s: (sum(effs) / len(effs) if effs else 1.0) for s, effs in executor.exec_efficiency.items()}
                        cap_min = cap_max = capital_util * 100.0
                        latency_min = latency_max = float(latency_ms)
                        max_dd = readiness.get('max_drawdown', 0.0)
                        metrics = {
                            'rolling_avg_ee': rolling_avg_ee * 100.0 if rolling_avg_ee <= 1.0 else rolling_avg_ee,
                            'window': window,
                            'ee_min': ee_min * 100.0 if ee_min <= 1.0 else ee_min,
                            'hysteresis': hysteresis * 100.0 if abs(hysteresis) <= 1.0 else hysteresis,
                            'per_strategy': per_strategy,
                            'latency_min_ms': latency_min,
                            'latency_max_ms': latency_max,
                            'capital_util_min_pct': cap_min,
                            'capital_util_max_pct': cap_max,
                            'max_dd': max_dd * 100.0 if abs(max_dd) <= 1.0 else max_dd
                        }
                        summary = ee_report(metrics)
                        logger.info(f"[EE_SUMMARY] {summary}")
                    except Exception:
                        logger.exception("Failed to emit EE summary")
                except Exception:
                    logger.exception("EE monitor failed to record cycle")

            cycle_counter += 1

            # If cycles limit provided, exit after reaching it
            if cycles is not None and cycle_counter >= int(cycles):
                logger.info(f"Reached requested cycle limit: {cycle_counter}/{cycles}. Exiting.")
                break

            time.sleep(CONFIG.get("LOOP_INTERVAL", 5))

        except KeyboardInterrupt:
            logger.warning("Manual shutdown requested.")
            executor.flatten_all()
            sys.exit(0)

        except Exception as e:
            logger.exception(f"Unhandled system error: {e}")
            executor.flatten_all()
            time.sleep(10)


# -----------------------------
# ENTRY
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Financial Organism main loop")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles to run (SHADOW smoke/burn-in)")
    parser.add_argument("--interval", type=float, default=None, help="Override LOOP_INTERVAL (seconds) for faster runs)")
    parser.add_argument("--reset-burn", action="store_true", help="Reset SHADOW burn-in checkpoint before running")
    args = parser.parse_args()

    main(cycles=args.cycles, interval=args.interval, reset_burn=args.reset_burn)
