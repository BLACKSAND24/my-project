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

from config import CONFIG
from utils.logger import get_logger

from data.fetch_data import fetch_market_data
from strategies.quant_models import StrategyEnsemble
from capital.capital_allocator import CapitalAllocator
from capital.capital_flight_detector import CapitalFlightDetector

from risk.risk_manager import RiskManager
from risk.black_swan_hedger import BlackSwanHedger

from simulation.crisis_simulator import CrisisSimulator
from evolution.genetic_engine import GeneticEngine

from execution.execution_engine import ExecutionEngine
from governance.ai_risk_committee import AIRiskCommittee


# -----------------------------
# INITIALIZATION
# -----------------------------

logger = get_logger("MAIN")

MODE = CONFIG.get("MODE", "PAPER")  # PAPER | SHADOW | LIVE


def system_banner():
    logger.info("=" * 80)
    logger.info("🧠 FINANCIAL ORGANISM BOOTING")
    logger.info(f"MODE            : {MODE}")
    logger.info("PRIORITY        : Survival > Prediction > Profit")
    logger.info("=" * 80)


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():

    system_banner()

    # --- Core Components ---
    strategy_ensemble = StrategyEnsemble()
    allocator = CapitalAllocator()
    capital_flight = CapitalFlightDetector()

    risk_manager = RiskManager()
    black_swan = BlackSwanHedger()

    crisis_simulator = CrisisSimulator()
    genetic_engine = GeneticEngine()

    executor = ExecutionEngine(mode=MODE)
    ai_committee = AIRiskCommittee()

    # --- Safety Gate ---
    if MODE == "LIVE":
        ai_committee.assert_go_live_ready()

    logger.info("System initialized successfully.")

    # -----------------------------
    # LIVE LOOP
    # -----------------------------
    while True:
        try:
            # 1️⃣ Fetch Market Data
            market_data = fetch_market_data()

            # 2️⃣ Run Crisis Injection (synthetic + historical)
            crisis_signals = crisis_simulator.evaluate(market_data)

            # 3️⃣ Strategy Decisions
            strategy_outputs = strategy_ensemble.generate_signals(
                market_data=market_data,
                crisis_signals=crisis_signals
            )

            # 4️⃣ Capital Allocation
            allocations = allocator.allocate(strategy_outputs)

            # 5️⃣ Capital Flight Detection
            if capital_flight.detect(market_data):
                logger.warning("🚨 Capital flight detected — reducing exposure")
                allocations = allocator.defensive_mode(allocations)

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
            executor.execute(allocations, hedge_orders)

            # 9️⃣ Evolution Pressure (Strategy Darwinism)
            genetic_engine.evolve(
                performance=strategy_ensemble.performance_metrics(),
                crisis_feedback=crisis_signals
            )

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
    main()
