import os
import csv
import time
from financial_organism.utils.logger import get_logger

LOGGER = get_logger("EE_MONITOR")


class EEMonitor:
    def __init__(self, csv_path=None):
        # Default CSV file at repo root
        self.csv_path = csv_path or os.path.join(os.getcwd(), "ee_metrics.csv")
        # Ensure header exists
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "strategy",
                        "requested",
                        "filled",
                        "fill_eff",
                        "latency_ms",
                        "capital_util",
                        "readiness"
                    ])
            except Exception:
                LOGGER.exception("Failed to create EE CSV header")

    def record(self, strategy, requested, filled, latency_ms, capital_util, readiness):
        try:
            timestamp = time.time()
            fill_eff = (filled / requested) if requested and requested != 0 else 1.0
            # Append CSV row
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    strategy,
                    float(requested),
                    float(filled),
                    float(fill_eff),
                    float(latency_ms),
                    float(capital_util),
                    bool(readiness)
                ])

            # Friendly short log for live tailing
            LOGGER.info(f"[EE_MONITOR] {strategy} | eff={fill_eff:.2%} | latency={latency_ms:.1f}ms | cap={capital_util:.2%} | ready={readiness}")
        except Exception:
            LOGGER.exception("Failed to record EE metric")


# Singleton instance for easy import
EE_MONITOR = EEMonitor()
