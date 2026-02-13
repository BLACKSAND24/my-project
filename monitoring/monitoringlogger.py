import csv
import os
from datetime import datetime

LOG_DIR = "logs"
CSV_LOG = os.path.join(LOG_DIR, "paper_trades.csv")
TXT_LOG = os.path.join(LOG_DIR, "paper_trades.txt")

os.makedirs(LOG_DIR, exist_ok=True)

# Initialize CSV if missing
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "trend_direction",
            "trend_exposure",
            "mean_direction",
            "mean_exposure",
            "hedge_used",
            "capital_flight"
        ])

def log_event(trend_dir, trend_exp, mean_dir, mean_exp, hedge, capital_flight):
    ts = datetime.now().isoformat(timespec="seconds")

    # CSV
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts, trend_dir, trend_exp,
            mean_dir, mean_exp,
            hedge, capital_flight
        ])

    # TXT (human readable)
    with open(TXT_LOG, "a") as f:
        f.write(
            f"[{ts}] "
            f"Trend={trend_dir}:{trend_exp:.4f} | "
            f"Mean={mean_dir}:{mean_exp:.4f} | "
            f"Hedge={hedge} | "
            f"CapitalFlight={capital_flight}\n"
        )
