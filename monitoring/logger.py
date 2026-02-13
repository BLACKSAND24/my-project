from pathlib import Path
import csv
from datetime import datetime, timezone

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_CSV_PATH = _LOG_DIR / "paper_trades.csv"
_TXT_PATH = _LOG_DIR / "paper_trades.txt"
_HEALTH_PATH = _LOG_DIR / "system_health.csv"
_ERROR_PATH = _LOG_DIR / "system_errors.txt"

def _ensure_files():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with _CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","trend_direction","trend_exposure","mean_direction","mean_exposure","hedge","capital_flight","offline_mode"])
    if not _HEALTH_PATH.exists():
        with _HEALTH_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","loop_count","last_success_ts","mode"])

def initialize_logs():
    _ensure_files()
    _TXT_PATH.touch(exist_ok=True)
    _ERROR_PATH.touch(exist_ok=True)

def log_event(trend_dir, trend_exp, mean_dir, mean_exp, hedge, capital_flight, offline_mode):
    _ensure_files()
    ts = datetime.now(timezone.utc).isoformat()
    with _CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, trend_dir, trend_exp, mean_dir, mean_exp, int(bool(hedge)), int(bool(capital_flight)), int(bool(offline_mode))])

def log_health(loop_count, last_success_ts, mode):
    _ensure_files()
    ts = datetime.now(timezone.utc).isoformat()
    with _HEALTH_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, loop_count, last_success_ts, mode])

def log_error(msg):
    _ensure_files()
    ts = datetime.now(timezone.utc).isoformat()
    with _ERROR_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")
