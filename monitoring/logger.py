from pathlib import Path
import csv
from datetime import datetime, timezone
import json

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_CSV_PATH = _LOG_DIR / "paper_trades.csv"
_TXT_PATH = _LOG_DIR / "paper_trades.txt"
_HEALTH_PATH = _LOG_DIR / "system_health.csv"
_EE_METRICS_PATH = _LOG_DIR / "ee_metrics.csv"
_ERROR_PATH = _LOG_DIR / "system_errors.txt"

def _ensure_files():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with _CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","trend_direction","trend_exposure","mean_direction","mean_exposure","hedge","capital_flight","offline_mode"])
    if not _HEALTH_PATH.exists():
        with _HEALTH_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp","loop_count","last_success_ts","mode"])
    if not _EE_METRICS_PATH.exists():
        with _EE_METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp",
                "rolling_avg_ee",
                "window",
                "ee_min",
                "hysteresis",
                "per_strategy",  # JSON dict
                "latency_min_ms",
                "latency_max_ms",
                "capital_util_min_pct",
                "capital_util_max_pct",
                "max_dd"
            ])

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

def log_ee_metrics(rolling_avg_ee, window, ee_min, hysteresis, per_strategy, latency_min_ms, latency_max_ms, capital_util_min_pct, capital_util_max_pct, max_dd):
    """Append a row to ee_metrics.csv capturing execution efficiency and related stats.

    per_strategy: dict of strategy_name -> efficiency (0-100)
    """
    _ensure_files()
    ts = datetime.now(timezone.utc).isoformat()
    # ensure per_strategy is JSON-serializable
    try:
        per_strategy_json = json.dumps(per_strategy, ensure_ascii=False)
    except Exception:
        per_strategy_json = str(per_strategy)
    with _EE_METRICS_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            ts,
            float(rolling_avg_ee),
            int(window),
            float(ee_min),
            float(hysteresis),
            per_strategy_json,
            float(latency_min_ms),
            float(latency_max_ms),
            float(capital_util_min_pct),
            float(capital_util_max_pct),
            float(max_dd)
        ])

def log_error(msg):
    _ensure_files()
    ts = datetime.now(timezone.utc).isoformat()
    with _ERROR_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")
