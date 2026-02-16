from .logger import log_ee_metrics

def report(metrics: dict) -> str:
	"""Record EE metrics and return a one-line summary.

	Expected keys in metrics dict (best-effort):
	  - rolling_avg_ee (float, percent)
	  - window (int)
	  - ee_min (float)
	  - hysteresis (float)
	  - per_strategy (dict)
	  - latency_min_ms (float)
	  - latency_max_ms (float)
	  - capital_util_min_pct (float)
	  - capital_util_max_pct (float)
	  - max_dd (float)
	"""
	# Defaults
	ra = float(metrics.get("rolling_avg_ee", 0.0))
	window = int(metrics.get("window", 10))
	ee_min = float(metrics.get("ee_min", 97.5))
	hysteresis = float(metrics.get("hysteresis", 0.3))
	per_strategy = metrics.get("per_strategy", {})
	latency_min = float(metrics.get("latency_min_ms", 0.0))
	latency_max = float(metrics.get("latency_max_ms", 0.0))
	cap_min = float(metrics.get("capital_util_min_pct", 0.0))
	cap_max = float(metrics.get("capital_util_max_pct", 0.0))
	max_dd = float(metrics.get("max_dd", 0.0))

	# persist
	try:
		log_ee_metrics(ra, window, ee_min, hysteresis, per_strategy, latency_min, latency_max, cap_min, cap_max, max_dd)
	except Exception:
		# logging should not raise for callers
		pass

	# build concise one-line summary
	summary = (
		f"EE={ra:.2f}% (w={window}) | Latency={latency_min:.0f}-{latency_max:.0f}ms | "
		f"Capital={cap_min:.0f}-{cap_max:.0f}% | max_dd={max_dd:.2f}%"
	)
	return summary
