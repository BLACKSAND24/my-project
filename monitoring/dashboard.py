from .analytics import report

def start_dashboard(metrics: dict) -> str:
		"""Return a one-line monitoring summary for quick display.

		Example:
			metrics = {"rolling_avg_ee":100.0, "window":10, ...}
		"""
		return report(metrics)
