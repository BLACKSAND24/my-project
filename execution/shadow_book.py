import os
import csv
import json
from datetime import datetime


class ShadowBook:
    def __init__(self):
        self.positions = {}
        self.order_history = []
        # Equity tracking & drawdown
        self.equity_curve = []
        self.peak_equity = float('nan')
        self.max_drawdown = 0.0
        # Regime replay mode: store execution regimes for research
        self.regime_history = []
        self._logs_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(self._logs_dir, exist_ok=True)
        self._paper_path = os.path.join(self._logs_dir, 'paper_execution_history.csv')
        self._regime_path = os.path.join(self._logs_dir, 'regime_replay.csv')
        # ensure CSV has header
        if not os.path.exists(self._paper_path):
            with open(self._paper_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(['timestamp', 'mode', 'allocations_json', 'hedges_json'])
        # ensure regime CSV has header
        if not os.path.exists(self._regime_path):
            with open(self._regime_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(['timestamp', 'mode', 'volatility', 'allocation_vector', 'hedges_json'])

    def apply_allocation(self, allocations):
        for symbol, notional in (allocations or {}).items():
            self.positions[symbol] = float(notional)
            self.order_history.append({"symbol": symbol, "notional": float(notional), "kind": "allocation"})

    def apply_hedges(self, hedge_orders):
        for hedge in (hedge_orders or []):
            self.order_history.append(dict(hedge))

    def update_equity(self, current_equity):
        """Track equity curve and compute drawdown."""
        self.equity_curve.append(float(current_equity))
        if len(self.equity_curve) == 1:
            self.peak_equity = float(current_equity)
        else:
            self.peak_equity = max(self.peak_equity, float(current_equity))
        # Drawdown = (current - peak) / peak
        if self.peak_equity > 0:
            dd = (float(current_equity) - self.peak_equity) / self.peak_equity
            self.max_drawdown = min(self.max_drawdown, dd)

    def get_metrics(self):
        """Return equity metrics: peak, current, max_drawdown."""
        current = self.equity_curve[-1] if self.equity_curve else 0.0
        return {
            "current_equity": current,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "regime_count": len(self.regime_history)
        }

    def record_regime(self, mode: str, allocations: dict, hedge_orders: list, volatility: float = 0.0):
        """Record execution regime for replay and research."""
        timestamp = datetime.utcnow().isoformat()
        alloc_vec = json.dumps(allocations or {})
        hedges_json = json.dumps(hedge_orders or [])
        regime = {"timestamp": timestamp, "mode": mode, "volatility": volatility, "allocations": allocations, "hedges": hedge_orders}
        self.regime_history.append(regime)
        try:
            if not os.path.exists(self._regime_path) or os.stat(self._regime_path).st_size == 0:
                with open(self._regime_path, 'w', newline='', encoding='utf-8') as fhw:
                    writer = csv.writer(fhw)
                    writer.writerow(['timestamp', 'mode', 'volatility', 'allocation_vector', 'hedges_json'])
            with open(self._regime_path, 'a', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow([timestamp, mode, volatility, alloc_vec, hedges_json])
        except Exception:
            pass

    def flatten(self):
        self.positions = {}
        self.order_history.append({"kind": "flatten"})

    def record_handoff(self, mode: str, allocations: dict, hedge_orders: list):
        """Record a handoff snapshot. Persist PAPER-mode handoffs to CSV."""
        timestamp = datetime.utcnow().isoformat()
        # append to CSV for PAPER mode
        try:
            # ensure header exists if file missing or empty
            if not os.path.exists(self._paper_path) or os.stat(self._paper_path).st_size == 0:
                with open(self._paper_path, 'w', newline='', encoding='utf-8') as fhw:
                    writer = csv.writer(fhw)
                    writer.writerow(['timestamp', 'mode', 'allocations_json', 'hedges_json'])
            with open(self._paper_path, 'a', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow([timestamp, mode, json.dumps(allocations or {}), json.dumps(hedge_orders or [])])
        except Exception:
            # best-effort, don't crash execution
            pass

    def get_handoff_snapshot(self, mode: str):
        """Return the most recent allocations for the given mode from the paper history CSV.
        For non-PAPER modes this returns the in-memory positions snapshot.
        """
        if mode == 'PAPER':
            try:
                with open(self._paper_path, 'r', newline='', encoding='utf-8') as fh:
                    rows = list(csv.reader(fh))
                    if len(rows) <= 1:
                        return {"timestamp": None, "allocations": {}, "hedges": []}
                    last = rows[-1]
                    # [timestamp, mode, allocations_json, hedges_json]
                    ts = last[0]
                    alloc = json.loads(last[2]) if last[2] else {}
                    hedges = json.loads(last[3]) if len(last) > 3 and last[3] else []
                    return {"timestamp": ts, "allocations": alloc, "hedges": hedges}
            except Exception:
                return {"timestamp": None, "allocations": {}, "hedges": []}
        else:
            return {"timestamp": None, "allocations": dict(self.positions), "hedges": []}

    def get_regime_history(self, limit: int = None):
        """Return regime history for replay or analysis."""
        if limit:
            return self.regime_history[-limit:]
        return self.regime_history
