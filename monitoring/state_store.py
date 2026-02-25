"""
State Store for Financial Organism.

This module provides:
- Runtime state persistence
- JSON-based state files for monitoring
- State snapshots for the main loop

The state store enables the Streamlit dashboard to display real-time
regime, weights, risk exposures, and execution metrics.
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("STATE_STORE")


class StateStore:
    """Manages runtime state persistence for monitoring."""
    
    def __init__(self, state_dir: str = "financial_organism/logs"):
        """Initialize state store.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, "runtime_state.json")
        self.history_file = os.path.join(state_dir, "state_history.json")
        self.max_history = 1000  # Keep last 1000 state snapshots
        
        # Ensure directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # Current state
        self.current_state = self._load_state()
        self.state_history = self._load_history()
        
        logger.info(f"StateStore initialized: {self.state_file}")
    
    def _load_state(self) -> Dict:
        """Load current state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
        return self._get_default_state()
    
    def _load_history(self) -> List[Dict]:
        """Load state history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
        return []
    
    def _get_default_state(self) -> Dict:
        """Get default state."""
        return {
            "mode": CONFIG.get("MODE", "PAPER"),
            "timestamp": datetime.now().isoformat(),
            "cycle": 0,
            "regime": "normal",
            "weights": {},
            "allocations": {},
            "risk_metrics": {
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "correlation": 0.0
            },
            "execution_metrics": {
                "execution_efficiency": 1.0,
                "latency_ms": 0.0,
                "capital_utilization": 0.0
            },
            "governance": {
                "in_cool_off": False,
                "position_limits_ok": True
            },
            "ml_diagnostics": {},
            "errors": []
        }
    
    def update(self, 
               regime: str = None,
               weights: Dict[str, float] = None,
               allocations: Dict[str, float] = None,
               risk_metrics: Dict = None,
               execution_metrics: Dict = None,
               governance: Dict = None,
               ml_diagnostics: Dict = None,
               cycle: int = None,
               errors: List[str] = None):
        """Update state with new values.
        
        Args:
            regime: Current market regime
            weights: Portfolio weights
            allocations: Current allocations
            risk_metrics: Risk metrics
            execution_metrics: Execution metrics
            governance: Governance state
            ml_diagnostics: ML policy diagnostics
            cycle: Current cycle number
            errors: List of errors
        """
        if regime is not None:
            self.current_state["regime"] = regime
        if weights is not None:
            self.current_state["weights"] = weights
        if allocations is not None:
            self.current_state["allocations"] = allocations
        if risk_metrics is not None:
            self.current_state["risk_metrics"].update(risk_metrics)
        if execution_metrics is not None:
            self.current_state["execution_metrics"].update(execution_metrics)
        if governance is not None:
            self.current_state["governance"].update(governance)
        if ml_diagnostics is not None:
            self.current_state["ml_diagnostics"] = ml_diagnostics
        if cycle is not None:
            self.current_state["cycle"] = cycle
        if errors is not None:
            self.current_state["errors"] = errors
        
        # Always update timestamp
        self.current_state["timestamp"] = datetime.now().isoformat()
    
    def save(self):
        """Save current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)
            
            # Add to history
            self.state_history.append(self.current_state.copy())
            
            # Trim history if needed
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history:]
            
            # Save history
            with open(self.history_file, 'w') as f:
                json.dump(self.state_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_state(self) -> Dict:
        """Get current state."""
        return self.current_state.copy()
    
    def get_history(self, limit: int = None) -> List[Dict]:
        """Get state history.
        
        Args:
            limit: Optional limit on number of history items
            
        Returns:
            List of historical states
        """
        if limit:
            return self.state_history[-limit:]
        return self.state_history.copy()
    
    def get_regime_history(self, limit: int = 100) -> List[str]:
        """Get regime history.
        
        Args:
            limit: Number of items to return
            
        Returns:
            List of regime values
        """
        history = self.get_history(limit)
        return [s.get("regime", "unknown") for s in history]
    
    def get_weights_history(self, limit: int = 100) -> List[Dict]:
        """Get weights history.
        
        Args:
            limit: Number of items to return
            
        Returns:
            List of weights dictionaries
        """
        history = self.get_history(limit)
        return [s.get("weights", {}) for s in history]
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of recent metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        if not self.state_history:
            return {}
        
        recent = self.state_history[-100:]
        
        # Calculate averages
        regimes = [s.get("regime", "unknown") for s in recent]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        execution_efficiencies = [
            s.get("execution_metrics", {}).get("execution_efficiency", 1.0)
            for s in recent
        ]
        avg_ee = sum(execution_efficiencies) / len(execution_efficiencies) if execution_efficiencies else 1.0
        
        capital_utils = [
            s.get("execution_metrics", {}).get("capital_utilization", 0.0)
            for s in recent
        ]
        avg_cap_util = sum(capital_utils) / len(capital_utils) if capital_utils else 0.0
        
        return {
            "regime_distribution": regime_counts,
            "avg_execution_efficiency": avg_ee,
            "avg_capital_utilization": avg_cap_util,
            "total_cycles": len(self.state_history)
        }
    
    def clear_history(self):
        """Clear state history."""
        self.state_history = []
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
        except Exception as e:
            logger.error(f"Could not clear history: {e}")


def create_state_store(config: dict = None) -> StateStore:
    """Factory function to create state store.
    
    Args:
        config: Optional config dict
        
    Returns:
        StateStore instance
    """
    config = config or {}
    state_dir = config.get("STATE_DIR", "financial_organism/logs")
    return StateStore(state_dir=state_dir)


# ============================================================
# Global state store instance
# ============================================================

_global_state_store = None


def get_state_store() -> StateStore:
    """Get global state store instance.
    
    Returns:
        StateStore instance
    """
    global _global_state_store
    if _global_state_store is None:
        _global_state_store = create_state_store()
    return _global_state_store


def update_runtime_state(**kwargs):
    """Quick function to update runtime state.
    
    Args:
        **kwargs: Key-value pairs to update
    """
    store = get_state_store()
    store.update(**kwargs)
    store.save()


def save_runtime_state():
    """Save current runtime state."""
    store = get_state_store()
    store.save()
