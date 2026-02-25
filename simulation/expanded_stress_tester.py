"""Expanded Stress Tester with Historical Crisis Replay.

This module extends the stress testing capabilities with:
- Historical crisis scenarios (2008, COVID, Flash Crash, etc.)
- CrisisReplay class for historical data replay
- Custom scenario injection
- Visualization of stress test results

This provides more comprehensive risk analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
import json
import os

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("EXPANDED_STRESS_TESTER")


# Historical crisis scenarios
CRISIS_SCENARIOS = {
    "2008_financial_crisis": {
        "name": "2008 Financial Crisis",
        "description": "Lehman Brothers collapse and global financial crisis",
        "duration_days": 365,
        "max_drawdown": -0.50,
        "recovery_months": 18,
        "returns": {
            "BTC": -0.70,  # Crypto didn't exist, but simulating similar asset
            "ETH": -0.75,
        },
        "volatility_multiplier": 3.0,
        "liquidity_shock": 0.5,  # 50% liquidity reduction
    },
    "covid_crash_2020": {
        "name": "COVID-19 Crash (March 2020)",
        "description": "Global pandemic market crash and quick recovery",
        "duration_days": 60,
        "max_drawdown": -0.35,
        "recovery_months": 3,
        "returns": {
            "BTC": -0.50,
            "ETH": -0.55,
        },
        "volatility_multiplier": 4.0,
        "liquidity_shock": 0.3,
    },
    "flash_crash_2010": {
        "name": "Flash Crash (May 2010)",
        "description": "High-frequency trading triggered crash",
        "duration_days": 5,
        "max_drawdown": -0.20,
        "recovery_months": 1,
        "returns": {
            "BTC": -0.25,
            "ETH": -0.30,
        },
        "volatility_multiplier": 5.0,
        "liquidity_shock": 0.7,
    },
    "rate_shock_2022": {
        "name": "Rate Shock (2022)",
        "description": "Aggressive Fed rate hikes",
        "duration_days": 180,
        "max_drawdown": -0.25,
        "recovery_months": 6,
        "returns": {
            "BTC": -0.35,
            "ETH": -0.40,
        },
        "volatility_multiplier": 2.5,
        "liquidity_shock": 0.4,
    },
    "exchange_hack": {
        "name": "Major Exchange Hack",
        "description": "Large exchange compromise (e.g., Mt. Gox)",
        "duration_days": 30,
        "max_drawdown": -0.40,
        "recovery_months": 12,
        "returns": {
            "BTC": -0.60,
            "ETH": -0.65,
        },
        "volatility_multiplier": 4.5,
        "liquidity_shock": 0.8,
    },
    "regulatory_ban": {
        "name": "Regulatory Ban",
        "description": "Major country bans cryptocurrency",
        "duration_days": 90,
        "max_drawdown": -0.45,
        "recovery_months": 9,
        "returns": {
            "BTC": -0.55,
            "ETH": -0.60,
        },
        "volatility_multiplier": 3.5,
        "liquidity_shock": 0.6,
    },
    "custom": {
        "name": "Custom Scenario",
        "description": "User-defined custom scenario",
        "duration_days": 30,
        "max_drawdown": -0.30,
        "recovery_months": 6,
        "returns": {
            "BTC": -0.30,
            "ETH": -0.35,
        },
        "volatility_multiplier": 2.0,
        "liquidity_shock": 0.4,
    }
}


class CrisisReplay:
    """Replays historical crisis scenarios for stress testing."""
    
    def __init__(self, scenario_library: Dict = None):
        """Initialize crisis replay.
        
        Args:
            scenario_library: Dict of crisis scenarios
        """
        self.scenario_library = scenario_library or CRISIS_SCENARIOS
        self.current_scenario = None
        self.current_day = 0
    
    def load_scenario(self, scenario_name: str) -> bool:
        """Load a crisis scenario.
        
        Args:
            scenario_name: Name of scenario to load
            
        Returns:
            True if loaded successfully
        """
        if scenario_name not in self.scenario_library:
            logger.error(f"Scenario '{scenario_name}' not found")
            return False
        
        self.current_scenario = self.scenario_library[scenario_name]
        self.current_day = 0
        logger.info(f"Loaded crisis scenario: {self.current_scenario['name']}")
        return True
    
    def get_current_returns(self) -> Dict[str, float]:
        """Get returns for current day in scenario.
        
        Returns:
            Dict of symbol -> return for current day
        """
        if not self.current_scenario:
            return {}
        
        # Generate daily returns based on scenario
        # This is simplified - in production would use actual historical data
        duration = self.current_scenario["duration_days"]
        returns = self.current_scenario["returns"]
        
        # Calculate daily return (simplified)
        daily_returns = {}
        for symbol, total_return in returns.items():
            # Add some noise based on volatility multiplier
            vol_mult = self.current_scenario["volatility_multiplier"]
            daily_vol = 0.02 * vol_mult  # 2% base daily vol
            
            # Calculate position in scenario
            progress = self.current_day / max(duration, 1)
            
            # Drawdown is typically worse in the middle of crisis
            if progress < 0.3:
                shock_factor = progress / 0.3
            elif progress < 0.7:
                shock_factor = 1.0
            else:
                shock_factor = 1.0 - (progress - 0.7) / 0.3
            
            daily_return = (total_return / duration) * shock_factor + np.random.normal(0, daily_vol)
            daily_returns[symbol] = daily_return
        
        return daily_returns
    
    def step(self) -> bool:
        """Advance one day in the scenario.
        
        Returns:
            True if scenario still active, False if complete
        """
        if not self.current_scenario:
            return False
        
        self.current_day += 1
        duration = self.current_scenario["duration_days"]
        
        return self.current_day < duration
    
    def reset(self):
        """Reset to beginning of scenario."""
        self.current_day = 0
    
    def get_scenario_info(self) -> Dict:
        """Get information about current scenario."""
        if not self.current_scenario:
            return {}
        
        return {
            "name": self.current_scenario["name"],
            "description": self.current_scenario["description"],
            "day": self.current_day,
            "total_days": self.current_scenario["duration_days"],
            "progress": self.current_day / max(self.current_scenario["duration_days"], 1)
        }
    
    def list_scenarios(self) -> List[str]:
        """List available scenario names."""
        return list(self.scenario_library.keys())


class ExpandedStressTester:
    """Expanded stress tester with historical scenarios and custom injection."""
    
    def __init__(self):
        """Initialize expanded stress tester."""
        self.crisis_replay = CrisisReplay()
        self.custom_scenarios = {}
        self.test_results = []
        
        # Load default scenarios
        self.scenarios = CRISIS_SCENARIOS.copy()
    
    def add_custom_scenario(self, name: str, scenario: Dict):
        """Add a custom scenario.
        
        Args:
            name: Scenario name
            scenario: Scenario definition dict
        """
        self.scenarios[name] = scenario
        self.custom_scenarios[name] = scenario
        logger.info(f"Added custom scenario: {name}")
    
    def run_scenario(self, 
                    scenario_name: str,
                    weights: Dict[str, float],
                    initial_capital: float = 10000.0) -> Dict:
        """Run a stress test scenario.
        
        Args:
            scenario_name: Name of scenario to run
            weights: Portfolio weights
            initial_capital: Starting capital
            
        Returns:
            Dict with test results
        """
        # Load scenario
        if not self.crisis_replay.load_scenario(scenario_name):
            return {"error": f"Scenario '{scenario_name}' not found"}
        
        scenario = self.crisis_replay.current_scenario
        
        # Run simulation
        equity = initial_capital
        equity_curve = [equity]
        max_drawdown = 0.0
        peak = equity
        
        daily_returns_list = []
        
        while self.crisis_replay.step():
            # Get returns for this day
            returns = self.crisis_replay.get_current_returns()
            
            # Calculate portfolio return
            port_return = 0.0
            for symbol, weight in weights.items():
                r = returns.get(symbol, 0.0)
                port_return += weight * r
            
            daily_returns_list.append(port_return)
            
            # Update equity
            equity *= (1.0 + port_return)
            equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
        
        # Calculate final metrics
        final_return = (equity - initial_capital) / initial_capital
        final_value = equity
        
        # Recovery time estimate
        recovery_months = scenario.get("recovery_months", 6)
        
        # Build result
        result = {
            "scenario_name": scenario_name,
            "scenario_description": scenario.get("description", ""),
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": final_return,
            "max_drawdown": -max_drawdown,
            "recovery_months": recovery_months,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns_list,
            "volatility_multiplier": scenario.get("volatility_multiplier", 1.0),
            "liquidity_shock": scenario.get("liquidity_shock", 0.0),
            "status": "completed"
        }
        
        self.test_results.append(result)
        
        logger.info(f"Stress test completed: {scenario_name}, "
                   f"return={final_return:.2%}, max_dd={max_drawdown:.2%}")
        
        return result
    
    def run_all_scenarios(self, 
                         weights: Dict[str, float],
                         initial_capital: float = 10000.0) -> List[Dict]:
        """Run all available scenarios.
        
        Args:
            weights: Portfolio weights
            initial_capital: Starting capital
            
        Returns:
            List of results for each scenario
        """
        results = []
        
        for scenario_name in self.scenarios.keys():
            result = self.run_scenario(scenario_name, weights, initial_capital)
            results.append(result)
        
        return results
    
    def compare_scenarios(self, results: List[Dict] = None) -> pd.DataFrame:
        """Compare results across scenarios.
        
        Args:
            results: List of scenario results (uses stored results if None)
            
        Returns:
            DataFrame comparing scenarios
        """
        results = results or self.test_results
        
        if not results:
            return None
        
        comparison = []
        for r in results:
            comparison.append({
                "Scenario": r.get("scenario_name", "Unknown"),
                "Description": r.get("scenario_description", ""),
                "Final Value": r.get("final_value", 0),
                "Total Return": r.get("total_return", 0),
                "Max Drawdown": r.get("max_drawdown", 0),
                "Recovery (months)": r.get("recovery_months", 0),
            })
        
        return comparison
    
    def get_worst_case(self, results: List[Dict] = None) -> Dict:
        """Get worst case scenario from results.
        
        Args:
            results: List of scenario results (uses stored results if None)
            
        Returns:
            Worst case scenario result
        """
        results = results or self.test_results
        
        if not results:
            return {}
        
        # Find scenario with worst max drawdown
        worst = min(results, key=lambda x: x.get("max_drawdown", 0))
        
        return worst
    
    def get_average_stress(self, results: List[Dict] = None) -> Dict:
        """Get average metrics across scenarios.
        
        Args:
            results: List of scenario results (uses stored results if None)
            
        Returns:
            Dict of average metrics
        """
        results = results or self.test_results
        
        if not results:
            return {}
        
        avg_return = np.mean([r.get("total_return", 0) for r in results])
        avg_dd = np.mean([abs(r.get("max_drawdown", 0)) for r in results])
        
        return {
            "average_return": avg_return,
            "average_max_drawdown": avg_dd,
            "scenarios_tested": len(results)
        }
    
    def create_custom_scenario(self,
                              name: str,
                              description: str,
                              max_drawdown: float,
                              duration_days: int,
                              returns: Dict[str, float],
                              volatility_multiplier: float = 2.0,
                              liquidity_shock: float = 0.4):
        """Create a custom scenario.
        
        Args:
            name: Scenario name
            description: Description
            max_drawdown: Expected max drawdown (e.g., -0.30)
            duration_days: Duration in days
            returns: Expected total returns by symbol
            volatility_multiplier: Volatility multiplier
            liquidity_shock: Liquidity shock factor (0-1)
        """
        scenario = {
            "name": name,
            "description": description,
            "max_drawdown": max_drawdown,
            "duration_days": duration_days,
            "returns": returns,
            "volatility_multiplier": volatility_multiplier,
            "liquidity_shock": liquidity_shock,
        }
        
        # Estimate recovery months based on drawdown
        if abs(max_drawdown) < 0.15:
            scenario["recovery_months"] = 1
        elif abs(max_drawdown) < 0.30:
            scenario["recovery_months"] = 3
        elif abs(max_drawdown) < 0.50:
            scenario["recovery_months"] = 6
        else:
            scenario["recovery_months"] = 12
        
        self.add_custom_scenario(name, scenario)
        
        return scenario
    
    def export_results(self, filepath: str):
        """Export test results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Exported stress test results to {filepath}")
    
    def import_results(self, filepath: str):
        """Import test results from JSON file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.test_results = json.load(f)
        logger.info(f"Imported {len(self.test_results)} results from {filepath}")


def create_expanded_stress_tester() -> ExpandedStressTester:
    """Factory function to create expanded stress tester.
    
    Returns:
        ExpandedStressTester instance
    """
    return ExpandedStressTester()


# Backwards compatibility - keep original functions working
def apply_weights_to_returns(weights: Dict[str, float], returns: Dict[str, List[float]]) -> List[float]:
    """Generate portfolio return series (from original stress_tester)."""
    if not weights or not returns:
        return []
    length = max(len(r) for r in returns.values())
    port = []
    for i in range(length):
        r = 0.0
        for strat, w in weights.items():
            series = returns.get(strat, [])
            if i < len(series):
                r += w * series[i]
        port.append(r)
    return port


def compute_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown (from original stress_tester)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    maxdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    return maxdd


def run_scenarios(weights: Dict[str, float], return_sets: List[Dict[str, List[float]]]) -> List[Dict]:
    """Run multiple return scenarios (from original stress_tester)."""
    results = []
    for idx, returns in enumerate(return_sets):
        port = apply_weights_to_returns(weights, returns)
        equity = []
        acc = 1.0
        for r in port:
            acc *= (1.0 + r)
            equity.append(acc)
        results.append({
            'scenario': idx,
            'final_return': equity[-1] if equity else 1.0,
            'max_drawdown': compute_drawdown(equity)
        })
    return results
