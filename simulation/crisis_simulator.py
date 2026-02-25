"""Crisis Simulator Module.

This module provides crisis simulation capabilities including:
- Historical crisis templates (GFC, COVID, inflation)
- Return path modification during evaluation
- Replay functionality for testing
"""
import math
import statistics
import random
from typing import Dict, List, Optional

from financial_organism.config import CONFIG


# Historical crisis templates
CRISIS_TEMPLATES = {
    "GFC": {
        "name": "Global Financial Crisis (2008)",
        "description": "Severe market crash with prolonged recovery",
        "returns": [-0.05] * 5 + [-0.10] * 3 + [-0.15] * 2 + [-0.08] * 5 + [0.02] * 10 + [0.03] * 5,
        "volatility_mult": 3.0,
    },
    "COVID": {
        "name": "COVID-19 Crash (2020)",
        "description": "Sharp V-shaped recovery",
        "returns": [-0.05, -0.07, -0.10, -0.12, -0.08, 0.05, 0.08, 0.10, 0.06, 0.04],
        "volatility_mult": 2.5,
    },
    "INFLATION": {
        "name": "Inflation Shock (2022)",
        "description": "Rising rates causing multiple sell-offs",
        "returns": [-0.02, -0.03, -0.01, -0.04, -0.02, 0.01, -0.02, -0.03, -0.02, -0.01, 0.02, -0.01],
        "volatility_mult": 2.0,
    },
}


class CrisisSimulator:
    def __init__(self, vol_threshold=None, drawdown_threshold=None):
        self.volatility_threshold = float(vol_threshold if vol_threshold is not None else CONFIG.get("CRISIS_VOL_THRESHOLD", 0.35))
        self.drawdown_threshold = float(drawdown_threshold if drawdown_threshold is not None else CONFIG.get("CRISIS_DRAWDOWN_THRESHOLD", -0.20))
        self.crisis_templates = CRISIS_TEMPLATES
        self.replay_data = []

    def evaluate(self, market_data, apply_crisis_modifier=False, crisis_template=None):
        returns = [float(x) for x in market_data.get("returns", [])]
        if not returns:
            return {"crisis": False, "severity": 0.0, "action": None, "volatility": 0.0, "recent_return": 0.0, "template": None}
        
        # Apply crisis template modifier if requested
        if apply_crisis_modifier and crisis_template and crisis_template in self.crisis_templates:
            template = self.crisis_templates[crisis_template]
            template_returns = template.get("returns", [])
            if template_returns:
                returns = template_returns[:len(returns)]
        
        volatility = statistics.pstdev(returns) * math.sqrt(252)
        cumulative, drawdown = 0.0, 0.0
        for v in returns:
            cumulative += v
            if cumulative < drawdown:
                drawdown = cumulative
        recent_return = returns[-1]
        crisis = volatility > self.volatility_threshold or drawdown < self.drawdown_threshold or recent_return < -0.05
        severity = min(1.0, max(
            volatility / max(self.volatility_threshold, 1e-6),
            abs(drawdown / min(self.drawdown_threshold, -1e-6)),
            abs(recent_return / -0.05),
        ))
        return {"crisis": crisis, "severity": round(severity, 2), "action": "REDUCE" if crisis else None, "volatility": volatility, "recent_return": recent_return, "template": crisis_template if apply_crisis_modifier else None}

    def get_crisis_template(self, template_name):
        return self.crisis_templates.get(template_name)
    
    def list_crisis_templates(self):
        return list(self.crisis_templates.keys())
    
    def add_replay_data(self, market_data):
        self.replay_data.append(market_data)
    
    def get_replay_data(self, limit=None):
        if limit:
            return self.replay_data[-limit:]
        return self.replay_data
    
    def clear_replay_data(self):
        self.replay_data = []
    
    def run_stress_test(self, scenario="RANDOM", num_periods=20):
        if scenario == "RANDOM":
            returns = [random.gauss(-0.001, 0.03) for _ in range(num_periods)]
            template_name = None
        elif scenario in self.crisis_templates:
            template = self.crisis_templates[scenario]
            returns = template.get("returns", [0.0] * num_periods)
            template_name = scenario
        else:
            returns = [random.gauss(-0.001, 0.03) for _ in range(num_periods)]
            template_name = None
        
        market_data = {"returns": returns}
        result = self.evaluate(market_data, apply_crisis_modifier=(template_name is not None), crisis_template=template_name)
        
        if template_name:
            template = self.crisis_templates.get(template_name, {})
            result["template_name"] = template.get("name", template_name)
        
        return result
