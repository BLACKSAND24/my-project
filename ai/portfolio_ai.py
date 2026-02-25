"""Portfolio-level AI allocator.

Provides a lightweight "institutional desk" that looks at strategy scores
and cross-strategy correlations (via the global risk brain) to produce a
set of portfolio weights.  The intent is to demonstrate a simple
"AI" layer that sits above individual strategies, punishing high
correlation and favoring those with strong raw signals.

This component is deliberately small and dependency-free apart from NumPy
(which is already required elsewhere).  In a real system you could replace
it with an ML model, reinforcement learner, or even a human-in-the-loop
system; the interface remains the same.
"""

from typing import Dict
import numpy as np


class PortfolioAI:
    def __init__(self, risk_brain):
        """Create with a reference to the global risk controller.

        Args:
            risk_brain (GlobalRiskController): used to compute correlations.
        """
        self.risk_brain = risk_brain

    def get_weights(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Return normalized portfolio weights for each strategy.

        Args:
            raw_scores: output of StrategyEnsemble.generate_signals

        Returns:
            dict mapping strategy name to weight (sums to 1.0, or zeros).
        """
        if not raw_scores:
            return {}

        # compute average correlation for each strategy from the global brain
        corr_info = self.risk_brain.correlation_matrix()
        names = corr_info.get("names", [])
        mat = corr_info.get("matrix")
        avg_corr = {}
        if mat is not None and mat.size > 1:
            for i, name in enumerate(names):
                coeffs = [abs(v) for j, v in enumerate(mat[i]) if j != i]
                avg_corr[name] = sum(coeffs) / len(coeffs) if coeffs else 0.0
        else:
            for name in raw_scores:
                avg_corr[name] = 0.0

        # adjust raw scores by penalizing high correlation
        adjusted = {}
        for name, score in raw_scores.items():
            penalty = avg_corr.get(name, 0.0)
            adjusted[name] = float(score) * (1.0 - penalty)

        total = sum(adjusted.values())
        if total <= 0:
            return {k: 0.0 for k in adjusted}
        return {k: v / total for k, v in adjusted.items()}
