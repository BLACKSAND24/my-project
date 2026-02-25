"""Simple stress-testing engine for portfolios.

Allows you to feed historical return scenarios or ad-hoc shocks and
see how a set of weights performs, computing peak drawdown and final
equity factor.  This is the beginning of a full "what-if" simulation
platform.
"""
from typing import Dict, List, Iterable


def apply_weights_to_returns(weights: Dict[str, float], returns: Dict[str, List[float]]) -> List[float]:
    """Generate portfolio return series given per-strategy returns.

    Args:
        weights: strategy -> weight (sums to 1.0)
        returns: strategy -> list of returns (same length for all)

    Returns:
        list of portfolio returns (weighted sum)
    """
    if not weights or not returns:
        return []
    # assume all return series same length
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
    """Compute maximum drawdown given equity series (cumulative returns)."""
    peak = equity_curve[0] if equity_curve else 0.0
    maxdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    return maxdd


def run_scenarios(weights: Dict[str, float], return_sets: Iterable[Dict[str, List[float]]]) -> List[Dict]:
    """Run multiple return scenarios.

    Args:
        weights: portfolio weights
        return_sets: iterable of return dicts (one scenario per item)

    Returns:
        list of dicts with keys 'scenario', 'final_return', 'max_drawdown'
    """
    results = []
    for idx, returns in enumerate(return_sets):
        port = apply_weights_to_returns(weights, returns)
        # compute cumulative equity assuming starting capital 1.0
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
