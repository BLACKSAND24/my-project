"""Enhanced Governance Layer for Financial Organism.

This module provides enhanced governance capabilities including:
- Position-level limits
- Risk parity rebalancing
- Automated cool-off periods after drawdowns
- Governance enforcement with configurable rules

This extends the existing governance layer with more sophisticated controls.
"""
import time
from typing import Dict, List, Optional
from collections import deque

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("ENHANCED_GOVERNANCE")


class PositionLimits:
    """Enforces position-level limits on allocations."""
    
    def __init__(self, max_position_pct: float = 0.25, max_total_exposure: float = 0.85):
        """Initialize position limits.
        
        Args:
            max_position_pct: Maximum position as % of portfolio (default 25%)
            max_total_exposure: Maximum total exposure (default 85%)
        """
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
    
    def enforce(self, allocations: Dict[str, float], total_capital: float) -> Dict[str, float]:
        """Enforce position limits on allocations.
        
        Args:
            allocations: Dict of symbol -> allocation amount
            total_capital: Total capital available
            
        Returns:
            Dict with enforced limits
        """
        if not allocations:
            return {}
        
        # Check total exposure
        total_allocated = sum(allocations.values())
        if total_allocated > total_capital * self.max_total_exposure:
            scale = (total_capital * self.max_total_exposure) / total_allocated
            allocations = {k: v * scale for k, v in allocations.items()}
            logger.warning(f"Total exposure limited: scaled by {scale:.2f}")
        
        # Check individual position limits
        max_position = total_capital * self.max_position_pct
        for symbol, allocation in allocations.items():
            if allocation > max_position:
                logger.warning(f"Position limit hit for {symbol}: {allocation:.2f} > {max_position:.2f}")
                allocations[symbol] = max_position
        
        return allocations
    
    def validate(self, allocations: Dict[str, float], total_capital: float) -> tuple:
        """Validate allocations against limits.
        
        Returns:
            (is_valid, list of violation messages)
        """
        violations = []
        
        # Check total exposure
        total = sum(allocations.values())
        if total > total_capital * self.max_total_exposure:
            violations.append(f"Total exposure {total:.2f} exceeds limit {total_capital * self.max_total_exposure:.2f}")
        
        # Check individual positions
        max_position = total_capital * self.max_position_pct
        for symbol, allocation in allocations.items():
            if allocation > max_position:
                violations.append(f"Position {symbol} {allocation:.2f} exceeds limit {max_position:.2f}")
        
        return len(violations) == 0, violations


class RiskParityRebalancer:
    """Implements risk parity rebalancing for portfolio.
    
    Risk parity aims to allocate capital based on risk contribution
    rather than equal capital allocation.
    """
    
    def __init__(self, target_volatility: float = 0.15):
        """Initialize risk parity rebalancer.
        
        Args:
            target_volatility: Target portfolio volatility (default 15%)
        """
        self.target_volatility = target_volatility
        self.volatility_history = {}  # symbol -> deque of volatilities
    
    def update_volatility(self, symbol: str, volatility: float):
        """Update volatility estimate for a symbol.
        
        Args:
            symbol: Asset symbol
            volatility: Annualized volatility
        """
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=20)
        self.volatility_history[symbol].append(volatility)
    
    def get_volatility(self, symbol: str) -> float:
        """Get average volatility for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Average volatility or default 20%
        """
        if symbol not in self.volatility_history or not self.volatility_history[symbol]:
            return 0.20
        return sum(self.volatility_history[symbol]) / len(self.volatility_history[symbol])
    
    def calculate_weights(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk parity weights.
        
        Args:
            allocations: Current allocations
            
        Returns:
            New weights based on risk parity
        """
        if not allocations:
            return {}
        
        # Get volatilities
        vols = {s: self.get_volatility(s) for s in allocations.keys()}
        
        # Inverse volatility weights
        inv_vols = {s: 1.0 / max(v, 0.01) for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol == 0:
            # Fall back to equal weights
            equal_weight = 1.0 / len(allocations)
            return {s: equal_weight for s in allocations.keys()}
        
        # Normalize
        risk_parity_weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}
        
        # Scale to target volatility
        # This is a simplified version - would use actual portfolio vol calculation
        return risk_parity_weights
    
    def rebalance(self, current_weights: Dict[str, float], 
                  target_weights: Dict[str, float],
                  threshold: float = 0.05) -> Dict[str, float]:
        """Rebalance to target weights if deviation exceeds threshold.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights from risk parity
            threshold: Rebalance threshold (default 5%)
            
        Returns:
            New weights (either current or target based on threshold)
        """
        if not current_weights:
            return target_weights
        
        # Calculate max deviation
        max_deviation = 0.0
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)
        
        if max_deviation > threshold:
            logger.info(f"Rebalancing: max deviation {max_deviation:.2%} > threshold {threshold:.2%}")
            return target_weights
        else:
            logger.debug(f"No rebalancing needed: max deviation {max_deviation:.2%} <= threshold {threshold:.2%}")
            return current_weights


class DrawdownCoolOff:
    """Implements automated cool-off periods after drawdowns."""
    
    def __init__(self, 
                 drawdown_threshold: float = 0.10,
                 cool_off_duration_seconds: int = 300,
                 recovery_threshold: float = 0.05):
        """Initialize drawdown cool-off.
        
        Args:
            drawdown_threshold: Drawdown % to trigger cool-off (default 10%)
            cool_off_duration_seconds: Duration of cool-off (default 5 minutes)
            recovery_threshold: Recovery % to exit cool-off (default 5%)
        """
        self.drawdown_threshold = drawdown_threshold
        self.cool_off_duration = cool_off_duration_seconds
        self.recovery_threshold = recovery_threshold
        
        # State
        self.cool_off_start = None
        self.in_cool_off = False
        self.peak_equity = 0.0
        self.cool_off_count = 0
    
    def check(self, current_equity: float) -> bool:
        """Check if we should be in cool-off.
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            True if in cool-off period, False otherwise
        """
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            drawdown = 0.0
        
        # Check if we should enter cool-off
        if drawdown >= self.drawdown_threshold and not self.in_cool_off:
            self.in_cool_off = True
            self.cool_off_start = time.time()
            self.cool_off_count += 1
            logger.warning(f"Entering cool-off period: drawdown {drawdown:.2%} >= threshold {self.drawdown_threshold:.2%}")
            return True
        
        # Check if we should exit cool-off
        if self.in_cool_off:
            elapsed = time.time() - self.cool_off_start
            
            # Check time-based exit
            if elapsed >= self.cool_off_duration:
                # Check recovery
                if drawdown <= self.recovery_threshold:
                    self.in_cool_off = False
                    self.cool_off_start = None
                    logger.info(f"Exiting cool-off period: recovered to {drawdown:.2%}")
                    return False
                else:
                    # Extend cool-off
                    logger.info(f"Extending cool-off: still in drawdown {drawdown:.2%}")
                    return True
        
        return self.in_cool_off
    
    def get_state(self) -> Dict:
        """Get current cool-off state."""
        if not self.in_cool_off:
            return {"in_cool_off": False}
        
        elapsed = time.time() - self.cool_off_start
        remaining = max(0, self.cool_off_duration - elapsed)
        
        return {
            "in_cool_off": True,
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "cool_off_count": self.cool_off_count
        }
    
    def reset(self):
        """Reset cool-off state."""
        self.in_cool_off = False
        self.cool_off_start = None
        self.peak_equity = 0.0


class GovernanceEnforcer:
    """Main governance enforcement class that combines all governance rules."""
    
    def __init__(self, config: dict = None):
        """Initialize governance enforcer.
        
        Args:
            config: Optional config dict
        """
        config = config or {}
        
        # Position limits
        self.position_limits = PositionLimits(
            max_position_pct=config.get("MAX_POSITION_PCT", 0.25),
            max_total_exposure=config.get("MAX_TOTAL_EXPOSURE", 0.85)
        )
        
        # Risk parity
        self.risk_parity = RiskParityRebalancer(
            target_volatility=config.get("TARGET_VOLATILITY", 0.15)
        )
        
        # Drawdown cool-off
        self.cool_off = DrawdownCoolOff(
            drawdown_threshold=config.get("DRAWDOWN_COOL_OFF_THRESHOLD", 0.10),
            cool_off_duration_seconds=config.get("COOL_OFF_DURATION_SECONDS", 300),
            recovery_threshold=config.get("RECOVERY_THRESHOLD", 0.05)
        )
        
        # State
        self.enabled = config.get("ENABLED", True)
        self.current_equity = 10000.0  # Default starting capital
        
        logger.info("GovernanceEnforcer initialized")
    
    def enforce(self, allocations: Dict[str, float], total_capital: float) -> Dict[str, float]:
        """Enforce all governance rules on allocations.
        
        Args:
            allocations: Proposed allocations
            total_capital: Total capital
            
        Returns:
            Enforced allocations
        """
        if not self.enabled:
            return allocations
        
        # 1. Check cool-off
        if self.cool_off.check(self.current_equity):
            logger.warning("Governance: In cool-off period, reducing allocations by 50%")
            allocations = {k: v * 0.5 for k, v in allocations.items()}
        
        # 2. Enforce position limits
        allocations = self.position_limits.enforce(allocations, total_capital)
        
        # 3. Apply risk parity rebalancing if enabled
        # (This would be done as a separate step in production)
        
        return allocations
    
    def validate(self, allocations: Dict[str, float], total_capital: float) -> tuple:
        """Validate allocations against all governance rules.
        
        Returns:
            (is_valid, list of violations)
        """
        all_violations = []
        
        # Position limits
        valid, violations = self.position_limits.validate(allocations, total_capital)
        all_violations.extend(violations)
        
        # Cool-off state
        cool_off_state = self.cool_off.get_state()
        if cool_off_state.get("in_cool_off"):
            all_violations.append(f"In cool-off period: {cool_off_state}")
        
        return len(all_violations) == 0, all_violations
    
    def update_equity(self, equity: float):
        """Update current equity for drawdown tracking.
        
        Args:
            equity: Current portfolio equity
        """
        self.current_equity = equity
    
    def update_volatility(self, symbol: str, volatility: float):
        """Update volatility for risk parity calculations.
        
        Args:
            symbol: Asset symbol
            volatility: Annualized volatility
        """
        self.risk_parity.update_volatility(symbol, volatility)
    
    def get_status(self) -> Dict:
        """Get governance status summary."""
        return {
            "enabled": self.enabled,
            "in_cool_off": self.cool_off.get_state().get("in_cool_off", False),
            "cool_off_count": self.cool_off.cool_off_count,
            "current_equity": self.current_equity,
            "peak_equity": self.cool_off.peak_equity
        }


def create_governance_enforcer(config: dict = None) -> GovernanceEnforcer:
    """Factory function to create governance enforcer.
    
    Args:
        config: Optional config dict
        
    Returns:
        GovernanceEnforcer instance
    """
    return GovernanceEnforcer(config)
