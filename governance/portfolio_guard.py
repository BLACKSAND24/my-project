"""Portfolio Guard Module.

This module provides portfolio governance capabilities including:
- Position-level caps
- Risk-parity-style rebalance
- Drawdown-triggered cool-off

This is a simplified wrapper around the enhanced governance layer.
"""
from typing import Dict, Optional

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("PORTFOLIO_GUARD")


class PortfolioGovernor:
    """Portfolio Governor with position-level caps, risk-parity rebalance, and drawdown cool-off."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Portfolio Governor.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Position limits
        self.max_position_pct = self.config.get("max_position_pct", 0.25)
        self.max_total_exposure = self.config.get("max_total_exposure", 0.85)
        
        # Drawdown settings
        self.drawdown_threshold = self.config.get("drawdown_threshold", 0.10)
        self.cool_off_active = False
        self.peak_equity = 10000.0
        self.current_equity = 10000.0
        
        # Risk parity settings
        self.volatility = {}  # symbol -> volatility
        
        logger.info("PortfolioGovernor initialized")
    
    def apply(self, allocations: Dict[str, float], market_state: Dict) -> Dict[str, float]:
        """Apply governance rules to allocations.
        
        Args:
            allocations: Dict of symbol -> allocation amount
            market_state: Dict with market information (e.g., {'max_drawdown': -0.02})
            
        Returns:
            Adjusted allocations after applying governance rules
        """
        if not allocations:
            return {}
        
        # Update current equity from market state
        if 'equity' in market_state:
            self.current_equity = market_state['equity']
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
        
        # Check for drawdown-triggered cool-off
        max_drawdown = market_state.get('max_drawdown', 0.0)
        if max_drawdown < -self.drawdown_threshold:
            self.cool_off_active = True
            logger.warning(f"Drawdown cool-off triggered: max_drawdown={max_drawdown:.2%}")
            # Reduce allocations by 50% during cool-off
            allocations = {k: v * 0.5 for k, v in allocations.items()}
        
        # Exit cool-off if recovered
        if self.cool_off_active and max_drawdown >= -0.02:  # Recovered to -2% DD
            self.cool_off_active = False
            logger.info("Exiting drawdown cool-off")
        
        # Apply position limits
        total_capital = self.config.get('total_capital', 10000.0)
        total_allocated = sum(allocations.values())
        
        # Check total exposure
        if total_allocated > total_capital * self.max_total_exposure:
            scale = (total_capital * self.max_total_exposure) / total_allocated
            allocations = {k: v * scale for k, v in allocations.items()}
            logger.info(f"Total exposure limited: scaled by {scale:.2f}")
        
        # Check individual position limits
        max_position = total_capital * self.max_position_pct
        for symbol, allocation in allocations.items():
            if allocation > max_position:
                logger.warning(f"Position limit for {symbol}: {allocation:.2f} > {max_position:.2f}")
                allocations[symbol] = max_position
        
        return allocations
    
    def get_status(self) -> Dict:
        """Get governance status.
        
        Returns:
            Dict with status information
        """
        return {
            "cool_off_active": self.cool_off_active,
            "max_position_pct": self.max_position_pct,
            "max_total_exposure": self.max_total_exposure,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity
        }


def create_portfolio_governor(config: Optional[Dict] = None) -> PortfolioGovernor:
    """Factory function to create PortfolioGovernor.
    
    Args:
        config: Optional config dict
        
    Returns:
        PortfolioGovernor instance
    """
    return PortfolioGovernor(config)
