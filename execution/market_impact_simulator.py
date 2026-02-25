"""Market Impact and Order Book Simulator.

This module provides realistic execution cost modeling that depends on:
- Position size relative to average daily volume (ADV)
- Order book depth and liquidity
- Market volatility
- Historical spread

The model is based on the Almgren-Chriss linear impact model with
additional considerations for order book liquidity.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import random

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("MARKET_IMPACT")


class OrderBookSimulator:
    """Simulates order book behavior for execution cost estimation."""
    
    def __init__(self, 
                 base_spread_bps: float = 5.0,
                 base_depth: float = 100000.0,
                 price_precision: float = 0.01):
        """Initialize order book simulator.
        
        Args:
            base_spread_bps: Base spread in basis points (default 5 bps)
            base_depth: Base order book depth at best bid/ask (default $100k)
            price_precision: Price precision (default $0.01)
        """
        self.base_spread_bps = base_spread_bps
        self.base_depth = base_depth
        self.price_precision = price_precision
        
        # Volatility-adjusted parameters
        self.vol_spread_multiplier = 1.0
        self.vol_depth_multiplier = 1.0
    
    def update_market_conditions(self, volatility: float, volume: float):
        """Update market conditions based on current vol and volume.
        
        Args:
            volatility: Current annualized volatility (e.g., 0.30 = 30%)
            volume: Current trading volume relative to average
        """
        # Higher volatility -> wider spreads, lower depth
        self.vol_spread_multiplier = 1.0 + volatility * 2.0
        self.vol_depth_multiplier = max(0.3, 1.0 - volatility)
    
    def get_spread(self, symbol: str = None) -> float:
        """Get current spread in dollars.
        
        Returns:
            Spread in dollars
        """
        # Base spread in bps
        spread_bps = self.base_spread_bps * self.vol_spread_multiplier
        
        # Convert to dollars (assume $50,000 reference price for crypto)
        ref_price = 50000.0 if symbol in ["BTC", "WBTC"] else 2500.0
        spread_dollars = ref_price * (spread_bps / 10000.0)
        
        return spread_dollars
    
    def get_depth_at_levels(self, levels: int = 5) -> list:
        """Get order book depth at multiple price levels.
        
        Args:
            levels: Number of price levels to return
            
        Returns:
            List of (price_distance_bps, depth) tuples
        """
        depths = []
        depth = self.base_depth * self.vol_depth_multiplier
        
        for i in range(levels):
            # Each level is roughly 1 bps apart
            price_distance = (i + 1) * 1.0  # bps
            # Depth decreases as we move away from best price
            level_depth = depth * (0.7 ** i)
            depths.append((price_distance, level_depth))
        
        return depths
    
    def estimate_market_impact(self, order_size: float, symbol: str = None) -> float:
        """Estimate instantaneous market impact for an order.
        
        Uses a simplified square-root impact model:
        impact = alpha * sqrt(size / avg_daily_volume)
        
        Args:
            order_size: Order size in dollars
            symbol: Trading symbol
            
        Returns:
            Estimated impact in dollars
        """
        # Reference ADV (average daily volume) in dollars
        adv = 1000000.0  # $1M default ADV
        
        # Impact coefficient (varies by asset)
        alpha = 0.1  # 10% of spread per sqrt(ADV)
        
        # Square root impact model
        if adv > 0:
            size_ratio = order_size / adv
            impact_bps = alpha * np.sqrt(size_ratio) * 100  # Convert to bps
        else:
            impact_bps = 0
        
        # Convert to dollars
        ref_price = 50000.0 if symbol in ["BTC", "WBTC"] else 2500.0
        impact_dollars = ref_price * (impact_bps / 10000.0)
        
        return impact_dollars


class MarketImpactSimulator:
    """Simulates realistic market impact and execution costs.
    
    This combines:
    - Order book spread
    - Market impact from order size
    - Temporary vs permanent impact
    - Partial fill probabilities
    """
    
    def __init__(self, 
                 base_slippage_bps: float = 2.0,
                 base_fee_bps: float = 4.0,
                 impact_coefficient: float = 0.1):
        """Initialize market impact simulator.
        
        Args:
            base_slippage_bps: Base slippage in bps (default 2)
            base_fee_bps: Base trading fee in bps (default 4)
            impact_coefficient: Market impact coefficient (alpha)
        """
        self.base_slippage_bps = base_slippage_bps
        self.base_fee_bps = base_fee_bps
        self.impact_coefficient = impact_coefficient
        
        # Order book simulator
        self.order_book = OrderBookSimulator()
        
        # State
        self.last_volatility = 0.3
        self.last_volume = 1.0
    
    def update_market_conditions(self, volatility: float, volume: float = 1.0):
        """Update market conditions for impact calculations.
        
        Args:
            volatility: Current annualized volatility
            volume: Current volume relative to average (1.0 = average)
        """
        self.last_volatility = volatility
        self.last_volume = volume
        
        # Update order book
        self.order_book.update_market_conditions(volatility, volume)
    
    def calculate_execution_cost(self, 
                               order_size: float,
                               symbol: str,
                               side: str = "buy",
                               is_limit_order: bool = False) -> Dict[str, float]:
        """Calculate total execution cost for an order.
        
        Args:
            order_size: Order size in dollars
            symbol: Trading symbol (e.g., "BTC")
            side: "buy" or "sell"
            is_limit_order: Whether this is a limit order (reduces slippage)
            
        Returns:
            Dict with keys:
                - total_cost: Total cost in dollars
                - slippage_cost: Slippage cost in dollars
                - fee_cost: Trading fee in dollars
                - impact_cost: Market impact cost in dollars
                - spread_cost: Bid-ask spread cost in dollars
                - effective_price: Effective execution price
                - fill_probability: Probability of full fill
        """
        # Reference price (would come from market data in production)
        ref_prices = {"BTC": 50000.0, "ETH": 2500.0, "WBTC": 50000.0}
        ref_price = ref_prices.get(symbol, 1000.0)
        
        # 1. Spread cost (half of spread)
        spread_cost = self.order_book.get_spread(symbol) / 2.0
        
        # 2. Market impact cost
        impact_cost = self.order_book.estimate_market_impact(order_size, symbol)
        
        # Adjust impact for volatility
        impact_cost *= (1.0 + self.last_volatility * 2.0)
        
        # 3. Slippage (varies by order type)
        if is_limit_order:
            # Limit orders have lower slippage but may not fill
            slippage_multiplier = 0.3
            fill_probability = 0.85
        else:
            # Market orders have full slippage
            slippage_multiplier = 1.0
            fill_probability = 0.98
        
        base_slippage = ref_price * (self.base_slippage_bps / 10000.0)
        slippage_cost = base_slippage * slippage_multiplier
        
        # Adjust slippage for size (larger orders = more slippage)
        size_factor = min(3.0, 1.0 + (order_size / 100000.0))
        slippage_cost *= size_factor
        
        # 4. Trading fee
        fee_cost = ref_price * (self.base_fee_bps / 10000.0)
        
        # 5. Total cost
        total_cost = spread_cost + impact_cost + slippage_cost + fee_cost
        
        # 6. Effective price
        if side.lower() == "buy":
            effective_price = ref_price + total_cost
        else:
            effective_price = ref_price - total_cost
        
        # Calculate costs as percentage
        total_cost_pct = (total_cost / ref_price) * 100 if ref_price > 0 else 0
        spread_pct = (spread_cost / ref_price) * 100 if ref_price > 0 else 0
        impact_pct = (impact_cost / ref_price) * 100 if ref_price > 0 else 0
        slippage_pct = (slippage_cost / ref_price) * 100 if ref_price > 0 else 0
        fee_pct = (fee_cost / ref_price) * 100 if ref_price > 0 else 0
        
        return {
            "total_cost": total_cost,
            "total_cost_bps": total_cost_pct * 100,  # Convert % to bps
            "slippage_cost": slippage_cost,
            "slippage_cost_bps": slippage_pct * 100,
            "fee_cost": fee_cost,
            "fee_cost_bps": fee_pct * 100,
            "impact_cost": impact_cost,
            "impact_cost_bps": impact_pct * 100,
            "spread_cost": spread_cost,
            "spread_cost_bps": spread_pct * 100,
            "effective_price": effective_price,
            "fill_probability": fill_probability,
            "order_size": order_size,
            "ref_price": ref_price
        }
    
    def simulate_partial_fill(self,
                             order_size: float,
                             fill_probability: float = 0.98) -> Tuple[float, float]:
        """Simulate partial fill scenario.
        
        Args:
            order_size: Requested order size
            fill_probability: Probability of getting full fill
            
        Returns:
            Tuple of (filled_size, remaining_size)
        """
        # Simulate fill based on probability
        if random.random() < fill_probability:
            # Full fill
            return order_size, 0.0
        else:
            # Partial fill (between 50-90% of order)
            fill_ratio = random.uniform(0.5, 0.9)
            filled = order_size * fill_ratio
            remaining = order_size - filled
            return filled, remaining
    
    def calculate_twap_cost(self,
                           order_size: float,
                           symbol: str,
                           num_slices: int = 5,
                           slice_interval_seconds: int = 60) -> Dict[str, float]:
        """Calculate execution cost for TWAP (Time-Weighted Average Price) order.
        
        TWAP reduces market impact by slicing orders over time.
        
        Args:
            order_size: Total order size
            symbol: Trading symbol
            num_slices: Number of slices
            slice_interval_seconds: Interval between slices
            
        Returns:
            Dict with TWAP execution details
        """
        slice_size = order_size / num_slices
        total_cost = 0.0
        total_impact = 0.0
        total_slippage = 0.0
        
        for i in range(num_slices):
            # Each slice has reduced impact (market impact is temporary)
            # Later slices have slightly higher cost due to alpha decay
            slice_impact_mult = 1.0 - (i * 0.05)  # 5% decay per slice
            
            cost_result = self.calculate_execution_cost(
                slice_size, symbol, is_limit_order=True
            )
            
            # Only permanent impact carries over
            total_impact += cost_result["impact_cost"] * slice_impact_mult * 0.3
            total_slippage += cost_result["slippage_cost"]
            total_cost += cost_result["total_cost"]
        
        # Average cost per dollar
        avg_cost_per_dollar = total_cost / order_size if order_size > 0 else 0
        
        return {
            "total_cost": total_cost,
            "avg_cost_bps": avg_cost_per_dollar * 10000,  # Convert to bps
            "total_impact": total_impact,
            "total_slippage": total_slippage,
            "num_slices": num_slices,
            "slice_size": slice_size,
            "slice_interval": slice_interval_seconds
        }
    
    def get_liquidity_score(self, symbol: str) -> float:
        """Get a liquidity score for a symbol (0-1, higher = more liquid).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Liquidity score between 0 and 1
        """
        # In production, this would analyze real order book
        liquidity = {
            "BTC": 0.95,
            "ETH": 0.85,
            "WBTC": 0.60,
            "SOL": 0.50,
            "OTHER": 0.30
        }
        return liquidity.get(symbol, liquidity["OTHER"])
    
    def estimate_execution_price(self,
                                  order_size: float,
                                  symbol: str,
                                  side: str = "buy") -> float:
        """Quick estimate of execution price for an order.
        
        Args:
            order_size: Order size in dollars
            symbol: Trading symbol
            side: "buy" or "sell"
            
        Returns:
            Estimated execution price
        """
        cost_result = self.calculate_execution_cost(order_size, symbol, side)
        return cost_result["effective_price"]

    def estimate(self, order_size: float, volume: float) -> dict:
        """Quick estimate of market impact for an order.
        
        Args:
            order_size: Order size in dollars
            volume: Average daily volume in dollars
            
        Returns:
            Dict with impact estimate
        """
        # Calculate participation rate
        participation = order_size / volume if volume > 0 else 0
        
        # Estimate impact using square root model
        # impact = alpha * sqrt(participation)
        alpha = 0.1  # Impact coefficient
        impact_pct = alpha * (participation ** 0.5) * 100 if participation > 0 else 0
        
        # Calculate cost
        ref_price = 50000.0  # Reference price
        impact_cost = ref_price * (impact_pct / 10000.0)
        
        return {
            "order_size": order_size,
            "volume": volume,
            "participation": participation,
            "impact_pct": impact_pct,
            "impact_cost": impact_cost
        }


def create_market_impact_simulator(config: dict = None) -> MarketImpactSimulator:
    """Factory function to create market impact simulator with config.
    
    Args:
        config: Optional config dict
        
    Returns:
        MarketImpactSimulator instance
    """
    config = config or {}
    
    slippage = config.get("MARKET_IMPACT_SLIPPAGE_BPS", 2.0)
    fee = config.get("MARKET_IMPACT_FEE_BPS", 4.0)
    impact_coeff = config.get("MARKET_IMPACT_COEFFICIENT", 0.1)
    
    return MarketImpactSimulator(
        base_slippage_bps=slippage,
        base_fee_bps=fee,
        impact_coefficient=impact_coeff
    )


# ============================================================
# Integration with ExecutionEngine
# ============================================================

def apply_market_impact_to_execution(executor, market_impact_sim: MarketImpactSimulator):
    """Apply market impact calculations to execution engine.
    
    This modifies the execution engine to use realistic market impact
    instead of fixed bps costs.
    
    Args:
        executor: ExecutionEngine instance
        market_impact_sim: MarketImpactSimulator instance
    """
    # Store the simulator reference in executor
    executor.market_impact_sim = market_impact_sim
    
    # Monkey-patch the _apply_costs method
    original_apply_costs = executor._apply_costs
    
    def new_apply_costs(self, notional):
        """Apply realistic market impact costs."""
        # Get current market conditions from executor state
        # In production, this would come from live market data
        volatility = 0.3  # Default, would be dynamic
        market_impact_sim.update_market_conditions(volatility)
        
        # Get execution cost estimate
        # Determine symbol from notional (simplified)
        symbol = "BTC" if notional > 1000 else "ETH"
        
        cost_result = market_impact_sim.calculate_execution_cost(
            notional, symbol, is_limit_order=False
        )
        
        # Return adjusted notional after costs
        return max(0.0, notional - cost_result["total_cost"])
    
    # Replace the method
    executor._apply_costs = lambda notional: new_apply_costs(executor, notional)
    
    logger.info("Applied market impact simulator to execution engine")
