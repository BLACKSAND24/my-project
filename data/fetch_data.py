"""Market Data Fetching Module.

This module provides functions to fetch simulated market data with
extended features including:
- Historical returns
- Macro data (sentiment, FRED series)
- Liquidity/volume features

These features are needed by the ML policy and execution stack.
"""
import random
import math


def fetch_market_data(window_size=120, include_macro=True, include_volume=True):
    """Fetch simulated market data with extended features.
    
    Args:
        window_size: Number of periods to generate
        include_macro: Include macro features (sentiment, FRED)
        include_volume: Include volume/liquidity features
        
    Returns:
        Dictionary with market data including:
        - returns: List of returns
        - prices: Dict of symbol -> price
        - volatility: Annualized volatility
        - volume: Volume data (if include_volume=True)
        - liquidity: Liquidity metrics (if include_volume=True)
        - macro_sentiment: Sentiment score (if include_macro=True)
    """
    # Generate base returns
    returns = [random.gauss(0.0001, 0.01) for _ in range(window_size)]
    
    # Calculate volatility
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    volatility = math.sqrt(variance * 252)  # Annualize
    
    # Generate prices (starting from 100)
    prices = {"BTC": 45000.0, "ETH": 2500.0}
    price_data = {}
    for symbol, base_price in prices.items():
        # Add some variation based on returns
        cumulative_return = sum(returns)
        price = base_price * (1 + cumulative_return)
        price_data[symbol] = price
    
    data = {
        "returns": returns,
        "prices": price_data,
        "volatility": volatility,
    }
    
    # Add volume features if requested
    if include_volume:
        # Generate synthetic volume data
        base_volume = 1000000  # $1M default
        volumes = [random.gauss(base_volume, base_volume * 0.2) for _ in range(window_size)]
        
        # Volume features
        avg_volume = sum(volumes) / len(volumes)
        volume_std = math.sqrt(sum((v - avg_volume) ** 2 for v in volumes) / len(volumes))
        
        data["volume"] = volumes[-1] if volumes else base_volume
        data["avg_volume"] = avg_volume
        data["volume_std"] = volume_std
        
        # Liquidity metrics (0-1 scale, higher = more liquid)
        # Estimate based on volume relative to price
        liquidity_scores = {}
        for symbol, price in price_data.items():
            if price > 0:
                # Higher volume/price ratio = higher liquidity
                volume_to_price = volumes[-1] / price if price > 0 else 0
                # Normalize to 0-1 (assuming max ratio of 100)
                liquidity = min(1.0, volume_to_price / 100)
                liquidity_scores[symbol] = liquidity
            else:
                liquidity_scores[symbol] = 0.5
        
        data["liquidity"] = liquidity_scores
        data["liquidity_score"] = sum(liquidity_scores.values()) / len(liquidity_scores) if liquidity_scores else 0.5
    
    # Add macro features if requested
    if include_macro:
        # Generate synthetic sentiment (random between -1 and 1)
        data["macro_sentiment"] = random.uniform(-1, 1)
        
        # Synthetic FRED data (would be real in production)
        # GDP growth rate (percentage)
        data["fred"] = [
            {"date": "2024-01-01", "value": "2.5"},
            {"date": "2024-02-01", "value": "2.3"},
            {"date": "2024-03-01", "value": "2.1"},
        ]
        
        # Add macro features to data dict for ML policy
        data["macro_features"] = {
            "sentiment": data["macro_sentiment"],
            "fred_gdp": float(data["fred"][-1]["value"]) if data["fred"] else 0.0,
        }
    
    return data
