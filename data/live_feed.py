"""Live Market Feed Module (SHADOW Mode)

READ-ONLY public market data fetching.
No auth. No write endpoints. No trading.

Used by SHADOW mode to get real-time prices while keeping execution simulated.

Extended to include macro and liquidity/volume features needed by the ML policy
and execution stack.
"""
import time
import random
from financial_organism.utils.logger import get_logger

logger = get_logger("LIVE_FEED")


class LiveMarketFeed:
    """Fetch LIVE market data (read-only, no auth required)."""
    
    def __init__(self, cache_seconds=5):
        """
        Args:
            cache_seconds: Cache live data for this long to avoid spam
        """
        self.cache_seconds = cache_seconds
        self.last_prices = {}
        self.last_fetch_time = {}
        self.last_returns = []
        self.last_volume = {}
        self.logger = logger
    
    def get_price(self, symbol: str) -> float:
        """Get latest price for a symbol (cached).
        
        Args:
            symbol: e.g., "BTC", "ETH"
        
        Returns:
            float price (or last known price if unavailable)
        """
        # Check cache first
        now = time.time()
        if symbol in self.last_fetch_time:
            age = now - self.last_fetch_time[symbol]
            if age < self.cache_seconds and symbol in self.last_prices:
                return self.last_prices[symbol]
        
        # Fetch fresh data (stub: would call real API here)
        try:
            price = self._fetch_price_from_source(symbol)
            self.last_prices[symbol] = price
            self.last_fetch_time[symbol] = now
            self.logger.debug(f"Fetched {symbol}: ${price:.2f}")
            return price
        except Exception as e:
            self.logger.warning(f"Failed to fetch {symbol}: {e}")
            # Return cached price or fallback
            return self.last_prices.get(symbol, 0.0)
    
    def get_volatility(self, symbol: str) -> float:
        """Get estimated volatility for a symbol.
        
        Args:
            symbol: e.g., "BTC", "ETH"
        
        Returns:
            float volatility (annualized, e.g., 0.35 = 35%)
        """
        try:
            vol = self._fetch_volatility_from_source(symbol)
            self.logger.debug(f"Fetched vol {symbol}: {vol:.2%}")
            return vol
        except Exception as e:
            self.logger.warning(f"Failed to fetch volatility {symbol}: {e}")
            return 0.0
    
    def get_volume(self, symbol: str) -> float:
        """Get estimated volume for a symbol.
        
        Args:
            symbol: e.g., "BTC", "ETH"
        
        Returns:
            float volume (in dollars)
        """
        try:
            volume = self._fetch_volume_from_source(symbol)
            self.last_volume[symbol] = volume
            return volume
        except Exception as e:
            self.logger.warning(f"Failed to fetch volume {symbol}: {e}")
            return self.last_volume.get(symbol, 1000000.0)
    
    def get_market_data(self) -> dict:
        """Get full market snapshot (BTC, ETH prices, vol, volume, liquidity).
        
        Extended to include macro and liquidity/volume features needed by ML policy
        and execution stack.
        
        Returns:
            dict with returns, volatility, prices, volume, liquidity, macro features
        """
        try:
            # Fetch live prices
            btc_price = self.get_price("BTC")
            eth_price = self.get_price("ETH")
            
            # Compute synthetic returns and volatility
            # (In production, would use real tick data)
            btc_vol = self.get_volatility("BTC")
            eth_vol = self.get_volatility("ETH")
            
            # Fetch volumes
            btc_volume = self.get_volume("BTC")
            eth_volume = self.get_volume("ETH")
            
            # Generate synthetic returns (in production, use actual returns)
            returns = self._generate_returns()
            
            # Calculate average volatility
            avg_volatility = (btc_vol + eth_vol) / 2
            
            # Calculate liquidity scores
            liquidity_scores = {}
            for symbol, price in [("BTC", btc_price), ("ETH", eth_price)]:
                volume = btc_volume if symbol == "BTC" else eth_volume
                if price > 0:
                    volume_to_price = volume / price
                    liquidity_scores[symbol] = min(1.0, volume_to_price / 100)
                else:
                    liquidity_scores[symbol] = 0.5
            
            data = {
                "returns": returns,
                "volatility": avg_volatility,
                "prices": {"BTC": btc_price, "ETH": eth_price},
                # Volume features
                "volume": (btc_volume + eth_volume) / 2,
                "volumes": {"BTC": btc_volume, "ETH": eth_volume},
                # Liquidity features
                "liquidity": liquidity_scores,
                "liquidity_score": sum(liquidity_scores.values()) / len(liquidity_scores),
                # Macro features (would come from real sources in production)
                "macro_sentiment": random.uniform(-1, 1),
                "fred": [
                    {"date": "2024-01-01", "value": "2.5"},
                    {"date": "2024-02-01", "value": "2.3"},
                    {"date": "2024-03-01", "value": "2.1"},
                ],
            }
            
            # Add macro features for ML policy
            data["macro_features"] = {
                "sentiment": data["macro_sentiment"],
                "fred_gdp": float(data["fred"][-1]["value"]) if data["fred"] else 0.0,
            }
            
            self.logger.debug(f"Market snapshot: {data}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get market snapshot: {e}")
            return self._fallback_market_data()
    
    def _generate_returns(self) -> list:
        """Generate synthetic returns for the live feed."""
        # Keep last 20 returns
        if len(self.last_returns) < 20:
            # Initialize with some returns
            self.last_returns = [random.gauss(0.0001, 0.01) for _ in range(20)]
        else:
            # Add new return
            new_return = random.gauss(0.0001, 0.01)
            self.last_returns.append(new_return)
            self.last_returns = self.last_returns[-20:]
        
        return self.last_returns
    
    # ============================================================
    # Protected methods (would integrate with real APIs in production)
    # ============================================================
    
    def _fetch_price_from_source(self, symbol: str) -> float:
        """Fetch live price from public endpoint (stub).
        
        In production, integrate with:
        - CoinGecko API (free, no auth)
        - Binance public ticker
        - Yahoo Finance
        """
        # Placeholder: would be real API call
        prices = {"BTC": 45000.0, "ETH": 2500.0}
        return prices.get(symbol, 1000.0)
    
    def _fetch_volatility_from_source(self, symbol: str) -> float:
        """Fetch volatility estimate (stub).
        
        In production, compute from recent returns or IV feeds.
        """
        # Placeholder
        vols = {"BTC": 0.35, "ETH": 0.40}
        return vols.get(symbol, 0.30)
    
    def _fetch_volume_from_source(self, symbol: str) -> float:
        """Fetch volume estimate (stub).
        
        In production, fetch from exchange APIs.
        """
        # Placeholder - would be real volume data
        volumes = {"BTC": 1000000000.0, "ETH": 500000000.0}  # In dollars
        return volumes.get(symbol, 1000000.0)
    
    def _fallback_market_data(self) -> dict:
        """Fallback data if live source unavailable."""
        return {
            "returns": [0.0] * 20,
            "volatility": 0.30,
            "prices": {"BTC": 45000.0, "ETH": 2500.0},
            "volume": 1000000.0,
            "volumes": {"BTC": 1000000000.0, "ETH": 500000000.0},
            "liquidity": {"BTC": 0.8, "ETH": 0.7},
            "liquidity_score": 0.75,
            "macro_sentiment": 0.0,
            "fred": [],
            "macro_features": {"sentiment": 0.0, "fred_gdp": 0.0},
        }
