"""Live Market Feed Module (SHADOW Mode)

READ-ONLY public market data fetching.
No auth. No write endpoints. No trading.

Used by SHADOW mode to get real-time prices while keeping execution simulated.
"""
import time
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
    
    def get_market_data(self) -> dict:
        """Get full market snapshot (BTC, ETH prices and vol).
        
        Returns:
            dict with returns, volatility, etc. (same format as PAPER)
        """
        try:
            # Fetch live prices
            btc_price = self.get_price("BTC")
            eth_price = self.get_price("ETH")
            
            # Compute synthetic returns and volatility
            # (In production, would use real tick data)
            btc_vol = self.get_volatility("BTC")
            eth_vol = self.get_volatility("ETH")
            
            data = {
                "returns": [0.001, 0.002, -0.001, 0.0015, -0.0005],  # Placeholder
                "volatility": (btc_vol + eth_vol) / 2,
                "prices": {"BTC": btc_price, "ETH": eth_price},
            }
            self.logger.debug(f"Market snapshot: {data}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to get market snapshot: {e}")
            return self._fallback_market_data()
    
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
    
    def _fallback_market_data(self) -> dict:
        """Fallback data if live source unavailable."""
        return {
            "returns": [0.0] * 20,
            "volatility": 0.30,
            "prices": {"BTC": 45000.0, "ETH": 2500.0},
        }
