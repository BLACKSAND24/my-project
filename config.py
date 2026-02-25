import os

CONFIG = {
    "MODE": "LIVE",  # PAPER | SHADOW | DRY_RUN | LIVE
    "LOOP_INTERVAL": 5,
    "STARTING_CAPITAL": 10000.0,
    "MAX_TOTAL_EXPOSURE": 0.30,
    "MAX_PORTFOLIO_DRAWDOWN": 0.10,
    # new governance / risk parameters
    "CORRELATION_THRESHOLD": 0.85,
    "DAILY_MAX_LOSS": 0.05,
    "WEEKLY_MAX_LOSS": 0.15,
    "VOLATILITY_ORDER_SIZE_THRESHOLD": 0.20,
    "TWAP_CHUNKS": 5,
    # regime detection thresholds (used by RegimeDetector)
    "LOW_VOL_THRESHOLD": 0.10,
    "HIGH_VOL_THRESHOLD": 0.30,
    "REGIME_WINDOW": 20,
    # macro data helpers
    "NEWS_RSS_URL": "https://news.google.com/rss/search?q=finance",
    "FRED_SERIES": "GDP",
    "AUTO_HEDGE_ENABLED": True,
    "AI_CRISIS_THRESHOLD": 0.70,
    "KILL_SWITCH_COOLDOWN": 300,
    "CRISIS_VOL_THRESHOLD": 0.35,
    "CRISIS_DRAWDOWN_THRESHOLD": -0.20,
    "MIN_TRADE_INTERVAL_SECONDS": 0.5,
    "HEDGE_VOL_TRIGGER": 0.18,
    "MAX_HEDGE_RATIO": 0.35,
    "HEDGE_SYMBOL": "INDEX_PUT_PROXY",
    # Credentials should be provided via environment variables to avoid committing secrets.
    # Example (PowerShell):
    #  $env:EXCHANGE_API_KEY = 'your-key'
    #  $env:EXCHANGE_API_SECRET = 'your-secret'
    "EXCHANGE_API_KEY": os.getenv("EXCHANGE_API_KEY", ""),
    "EXCHANGE_API_SECRET": os.getenv("EXCHANGE_API_SECRET", ""),
    "RISK_ACKNOWLEDGED": True,  # Must be True to pass go-live checklist
    "LOG_LEVEL": "INFO",
    # Live market feed (SHADOW mode)
    "LIVE_FEED_ENABLED": True,
    "LIVE_FEED_CACHE_SECONDS": 5,
    # DRY-RUN phase (Option 2): Micro-capital validation before LIVE
    "DRY_RUN_ENABLED": False,
    "DRY_RUN_EXCHANGE": "SANDBOX",  # SANDBOX | TESTNET | PAPER_TRADING
    "DRY_RUN_POSITION_LIMIT": 10.0,  # Max $10 per order
    "DRY_RUN_BURN_IN_CYCLES": 50,  # Run 50+ SHADOW cycles before dry-run
    # Binance testnet (recommended for fast local dry-run integration)
    "EXCHANGE_SANDBOX_URL": "https://testnet.binance.vision/api",
    "EXCHANGE_SANDBOX_KEY": "",
    "EXCHANGE_SANDBOX_SECRET": "",
    # Exchange selection (used to tune thresholds)
    "EXCHANGE": "BINANCE",
    # Execution efficiency thresholds per exchange (hysteresis model)
    "EXEC_EFFICIENCY_THRESHOLDS": {
        "DEFAULT": {
            "TO_GO_READY": 0.975,
            "TO_STAY_READY": 0.970,
            "MINIMUM": 0.960,
            "ROLLING_WINDOW": 10
        },
        "BINANCE": {
            # Binance is typically very liquid; require higher entry bar but keep stay threshold
            "TO_GO_READY": 0.975,
            "TO_STAY_READY": 0.970,
            "MINIMUM": 0.960,
            "ROLLING_WINDOW": 20
        }
    },
}

# Safety assertion
_valid_modes = ("PAPER", "SHADOW", "DRY_RUN", "LIVE")
_current_mode = CONFIG.get("MODE", "PAPER")
assert _current_mode in _valid_modes, f"Invalid MODE: {_current_mode}. Must be one of {_valid_modes}"
