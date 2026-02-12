MODE = 'PAPER'
MAX_DRAWDOWN = 0.15
"""
Global configuration for the Financial Organism
Central source of truth for modes, limits, and system behavior
"""

CONFIG = {
    # -------------------------
    # SYSTEM MODE
    # -------------------------
    # PAPER  : no real orders
    # SHADOW : live data, fake execution
    # LIVE   : real capital (locked)
    "MODE": "PAPER",

    # -------------------------
    # LOOP CONTROL
    # -------------------------
    "LOOP_INTERVAL": 5,  # seconds between cycles

    # -------------------------
    # CAPITAL SETTINGS
    # -------------------------
    "STARTING_CAPITAL": 10_000.0,
    "MAX_TOTAL_EXPOSURE": 0.30,     # 30% of capital
    "MAX_SINGLE_STRATEGY": 0.10,    # 10% per strategy
    "MAX_LEVERAGE": 2.0,

    # -------------------------
    # RISK LIMITS
    # -------------------------
    "MAX_DAILY_DRAWDOWN": 0.03,     # 3%
    "MAX_PORTFOLIO_DRAWDOWN": 0.10, # 10%
    "TAIL_RISK_LIMIT": 0.01,

    # -------------------------
    # CRISIS / BLACK SWAN
    # -------------------------
    "CRISIS_SENSITIVITY": 0.7,
    "AUTO_HEDGE_ENABLED": True,

    # -------------------------
    # EVOLUTION ENGINE
    # -------------------------
    "EVOLUTION_ENABLED": True,
    "GENETIC_MUTATION_RATE": 0.05,
    "GENETIC_CULL_THRESHOLD": -0.02,

    # -------------------------
    # LOGGING
    # -------------------------
    "LOG_LEVEL": "INFO"
}
