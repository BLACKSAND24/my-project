"""Human‑AI governance safety layer.

Provides:
  * hard limits (daily / weekly drawdown)
  * manual override (HALT_ALL) with telegram notification
  * simple audit trail for external review

The implementation is intentionally minimal; the `halt()` method can be
called from a CLI script, a Telegram webhook listener, or tests.
"""
import datetime
import os
from financial_organism.config import CONFIG
from financial_organism.monitoring.telegram_alerts import alert as telegram_alert


class HumanAIGovernance:
    # state is kept as class attributes so that any imported reference
    # sees the same flag.  Persistence / distributed coordination is
    # left to the operator (e.g. writing a file) if needed.
    halted = False
    last_daily_peak = None  # equity value at start of day
    last_weekly_peak = None  # equity value at start of week

    @classmethod
    def halt(cls, reason: str = "manual"):
        """Trigger a full-system halt.

        This method should be called when a human or an automated check
        decides the environment must stop immediately.  It is safe to
        call multiple times.
        """
        cls.halted = True
        try:
            telegram_alert(f"🔴 HALT_ALL triggered ({reason})")
        except Exception:
            # best-effort, do not raise
            pass

    @classmethod
    def is_halted(cls) -> bool:
        return cls.halted

    @classmethod
    def reset_halt(cls):
        cls.halted = False

    @classmethod
    def update_peaks(cls, equity: float, now: datetime.datetime = None):
        """Maintain daily/weekly peak trackers for hard loss limits."""
        now = now or datetime.datetime.utcnow()
        # initialise peaks if missing
        if cls.last_daily_peak is None:
            cls.last_daily_peak = equity
        if cls.last_weekly_peak is None:
            cls.last_weekly_peak = equity

        # roll over daily
        if getattr(cls, "_last_date", None) is None:
            # first invocation: record date but do not overwrite existing peak
            cls._last_date = now.date()
        elif now.date() != cls._last_date:
            cls.last_daily_peak = equity
            cls._last_date = now.date()
        # roll over weekly (ISO week number)
        iso_week = now.isocalendar()[1]
        if getattr(cls, "_last_week", None) is None:
            cls._last_week = iso_week
        elif iso_week != cls._last_week:
            cls.last_weekly_peak = equity
            cls._last_week = iso_week

    @classmethod
    def check_limits(cls, equity: float) -> bool:
        """Return False if a hard daily/weekly loss limit has been breached."""
        cls.update_peaks(equity)
        # guard against division by zero
        if cls.last_daily_peak and cls.last_daily_peak > 0:
            daily_dd = (cls.last_daily_peak - equity) / cls.last_daily_peak
        else:
            daily_dd = 0.0
        if cls.last_weekly_peak and cls.last_weekly_peak > 0:
            weekly_dd = (cls.last_weekly_peak - equity) / cls.last_weekly_peak
        else:
            weekly_dd = 0.0

        if daily_dd >= float(CONFIG.get("DAILY_MAX_LOSS", 0.05)):
            telegram_alert(f"⚠️ DAILY loss limit breached ({daily_dd:.1%})")
            return False
        if weekly_dd >= float(CONFIG.get("WEEKLY_MAX_LOSS", 0.15)):
            telegram_alert(f"⚠️ WEEKLY loss limit breached ({weekly_dd:.1%})")
            return False
        return True
