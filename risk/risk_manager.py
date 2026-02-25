import numpy as np

from financial_organism.config import CONFIG
from financial_organism.risk.kill_switch import KillSwitch
from financial_organism.risk.global_risk_controller import GlobalRiskController
from financial_organism.governance.human_ai_governance import HumanAIGovernance

class RiskManager:
    def __init__(self):
        start = float(CONFIG.get("STARTING_CAPITAL", 10000.0))
        self.equity_peak = start
        self.current_equity = start
        self.kill_switch = KillSwitch(
            max_drawdown=float(CONFIG.get("MAX_PORTFOLIO_DRAWDOWN", 0.10)),
            cooldown_seconds=int(CONFIG.get("KILL_SWITCH_COOLDOWN", 300)),
        )
        # extended tracking for daily/weekly hard limits
        self.last_update_date = None
        self.last_update_week = None
        # global risk brain for cross-strategy awareness
        self.global_brain = GlobalRiskController(history_len=int(CONFIG.get("RISK_HISTORY_LENGTH", 100)))

    def evaluate(self, allocations, market_data, crisis_signals):
        """Return True if system may continue trading.

        Adds the following layers on top of the original kill switch:
        1. human-AI governance overrides (manual halt, daily/weekly limits)
        2. global exposure monitoring / correlation checks
        """
        # check for manual halt first – this saves a lot of downstream work
        if HumanAIGovernance.is_halted():
            return False

        # update equity using the same primitive as before
        returns = market_data.get("returns", [])
        if returns:
            self.current_equity += float(returns[-1]) * 100.0
            self.equity_peak = max(self.equity_peak, self.current_equity)
        # update human governance limits
        if not HumanAIGovernance.check_limits(self.current_equity):
            return False

        # global brain handles exposures and correlations
        self.global_brain.update_exposure(allocations)
        self.global_brain.record_equity(self.current_equity)

        # enforce very simple correlation ceiling if configured
        corr_cfg = float(CONFIG.get("CORRELATION_THRESHOLD", 0.85))
        corr_info = self.global_brain.correlation_matrix()
        mat = corr_info.get("matrix")
        if mat is not None and mat.size > 1:
            # off-diagonal entries represent pairwise correlations
            # if any exceed the threshold we bail out
            # note: numpy corrcoef returns 1s on diagonal
            off_diag = mat.copy()
            np.fill_diagonal(off_diag, 0.0)
            # convert NaN to 0 so comparisons behave predictably
            off_diag = np.nan_to_num(off_diag, nan=0.0)
            if (off_diag >= corr_cfg).any():
                # log a warning but still allow risk function to return False
                print(f"[RISK] high correlation detected, threshold={corr_cfg}")
                return False

        # original kill switch semantics
        if self.kill_switch.is_locked():
            return False
        if self.kill_switch.check(self.equity_peak, self.current_equity):
            return False
        return True
