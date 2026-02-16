# Live Deployment Verification Checklist

Run these commands in order from the repo root to verify everything is working.

---

## 1️⃣ Verify Git Push Succeeded

```powershell
git log --oneline -5
git tag -l | grep v1.0.0
git remote -v
```

✅ **Expected:**
- Latest commit message: "feat: switch to LIVE mode..."
- `v1.0.0` tag listed
- Remote shows `origin https://github.com/BLACKSAND24/my-project.git`

---

## 2️⃣ Verify Config is LIVE

```powershell
python -c "from financial_organism.config import CONFIG; print(f'MODE={CONFIG[\"MODE\"]}')"
```

✅ **Expected:** `MODE=LIVE`

---

## 3️⃣ Verify EE Metrics Logging is in Place

```powershell
python -c "from financial_organism.monitoring.logger import log_ee_metrics; print('✓ EE logging imported successfully')"
python -c "from financial_organism.monitoring.analytics import report; print('✓ Analytics report imported successfully')"
```

✅ **Expected:** Both lines print "✓ ..."

---

## 4️⃣ Run System and Watch for EE Summary Logs

**Option A: Quick 5-cycle test (LIVE mode)**

```powershell
cd financial_organism
python main.py --cycles 5 --interval 2
```

**Option B: Watch continuously (default)**

```powershell
cd financial_organism
python main.py
```

To stop: Press `Ctrl+C`

---

## 5️⃣ Expected Output

Watch logs for these key indicators:

### Boot Banner ✅
```
================================================================================
🧠 FINANCIAL ORGANISM BOOTING
MODE            : LIVE
PRIORITY        : Survival > Prediction > Profit
================================================================================
```

### Per-cycle governance + EE Summary ✅
```
[GOVERNANCE] readiness=LIVE_READY | ...
[EE_SUMMARY] EE=100.00% (w=10) | Latency=11-47ms | Capital=72-84% | max_dd=0.00%
[EE_MONITOR] BTC | eff=100% | latency=15.3ms | cap=80% | ready=True
```

### CSV Creation ✅
After running, check:

```powershell
ls logs/ee_metrics.csv
Get-Content logs/ee_metrics.csv -Head 3
```

Expected header:
```
timestamp,rolling_avg_ee,window,ee_min,hysteresis,per_strategy,latency_min_ms,...
```

---

## 6️⃣ Readiness Gate Check

```powershell
python -c "
from financial_organism.governance.paper_to_live_readiness import ReadinessGate
from financial_organism.execution.execution_engine import ExecutionEngine
from financial_organism.risk.risk_manager import RiskManager

gate = ReadinessGate()
engine = ExecutionEngine(mode='LIVE')
risk = RiskManager()

# Simulate minimal execution for readiness query
readiness = gate.compute_readiness(engine, risk)
print(f'Readiness Status: {readiness.get(\"LIVE_READY\", False)}')
print(f'Rolling Avg EE: {readiness.get(\"rolling_avg_execution_efficiency\", 0):.2%}')
"
```

✅ **Expected:**
- `Readiness Status: True` (or `False` before burn-in complete)
- Rolling EE shows 97.5%+ if LIVE_READY

---

## 7️⃣ Checklist Summary

| Item | Command | Expected |
|------|---------|----------|
| Git push | `git log --oneline -1` | "feat: switch to LIVE mode..." |
| Tag | `git tag -l \| grep v1.0.0` | `v1.0.0` |
| MODE | `python -c "from financial_organism.config import CONFIG; print(CONFIG['MODE'])"` | `LIVE` |
| EE Logger | `python -c "from financial_organism.monitoring.logger import log_ee_metrics; print('✓')"` | `✓` |
| Analytics | `python -c "from financial_organism.monitoring.analytics import report; print('✓')"` | `✓` |
| Main loop | `python main.py --cycles 5` | Boots + logs 5 cycles + creates `logs/ee_metrics.csv` |
| CSV exists | `ls logs/ee_metrics.csv` | File exists with header row |
| [EE_SUMMARY] | Watch logs | `EE=100.00% (w=10) \| ...` |

---

## 🚨 Troubleshooting

| Issue | Diagnostic | Fix |
|-------|-----------|-----|
| `MODE=SHADOW` | Config not updated | Re-check `config.py` line 1 |
| `ImportError: ... analytics` | Missing patch | Verify `monitoring/analytics.py` has `from .logger import log_ee_metrics` |
| `logs/ee_metrics.csv` not created | Runtime perms | Ensure `logs/` dir is writable |
| No `[EE_SUMMARY]` in logs | Missing hook in main loop | Verify `main.py` line ~273 has `ee_report(metrics)` call |
| Git push failed | Remote not configured | Run `git remote add origin https://github.com/BLACKSAND24/my-project.git` |

---

## ✅ Final Validation

Once all checks pass, your system is **LIVE-ready** with:
- ✅ Code committed & tagged (`v1.0.0`)
- ✅ MODE switched to LIVE
- ✅ EE metrics persisting to CSV
- ✅ One-line monitoring summary in logs
- ✅ Governance readiness gate active

You can now:
1. **Monitor** `logs/ee_metrics.csv` in production
2. **Alert** on EE < 97.5% (readiness threshold)
3. **Scale** capital gradually (10% → 25% → 50% ...) and re-verify at each ramp
