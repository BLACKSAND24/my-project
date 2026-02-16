# DRY-RUN → LIVE Checklist

This checklist collects essential steps and commands to move from DRY-RUN/PAPER to LIVE.

## Preflight
- Confirm `governance.go_live_checklist.evaluate_checklist(build_context())` passes all items.
- Confirm shadow/burn-in metrics: rolling EE window=10, EE_MIN=97.5%, hysteresis=0.3% (observed stable).
- Confirm `max_dd` <= acceptable threshold (e.g., 1%).

## Repo / Git
- Ensure a remote is set. Example (HTTPS):

```powershell
git remote add origin https://github.com/<your-org>/<your-repo>.git
```

- Or SSH:

```powershell
git remote add origin git@github.com:<your-org>/<your-repo>.git
```

- Push current branch and tag a release:

```powershell
git push -u origin HEAD
git tag -a v1.0.0 -m "live-ready: EE stable" 
git push origin v1.0.0
```

## Logging / Metrics
- `ee_metrics.csv` is written to `logs/` automatically by `monitoring.logger.log_ee_metrics(...)`.
- Confirm the process has write access to `logs/`.
- Sample invocation (from your monitoring loop):

```python
from monitoring.analytics import report
metrics = {
    "rolling_avg_ee": 100.0,
    "window": 10,
    "ee_min": 97.5,
    "hysteresis": 0.3,
    "per_strategy": {"mean_revert": 100.0, "vol_breakout": 100.0, "momentum": 80.0},
    "latency_min_ms": 11,
    "latency_max_ms": 47,
    "capital_util_min_pct": 72,
    "capital_util_max_pct": 84,
    "max_dd": 0.0
}
summary = report(metrics)
print(summary)
```

## Go-Live steps
1. Create/confirm remote and push (see above).
2. Tag a release.
3. Switch runtime mode to `LIVE` in `config.py` or via env var.
4. Start in a limited-capacity ramp (e.g., 10% capital) and monitor `ee_metrics.csv` and dashboard one-line summary.
5. If all green for a burn-in period, increase allocation per policy.

## Rollback
- Have the kill-switch handler ready: see `risk/kill_switch.py` and ensure contact points for alerts are configured.

---

If you prefer, I can run the `git` commands for you (you'll need to provide the remote URL and auth method), or update an existing CI job to push & tag automatically.
