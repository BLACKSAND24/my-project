# Financial Organism

Run in PAPER mode only.

This repository implements a self‑contained algorithmic trading "organism" designed for solo/quasi-hedge‑fund operation.  Recent architectural enhancements bring it closer to the behaviour of large asset managers by introducing:

* **Global Risk Controller** – tracks total exposure, inter-strategy correlations and portfolio drawdown.
* **Regime Detection** – classifies markets into LOW_VOL / HIGH_VOL / CRISIS using free volatility data.
* **Macro & Sentiment Feeds** – optional FRED series polling and RSS headline sentiment (no paid data required).
* **Portfolio‑level AI & Confidence Gating** – weights are adjusted for strategy correlation, and very low conviction signals automatically shrink exposure (configurable).
* **Human‑AI Governance** – daily/weekly loss limits, hard kill switch and `/HALT_ALL` Telegram command.
* **Execution Improvements** – TWAP slicing & volatility‑aware sizing, smarter order routing simulation.
* **Resilience Hooks** – lightweight watchdog script and Docker readiness (see `monitoring/watchdog.py`).

All components are written in pure Python and rely only on standard library plus NumPy (already a dependency for the genetic engine).  You can run the full test suite under `tests/` to verify functionality.
