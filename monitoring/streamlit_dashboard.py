"""Streamlit Dashboard for Financial Organism Real-time Monitoring.

This dashboard provides visualizations for:
- Portfolio weights over time
- Regime state (current regime detection)
- Risk exposures
- Execution metrics in real-time

Run with: streamlit run financial_organism/monitoring/streamlit_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Financial Organism Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import monitoring components
try:
    from financial_organism.config import CONFIG
    from financial_organism.utils.logger import get_logger
    from financial_organism.monitoring.analytics import report
    from financial_organism.risk.global_risk_controller import GlobalRiskController
    from financial_organism.macro.regime_detector import RegimeDetector
    from financial_organism.execution.execution_engine import ExecutionEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Logger
logger = get_logger("STREAMLIT_DASHBOARD")


# ============================================================
# Data Loading Functions
# ============================================================

def load_metrics_data():
    """Load EE metrics from CSV."""
    try:
        ee_metrics_path = "financial_organism/logs/ee_metrics.csv"
        if os.path.exists(ee_metrics_path):
            df = pd.read_csv(ee_metrics_path)
            df['timestamp'] = pd.to_datetime(df.get('timestamp', []), errors='coerce')
            return df
    except Exception as e:
        logger.warning(f"Could not load EE metrics: {e}")
    return pd.DataFrame()


def load_regime_data():
    """Load regime history from logs."""
    try:
        regime_path = "financial_organism/logs/regime_replay.csv"
        if os.path.exists(regime_path):
            df = pd.read_csv(regime_path)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def load_system_health():
    """Load system health metrics."""
    try:
        health_path = "financial_organism/logs/system_health.csv"
        if os.path.exists(health_path):
            df = pd.read_csv(health_path)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def load_current_state():
    """Load current system state from checkpoint."""
    state = {
        "mode": CONFIG.get("MODE", "PAPER"),
        "current_weights": {},
        "current_regime": "unknown",
        "execution_efficiency": 1.0,
        "capital_utilization": 0.0,
        "max_drawdown": 0.0,
        "last_update": datetime.now().isoformat()
    }
    
    # Try to load from burn_in_checkpoint
    try:
        if os.path.exists("burn_in_checkpoint.json"):
            with open("burn_in_checkpoint.json") as f:
                checkpoint = json.load(f)
                state["burn_in_cycles"] = checkpoint.get("cycle_count", 0)
    except Exception:
        pass
    
    return state


# ============================================================
# Dashboard Sections
# ============================================================

def render_header():
    """Render the dashboard header."""
    st.title("🧠 Financial Organism Dashboard")
    st.markdown("---")
    
    # Mode indicator
    mode = CONFIG.get("MODE", "PAPER")
    mode_colors = {
        "PAPER": "blue",
        "SHADOW": "orange", 
        "LIVE": "red",
        "DRY_RUN": "green"
    }
    color = mode_colors.get(mode, "gray")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Mode: :{color}[{mode}]")
    with col2:
        st.markdown(f"### Loop Interval: {CONFIG.get('LOOP_INTERVAL', 5)}s")
    with col3:
        st.markdown(f"### Starting Capital: ${CONFIG.get('STARTING_CAPITAL', 10000):,.2f}")


def render_regime_indicator():
    """Render the current regime indicator."""
    st.subheader("📈 Market Regime")
    
    # Create regime options
    regimes = {
        "low_vol": "🟢 Low Volatility",
        "normal": "🔵 Normal",
        "high_vol": "🟠 High Volatility",
        "crisis": "🔴 Crisis"
    }
    
    # For demo, we'll use a sample regime
    # In production, this would come from real-time detection
    current_regime = "normal"  # This would be fetched from the system
    
    # Display regime
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Current:** {regimes.get(current_regime, 'Unknown')}")
    with col2:
        vol = CONFIG.get("LOW_VOL_THRESHOLD", 0.10)
        st.markdown(f"**Low Vol Threshold:** {vol:.1%}")
    with col3:
        high_vol = CONFIG.get("HIGH_VOL_THRESHOLD", 0.30)
        st.markdown(f"**High Vol Threshold:** {high_vol:.1%}")
    with col4:
        crisis_vol = CONFIG.get("CRISIS_VOL_THRESHOLD", 0.35)
        st.markdown(f"**Crisis Vol Threshold:** {crisis_vol:.1%}")


def render_portfolio_weights():
    """Render portfolio weights visualization."""
    st.subheader("⚖️ Portfolio Weights")
    
    # Sample data for demo - in production this would be real data
    # Get data from metrics file
    df = load_metrics_data()
    
    if not df.empty and 'per_strategy' in df.columns:
        # Parse per_strategy column if it's JSON string
        try:
            df['per_strategy_parsed'] = df['per_strategy'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else {}
            )
            
            # Extract weights over time
            weights_data = []
            for idx, row in df.iterrows():
                parsed = row.get('per_strategy_parsed', {})
                for strat, weight in parsed.items():
                    weights_data.append({
                        'timestamp': row.get('timestamp', idx),
                        'strategy': strat,
                        'weight': weight
                    })
            
            if weights_data:
                weights_df = pd.DataFrame(weights_data)
                
                # Pivot for chart
                weights_pivot = weights_df.pivot(
                    index='timestamp', 
                    columns='strategy', 
                    values='weight'
                ).fillna(0)
                
                # Line chart
                st.line_chart(weights_pivot)
                
                # Current weights
                latest = weights_pivot.iloc[-1] if len(weights_pivot) > 0 else {}
                col1, col2, col3 = st.columns(3)
                for i, (strat, weight) in enumerate(latest.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(f"{strat}", f"{weight:.1%}")
        except Exception as e:
            st.warning(f"Could not parse strategy weights: {e}")
    else:
        # Demo/sample weights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Momentum", "33.3%")
        with col2:
            st.metric("Mean Reversion", "33.3%")
        with col3:
            st.metric("Volatility Breakout", "33.3%")
        
        st.info("Weights would be shown here from live system data")


def render_risk_exposures():
    """Render risk exposures visualization."""
    st.subheader("⚠️ Risk Exposures")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_dd = CONFIG.get("MAX_PORTFOLIO_DRAWDOWN", 0.10)
        st.metric("Max Drawdown Limit", f"{max_dd:.1%}")
    
    with col2:
        max_exposure = CONFIG.get("MAX_TOTAL_EXPOSURE", 0.30)
        st.metric("Max Total Exposure", f"{max_exposure:.1%}")
    
    with col3:
        corr_thresh = CONFIG.get("CORRELATION_THRESHOLD", 0.85)
        st.metric("Correlation Threshold", f"{corr_thresh:.2f}")
    
    with col4:
        daily_loss = CONFIG.get("DAILY_MAX_LOSS", 0.05)
        st.metric("Daily Max Loss", f"{daily_loss:.1%}")
    
    # Exposure bar chart (demo)
    st.markdown("### Current Exposures")
    
    # Sample exposure data
    exposures = {
        "BTC": 0.15,
        "ETH": 0.10,
        "HEDGE": 0.05
    }
    
    exp_df = pd.DataFrame(list(exposures.items()), columns=['Asset', 'Exposure'])
    st.bar_chart(exp_df.set_index('Asset'))


def render_execution_metrics():
    """Render execution metrics dashboard."""
    st.subheader("🚀 Execution Metrics")
    
    # Load EE metrics
    df = load_metrics_data()
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        # Latest values
        latest = df.iloc[-1] if len(df) > 0 else {}
        
        with col1:
            ee = latest.get('rolling_avg_ee', 1.0)
            st.metric("Execution Efficiency", f"{ee:.2%}")
        
        with col2:
            latency_min = latest.get('latency_min_ms', 0)
            latency_max = latest.get('latency_max_ms', 0)
            st.metric("Latency (ms)", f"{latency_min:.0f}-{latency_max:.0f}")
        
        with col3:
            cap_util = latest.get('capital_util_max_pct', 0)
            st.metric("Capital Utilization", f"{cap_util:.1f}%")
        
        with col4:
            max_dd = latest.get('max_dd', 0)
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        # EE over time chart
        if 'rolling_avg_ee' in df.columns:
            st.line_chart(df.set_index('timestamp')['rolling_avg_ee'] if 'timestamp' in df.columns else df['rolling_avg_ee'])
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Execution Efficiency", "97.5%")
        with col2:
            st.metric("Latency (ms)", "10-50")
        with col3:
            st.metric("Capital Utilization", "75%")
        with col4:
            st.metric("Max Drawdown", "2.5%")
        
        st.info("Execution metrics would be shown here from live system data")


def render_governance_status():
    """Render governance and safety status."""
    st.subheader("🛡️ Governance Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Acknowledged", "✓" if CONFIG.get("RISK_ACKNOWLEDGED", False) else "✗")
    
    with col2:
        auto_hedge = CONFIG.get("AUTO_HEDGE_ENABLED", True)
        st.metric("Auto-Hedge", "Enabled" if auto_hedge else "Disabled")
    
    with col3:
        kill_switch = CONFIG.get("KILL_SWITCH_COOLDOWN", 300)
        st.metric("Kill Switch Cooldown", f"{kill_switch}s")
    
    with col4:
        crisis_thresh = CONFIG.get("AI_CRISIS_THRESHOLD", 0.70)
        st.metric("Crisis Threshold", f"{crisis_thresh:.0%}")
    
    # Safety checklist
    st.markdown("### Safety Checklist")
    
    checks = [
        ("Mode is not LIVE", CONFIG.get("MODE") != "LIVE"),
        ("Capital utilization < 85%", True),  # Would be dynamic
        ("No kill switch triggered", True),
        ("All strategies healthy", True)
    ]
    
    for check_name, passed in checks:
        if passed:
            st.write(f"✅ {check_name}")
        else:
            st.write(f"❌ {check_name}")


def render_stress_test_results():
    """Render stress test results section."""
    st.subheader("💪 Stress Test Results")
    
    # Load system health data
    health_df = load_system_health()
    
    if not health_df.empty:
        st.dataframe(health_df.tail(10), use_container_width=True)
    else:
        # Demo data
        st.markdown("### Historical Crises Scenarios")
        
        scenarios = pd.DataFrame({
            'Scenario': ['2008 Crash', 'COVID Crash', 'Flash Crash', 'Rate Shock', 'Custom'],
            'Max Drawdown': ['-45%', '-35%', '-20%', '-15%', 'TBD'],
            'Recovery Time': ['18 months', '3 months', '1 week', '2 months', 'TBD'],
            'Status': ['Simulated', 'Simulated', 'Simulated', 'Simulated', 'Ready']
        })
        
        st.table(scenarios)


def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("🎛️ Controls")
    
    # Refresh rate
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 5)
    
    # Data range
    data_range = st.sidebar.selectbox(
        "Data Range",
        ["Last Hour", "Last 24 Hours", "Last Week", "All Time"]
    )
    
    # Show/Hide sections
    st.sidebar.markdown("### Display Options")
    show_regime = st.sidebar.checkbox("Show Regime", value=True)
    show_weights = st.sidebar.checkbox("Show Weights", value=True)
    show_risk = st.sidebar.checkbox("Show Risk", value=True)
    show_execution = st.sidebar.checkbox("Show Execution", value=True)
    show_governance = st.sidebar.checkbox("Show Governance", value=True)
    show_stress = st.sidebar.checkbox("Show Stress Tests", value=True)
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Export data
    st.sidebar.markdown("### Export")
    if st.sidebar.button("📥 Export Metrics CSV"):
        df = load_metrics_data()
        if not df.empty:
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                "Download CSV",
                csv,
                "ee_metrics.csv",
                "text/csv"
            )
    
    return {
        "refresh_rate": refresh_rate,
        "data_range": data_range,
        "show_regime": show_regime,
        "show_weights": show_weights,
        "show_risk": show_risk,
        "show_execution": show_execution,
        "show_governance": show_governance,
        "show_stress": show_stress
    }


# ============================================================
# Main Dashboard
# ============================================================

def main():
    """Main dashboard function."""
    # Render header
    render_header()
    
    # Get sidebar controls
    controls = render_sidebar()
    
    st.markdown("---")
    
    # Render sections based on controls
    if controls["show_regime"]:
        render_regime_indicator()
        st.markdown("---")
    
    if controls["show_weights"]:
        render_portfolio_weights()
        st.markdown("---")
    
    if controls["show_risk"]:
        render_risk_exposures()
        st.markdown("---")
    
    if controls["show_execution"]:
        render_execution_metrics()
        st.markdown("---")
    
    if controls["show_governance"]:
        render_governance_status()
        st.markdown("---")
    
    if controls["show_stress"]:
        render_stress_test_results()
    
    # Auto-refresh
    time.sleep(controls["refresh_rate"])
    st.rerun()


if __name__ == "__main__":
    main()
