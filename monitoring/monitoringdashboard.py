import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

CSV_LOG = "logs/paper_trades.csv"

def start_dashboard():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    def update(_):
        if not os.path.exists(CSV_LOG):
            return

        df = pd.read_csv(CSV_LOG, parse_dates=["timestamp"])
        if df.empty:
            return

        ax1.cla(); ax2.cla(); ax3.cla()

        # Exposure
        ax1.plot(df["timestamp"], df["trend_exposure"], label="Trend")
        ax1.plot(df["timestamp"], df["mean_exposure"], label="Mean Reversion")
        ax1.set_title("Exposure Over Time")
        ax1.legend(); ax1.grid()

        # Hedge usage
        ax2.scatter(df["timestamp"], df["hedge_used"], marker="|", s=200)
        ax2.set_title("Hedge Usage")
        ax2.grid()

        # Capital flight
        ax3.scatter(df["timestamp"], df["capital_flight"], marker="|", s=200, color="red")
        ax3.set_title("Capital Flight Triggers")
        ax3.grid()

    FuncAnimation(fig, update, interval=5000)
    plt.tight_layout()
    plt.show()
