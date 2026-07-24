import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import run_simulation
from strategy import run_strategies

TRADING_DAYS = 252


def compute_sharpe(pnl_series):
    daily_returns = pd.Series(pnl_series).diff().dropna()
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    
    return (daily_returns.mean() / std) * np.sqrt(TRADING_DAYS)


def compute_max_drawdown(pnl_series):
    pnl = pd.Series(pnl_series)
    rolling_max = pnl.cummax()
    drawdown = pnl - rolling_max
    return drawdown.min()


def compute_hit_rate(trades):
    if len(trades) == 0:
        return 0.0
    winning = sum(1 for t in trades if t["pnl_delta"] > 0)
    return winning / len(trades)


def compute_spread_capture(trades):
    if len(trades) == 0:
        return 0.0
    return np.mean([abs(t["pnl_delta"]) for t in trades])


def compute_inventory_risk(inventory_series):
    inv = pd.Series([x for x in inventory_series if isinstance(x, (int, float))])
    if inv.empty:
        return 0.0
    return inv.std()


def plot_results(results, prices):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle("Market Making Strategy Comparison", fontsize=14)


    colors = {
        "fixed": "#888888",
        "inventory": "#60a5fa",
        "avellaneda_stoikov": "#4ade80",
    }

    axes[0].plot(prices, color="white", linewidth=1.5, label="Mid Price")
    axes[0].plot(results["avellaneda_stoikov"]["bids"], color="#4ade80",
                 linewidth=0.5, alpha=0.5, label="AS Bid")
    axes[0].plot(results["avellaneda_stoikov"]["asks"], color="#f87171",
                 linewidth=0.5, alpha=0.5, label="AS Ask")
    axes[0].set_title("Price with Avellaneda-Stoikov Quotes")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(fontsize=8)

    for name, history in results.items():
        axes[1].plot(history["pnl"], color=colors[name], linewidth=1.5, label=name)
    axes[1].axhline(y=0, color="white", linestyle="--", linewidth=0.8)
    axes[1].set_title("Cumulative PnL — All Strategies")
    axes[1].set_ylabel("PnL ($)")
    axes[1].legend(fontsize=8)

    for name, history in results.items():
        axes[2].plot(history["inventory"], color=colors[name], linewidth=1.0, label=name)
    axes[2].axhline(y=0, color="white", linestyle="--", linewidth=0.8)
    axes[2].set_title("Inventory Over Time")
    axes[2].set_ylabel("Shares")
    axes[2].legend(fontsize=8)

    for name, history in results.items():
        trade_pnls = [t["pnl_delta"] for t in history["trades"]]
        axes[3].hist(trade_pnls, bins=50, alpha=0.5, color=colors[name], label=name)
    axes[3].axvline(x=0, color="white", linestyle="--", linewidth=0.8)
    axes[3].set_title("Trade PnL Distribution")
    axes[3].set_xlabel("PnL per Trade ($)")
    axes[3].set_ylabel("Count")
    axes[3].legend(fontsize=8)

    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#888")
        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("#2a2d3a")

    plt.tight_layout()
    plt.savefig("performance.png", dpi=150, facecolor="#0f1117")
    print("Chart saved to performance.png")
    plt.show()


def run_analytics(results, prices):
    print("\n" + "=" * 55)
    print("     DETAILED PERFORMANCE METRICS")
    print("=" * 55)

    for name, history in results.items():
        pnl_series = history["pnl"]
        trades = history["trades"]
        inventory = history["inventory"] 

        sharpe = compute_sharpe(pnl_series)
        max_dd = compute_max_drawdown(pnl_series)
        hit_rate = compute_hit_rate(trades)
        spread_cap = compute_spread_capture(trades)
        inv_risk = compute_inventory_risk(inventory)
        final_pnl = pnl_series[-1]
        annualized = pd.Series(pnl_series).diff().mean() * TRADING_DAYS

        print(f"\n{name.upper()}")
        print(f"  Final PnL          : ${final_pnl:.2f}")
        print(f"  Annualized PnL     : ${annualized:.2f}")
        print(f"  Sharpe Ratio       : {sharpe:.2f}")
        print(f"  Max Drawdown       : ${max_dd:.2f}")
        print(f"  Hit Rate           : {hit_rate:.2%}")
        print(f"  Avg Spread Capture : ${spread_cap:.4f}")
        print(f"  Inventory Risk     : {inv_risk:.1f} shares (std)")
        print(f"  Total Trades       : {len(trades)}")
        print(f"  Final Inventory    : {inventory[-1]} shares")

    print("=" * 55)

    plot_results(results, prices)


if __name__ == "__main__":
    results = run_strategies()


    if "prices" in results:
        prices = results.pop("prices")
    else:

        print("WARNING: falling back to a fresh run_simulation() price path; "
              "return 'prices' from run_strategies() for correct plots.")
        prices = run_simulation()["prices"]

    run_analytics(results, prices)
