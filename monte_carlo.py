import numpy as np
import pandas as pd
from simulation import run_simulation
from strategy import run_strategy

N_RUNS = 500
START_SEED = 0
STRATEGIES = ["fixed", "inventory", "avellaneda_stoikov"]
OUTPUT_CSV = "monte_carlo_results.csv"

def compute_metrics(history):
    """
    Compute performance metrics for a single strategy run.
    Mirrors the calculations in strategy.py so the numbers stay consistent.
    """
    pnl_series = pd.Series(history["pnl"])
    steps = pnl_series.diff().dropna()

    final_pnl = pnl_series.iloc[-1]
    max_dd = (pnl_series - pnl_series.cummax()).min()

    if steps.std() > 0:
        sharpe = (steps.mean() / steps.std()) * np.sqrt(252)
    else:
        sharpe = np.nan

    return {
        "final_pnl": final_pnl,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "n_trades": len(history["trades"]),
        "final_inventory": history["inventory"][-1]
    }

def run_monte_carlo(n_runs=N_RUNS, start_seed=START_SEED, verbose=True):
    """
    Runs every strategy across many independent simulations.

    Each seed produces one price path and one order flow, and all three
    strategies trade against that same market. Difference in Pnl therefore
    come from the strategies themselves rather than from one from one strategy drawing
    a friendlier market than another.
    """

    rows = []

    for i in range(n_runs):
        seed = start_seed + i
        sim = run_simulation(seed=seed, verbose=False)

        for name in STRATEGIES:
            history = run_strategy(name, sim)
            metrics = compute_metrics(history)
            metrics["strategy"] = name
            metrics["seed"] = seed
            rows.append(metrics)

        if verbose and (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n_runs} simulations complete")

    return pd.DataFrame(rows)

def summarize(df):
    """
    Aggregate per-run results into distribution statistics.
    The standard deviation matters as much as the mean - a large average
    profit with a larger standard deviation means the strategy is a coin flip.
    """

    summary = df.groupby("strategy").agg(
        mean_pnl = ("final_pnl", "mean"),
        std_pnl = ("final_pnl", "std"),
        median_pnl = ("final_pnl", "median"),
        worst_pnl = ("final_pnl", "min"),
        best_pnl = ("final_pnl", "max"),
        pct_profitable = ("final_pnl", lambda s: (s >0).mean() * 100),
        mean_sharpe = ("sharpe", "mean"),
        std_sharpe = ("sharpe", "std"),
        mean_drawdown = ("max_drawdown", "mean"),
        mean_trades = ("n_trades", "mean")
    )

    return summary.reindex(STRATEGIES)

def head_to_head(df):
    """
    Fraction of simulations where Avellaneda-Stoikov beat each baseline
    on the same price path. This is the number that settles whether the 
    strategy is better or where one run got lucky.
    """

    pivot = df.pivot(index="seed", columns="strategy", values="final_pnl")

    beats_fixed     = (pivot["avellaneda_stoikov"] > pivot["fixed"]).mean() * 100
    beats_inventory = (pivot["avellaneda_stoikov"] > pivot["inventory"]).mean() * 100
    beats_both      = ((pivot["avellaneda_stoikov"] > pivot["fixed"]) &
                       (pivot["avellaneda_stoikov"] > pivot["inventory"])).mean() * 100

    return {
        "beats_fixed": beats_fixed,
        "beats_inventory": beats_inventory,
        "beats_both": beats_both
    }