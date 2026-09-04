import numpy as np
import pandas as pd
from simulation import run_simulation
from market_maker import fill_orders

SPREAD = 0.10
MAX_INV = 500
GAMMA = 0.1
KAPPA = 1.5

def fixed_spread_strategy(mid_price, spread, inventory):
    """
    Simplest possible strategy - always quote the same fixed spread.
    No inventory adjustment, no market awareness.
    This is the baseline to compare against.
    """
    bid = mid_price - spread / 2
    ask = mid_price + spread / 2
    return bid, ask

def inventory_adjusted_strategy(mid_price, spread, inventory, inv_penalty=0.01):
    """
    Adjusts quotes based on current inventory.
    If long: shift quotes down to attract sellers and discourage buyers
    If short: shift quotes up to attract buyers and discourage sellers
    Also widens the spread proportionally to inventory size.
    """
    inv_adjustment = inv_penalty * inventory
    inv_spread_widening = abs(inventory) * 0.001

    bid = mid_price - (spread + inv_spread_widening) / 2 - inv_adjustment
    ask = mid_price + (spread + inv_spread_widening) / 2 - inv_adjustment

    return bid, ask

def avellaneda_stoikov_strategy(mid_price, inventory, sigma, t, T, gamma=GAMMA, kappa=KAPPA):
    """
    Optimal market making strategy from Avellaneda & Stoikov (2008).

    The strategy solves an optimization problem to find the reservation price
    and optimal spread given:
    - Current inventory (q)
    - Time remaining (T - t)
    - Market volatility (sigma)
    - Risk aversion (gamma)
    - Order arrival rate (kappa)

    Reservation price: r = mid - q * gamma * sigma^2 * (T-t)
    Optimal spread: delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

    The reservation price shifts based on inventory - if long, r drops below mid
    to attract sellers and discourage further buying. The spread widens as
    volatility or time remaining increases.
    """
    time_remaining = T - t

    reservation_price = mid_price - inventory * gamma * sigma**2 * time_remaining

    optimal_spread = (gamma * sigma**2 * time_remaining + (2 / gamma) * np.log(1 + gamma / kappa))

    bid = reservation_price - optimal_spread / 2
    ask = reservation_price + optimal_spread / 2

    return bid, ask

def run_strategy(strategy_name, sim, spread=SPREAD, gamma=GAMMA, kappa=KAPPA):
    """
    Runs a single strategy on the simulated price path.
    Returns the full history of prices, quotes, inventory, and PnL.

    The time horizon is derived from the simulation so that t, T, and sigma
    are all measured in years.
    """
    prices = sim["prices"]
    order_flow = sim["order_flow"]
    n_steps = sim["n_steps"]
    sigma = sim["sigma"]
    dt = sim["dt"]

    T = n_steps * dt

    inventory = 0
    pnl = 0.0

    history = {
        "prices": [],
        "bids": [],
        "asks": [],
        "inventory": [],
        "pnl": [],
        "trades": []
    }

    for t in range(n_steps):
        mid_price = prices[t]
        time = t * dt

        if strategy_name == "fixed":
            bid, ask = fixed_spread_strategy(mid_price, spread, inventory)

        elif strategy_name == "inventory":
            bid, ask = inventory_adjusted_strategy(mid_price, spread, inventory)

        elif strategy_name == "avellaneda_stoikov":
            bid, ask = avellaneda_stoikov_strategy(
                mid_price, inventory, sigma, time, T, gamma, kappa
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        inventory, pnl, trades = fill_orders(
            order_flow[t], bid, ask, inventory, pnl, mid_price
        )

        marked_pnl = pnl + inventory * (mid_price - prices[0])

        history["prices"].append(mid_price)
        history["bids"].append(bid)
        history["asks"].append(ask)
        history["inventory"].append(inventory)
        history["pnl"].append(marked_pnl)
        history["trades"].extend(trades)

    return history

def run_strategies(sim=None, verbose=True):
    """
    Runs all three strategies on the same simulation and compares results.
    Accepts an existing simulation so callers can control the seed.
    """
    if sim is None:
        sim = run_simulation()

    strategies = ["fixed", "inventory", "avellaneda_stoikov"]
    results = {}

    for strategy in strategies:
        if verbose:
            print(f"Running {strategy} strategy...")
        results[strategy] = run_strategy(strategy, sim)

    if verbose:
        print("\n" + "="*50)
        print(" STRATEGY COMPARISON")
        print("="*50)

        for name, history in results.items():
            pnl_series = pd.Series(history["pnl"])
            final_pnl = pnl_series.iloc[-1]
            max_dd = (pnl_series - pnl_series.cummax()).min()
            sharpe = (pnl_series.diff().mean() / pnl_series.diff().std()) * np.sqrt(252)
            n_trades = len(history["trades"])

            print(f"\n{name.upper()}")
            print(f" Final PnL: ${final_pnl:.2f}")
            print(f" Max Drawdown : ${max_dd:.2f}")
            print(f" Sharpe Ratio : {sharpe:.2f}")
            print(f" Total Trades: {n_trades}")
            print(f" Final Inventory: {history['inventory'][-1]} shares")

        print("="*50)

    return results

if __name__ == "__main__":
    results = run_strategies()