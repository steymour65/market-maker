import numpy as np
import pandas as pd
from simulation import run_simulation

SPREAD = 0.10
MAX_INV = 500
INV_PENALTY = 0.01
TRANSACTION_COST = 0.01
KAPPA = 1.5


def fill_orders(orders, bid, ask, inventory, pnl, mid_price, kappa=KAPPA, rng=None):
    """
    Fills arriving market orders probabilistically.

    Avellaneda-Stoikov assumes fill intensity decays exponentially with distance
    from mid: an order hits your quote with probability exp(-kappa * delta), where
    delta is how far that quote sits from the mid price. Quote tight and you fill
    often but earn little; quote wide and you earn more per fill but trade rarely.
    This tension is what the model optimizes, and it is what makes inventory
    skewing effective - shifting your bid away from mid genuinely reduces the
    chance you buy more.

    PnL is booked as edge relative to mid, so a fill at mid contributes nothing.
    This means pnl is already marked to market and needs no further adjustment.
    """
    if rng is None:
        rng = np.random

    trades = []

    for order in orders:
        if order == 1:
            delta = max(ask - mid_price, 0.0)
            if rng.random() < np.exp(-kappa * delta) and abs(inventory - 1) <= MAX_INV:
                trade_pnl = ask - mid_price - TRANSACTION_COST
                pnl += trade_pnl
                inventory -= 1
                trades.append({
                    "side": "sell",
                    "price": round(ask, 4),
                    "pnl_delta": round(trade_pnl, 4)
                })

        elif order == -1:
            delta = max(mid_price - bid, 0.0)
            if rng.random() < np.exp(-kappa * delta) and abs(inventory + 1) <= MAX_INV:
                trade_pnl = mid_price - bid - TRANSACTION_COST
                pnl += trade_pnl
                inventory += 1
                trades.append({
                    "side": "buy",
                    "price": round(bid, 4),
                    "pnl_delta": round(trade_pnl, 4)
                })

    return inventory, pnl, trades


def run_market_maker(spread=SPREAD, inv_penalty=INV_PENALTY, seed=42):
    """
    Standalone runner for a single inventory-adjusted market maker.
    The strategy comparison lives in strategy.py - this exists for quick
    testing of the fill logic in isolation.
    """
    sim = run_simulation(seed=seed)
    prices = sim["prices"]
    order_flow = sim["order_flow"]
    n_steps = sim["n_steps"]

    inventory = 0
    pnl = 0.0

    history = {
        "prices": [],
        "bids": [],
        "asks": [],
        "inventory": [],
        "pnl": [],
        "trades": [],
        "spread_earned": []
    }

    print("Running market maker...")

    for t in range(n_steps):
        mid_price = prices[t]

        inv_adjustment = inv_penalty * inventory
        bid = mid_price - spread / 2 - inv_adjustment
        ask = mid_price + spread / 2 - inv_adjustment

        inventory, pnl, trades = fill_orders(
            order_flow[t], bid, ask, inventory, pnl, mid_price
        )
        marked_pnl = pnl

        history["prices"].append(mid_price)
        history["bids"].append(bid)
        history["asks"].append(ask)
        history["inventory"].append(inventory)
        history["pnl"].append(marked_pnl)
        history["trades"].extend(trades)
        history["spread_earned"].append(spread * len(trades))

    print(f"\nMarket maker complete.")
    print(f"Final inventory : {inventory} shares")
    print(f"Final PnL       : ${marked_pnl:.2f}")
    print(f"Total trades    : {len(history['trades'])}")
    print(f"Fill rate       : {len(history['trades']) / sum(len(o) for o in order_flow) * 100:.1f}%")

    return history


if __name__ == "__main__":
    history = run_market_maker()

    print("\n--- PnL Summary ---")
    pnl_series = pd.Series(history["pnl"])
    print(f"Max PnL   : ${pnl_series.max():.2f}")
    print(f"Min PnL   : ${pnl_series.min():.2f}")
    print(f"Final PnL : ${pnl_series.iloc[-1]:.2f}")

    print("\n--- Inventory Summary ---")
    inv_series = pd.Series(history["inventory"])
    print(f"Max inventory : {inv_series.max()} shares")
    print(f"Min inventory : {inv_series.min()} shares")