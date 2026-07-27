import numpy as np
import pandas as pd
from simulation import run_simulation

SPREAD = 0.10
MAX_INV = 500
INV_PENALTY = 0.01
TRANSACTION_COST = 0.01

def compute_quotes(mid_price, spread, inventory, inv_penalty=INV_PENALTY):
    inv_adjustment = inv_penalty * inventory
    bid = mid_price - spread / 2 - inv_adjustment
    ask = mid_price + spread / 2 - inv_adjustment
    return bid, ask


def fill_orders(orders, bid, ask, inventory, pnl, mid_price):
    trades = []

    for order in orders:
        if order == 1:
            if abs(inventory - 1) <= MAX_INV:
                trade_pnl = ask - mid_price - TRANSACTION_COST
                pnl += trade_pnl
                inventory -= 1
                trades.append({
                    "side": "sell",
                    "price": round(ask, 4),
                    "pnl_delta": round(trade_pnl, 4)
                })

        elif order == -1:
            if abs(inventory + 1) <= MAX_INV:
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
        bid, ask = compute_quotes(mid_price, spread, inventory, inv_penalty)
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
        history["spread_earned"].append(spread * len(trades))

    print(f"\nMarket maker complete.")
    print(f"Final inventory : {inventory} shares")
    print(f"Final PnL       : ${marked_pnl:.2f}")
    print(f"Total trades    : {len(history['trades'])}")
    print(f"Total spread earned : ${sum(history['spread_earned']):.2f}")

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