import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import pandas as pd
import json
import time
from simulation import run_simulation
from strategy import run_strategies, run_strategy
from strategy import fixed_spread_strategy, inventory_adjusted_strategy, avellaneda_stoikov_strategy
from market_maker import fill_orders
from analytics import (compute_sharpe, compute_max_drawdown,
                       compute_hit_rate, compute_spread_capture,
                       compute_inventory_risk)

app = Flask(__name__)


def get_metrics(history):
    pnl_series = history["pnl"]
    trades     = history["trades"]
    inventory  = history["inventory"]

    return {
        "final_pnl":       round(pnl_series[-1], 2),
        "annualized_pnl":  round(pd.Series(pnl_series).diff().mean() * 252, 2),
        "sharpe":          round(compute_sharpe(pnl_series), 2),
        "max_drawdown":    round(compute_max_drawdown(pnl_series), 2),
        "hit_rate":        round(compute_hit_rate(trades) * 100, 2),
        "spread_capture":  round(compute_spread_capture(trades), 4),
        "inventory_risk":  round(compute_inventory_risk(inventory), 1),
        "total_trades":    len(trades),
        "final_inventory": inventory[-1]
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data     = request.json
    strategy = data.get("strategy", "avellaneda_stoikov")
    spread   = float(data.get("spread", 0.10))
    gamma    = float(data.get("gamma", 0.1))
    seed     = int(data.get("seed", 42))

    sim    = run_simulation(seed=seed)
    prices = sim["prices"]

    history = run_strategy(strategy, sim, spread=spread, gamma=gamma)
    metrics = get_metrics(history)

    recent_trades = history["trades"][-10:]
    recent_trades = [
        {
            "side":      t["side"],
            "price":     round(t["price"], 2),
            "pnl_delta": round(t["pnl_delta"], 4)
        }
        for t in recent_trades
    ]

    return jsonify({
        "metrics":       metrics,
        "prices":        [round(p, 2) for p in prices[:500]],
        "bids":          [round(b, 2) for b in history["bids"][:500]],
        "asks":          [round(a, 2) for a in history["asks"][:500]],
        "pnl":           [round(p, 2) for p in history["pnl"]],
        "inventory":     history["inventory"],
        "recent_trades": recent_trades
    })


@app.route("/compare", methods=["GET"])
def compare():
    sim     = run_simulation(seed=42)
    results = {}

    for name in ["fixed", "inventory", "avellaneda_stoikov"]:
        history       = run_strategy(name, sim)
        results[name] = {
            "pnl":     [round(p, 2) for p in history["pnl"]],
            "metrics": get_metrics(history)
        }

    return jsonify(results)


@app.route("/stream", methods=["POST"])
def stream():
    data     = request.json
    strategy = data.get("strategy", "avellaneda_stoikov")
    spread   = float(data.get("spread", 0.10))
    gamma    = float(data.get("gamma", 0.1))
    seed     = int(data.get("seed", 42))
    speed    = int(data.get("speed", 20))

    sim        = run_simulation(seed=seed)
    prices     = sim["prices"]
    order_flow = sim["order_flow"]
    n_steps    = sim["n_steps"]
    sigma      = sim["sigma"]
    T          = 1.0

    inventory  = 0
    pnl        = 0.0

    def generate():
        nonlocal inventory, pnl

        for t in range(n_steps):
            mid_price = prices[t]
            time_now  = t / n_steps * T

            if strategy == "fixed":
                bid, ask = fixed_spread_strategy(mid_price, spread, inventory)
            elif strategy == "inventory":
                bid, ask = inventory_adjusted_strategy(mid_price, spread, inventory)
            else:
                bid, ask = avellaneda_stoikov_strategy(
                    mid_price, inventory, sigma, time_now, T, gamma
                )

            inventory, pnl, trades = fill_orders(
                order_flow[t], bid, ask, inventory, pnl, mid_price
            )

            marked_pnl = pnl + inventory * (mid_price - prices[0])

            frame = {
                "t":         t,
                "n_steps":   n_steps,
                "price":     round(mid_price, 2),
                "bid":       round(bid, 2),
                "ask":       round(ask, 2),
                "inventory": inventory,
                "pnl":       round(marked_pnl, 2),
                "trades":    [
                    {
                        "side":      tr["side"],
                        "price":     round(tr["price"], 2),
                        "pnl_delta": round(tr["pnl_delta"], 4)
                    }
                    for tr in trades
                ]
            }

            yield f"data: {json.dumps(frame)}\n\n"

            if t % speed == 0:
                time.sleep(0.05)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)