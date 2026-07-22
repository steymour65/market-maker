import numpy as np
import pandas as pd

SEED = 42
N_STEPS = 1000
DT = 1/252
S0 = 100.0
MU = 0.05
SIGMA = 0.20
LAMBDA = 10.0

def generate_price_path(n_steps=N_STEPS, dt=DT, s0=S0, mu=MU, sigma=SIGMA, seed=SEED):
    """
    Generates a stock price path using Geometric Brownian Motion.

    Formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z is a standard normal random variable.

    This is the same process assumed by Black-Scholes. It ensures:
    - Prices are always positive
    - Log returns are normally distributed
    - Volatility scales with sqrt(time)
    """
    np.random.seed(seed)

    prices = np.zeros(n_steps)
    prices[0] = s0

    for t in range(1, n_steps):
        Z = np.random.standard_normal()
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return prices


def generate_order_flow(n_steps=N_STEPS, lam=LAMBDA, seed=SEED):
    """
    Generates random order arrivals using a Poisson process.

    At each time step:
    - Number of orders arriving follows Poisson(lambda)
    - Each order is a market buy (+1) or market sell (-1) with equal probability

    Poisson process is the standard model for order arrival in market microstructure
    because orders arrive randomly but at a predictable average rate.
    """
    np.random.seed(seed + 1)

    order_flow = []

    for t in range(n_steps):
        n_orders = np.random.poisson(lam)
        orders = np.random.choice([-1, 1], size=n_orders)
        order_flow.append(orders)

    return order_flow


def run_simulation(n_steps=N_STEPS, dt=DT, s0=S0, mu=MU, sigma=SIGMA, lam=LAMBDA, seed=SEED):
    """
    Master function that runs the full simulation.
    Returns prices and order flow packaged as a dictionary.
    """
    print("Generating price path...")
    prices = generate_price_path(n_steps, dt, s0, mu, sigma, seed)

    print("Generating order flow...")
    order_flow = generate_order_flow(n_steps, lam, seed)

    print(f"\nSimulation complete.")
    print(f"Steps      : {n_steps}")
    print(f"Start price: ${prices[0]:.2f}")
    print(f"End price  : ${prices[-1]:.2f}")
    print(f"Min price  : ${prices.min():.2f}")
    print(f"Max price  : ${prices.max():.2f}")
    print(f"Total orders: {sum(len(o) for o in order_flow)}")

    return {
        "prices":     prices,
        "order_flow": order_flow,
        "n_steps":    n_steps,
        "dt":         dt,
        "sigma":      sigma
    }


if __name__ == "__main__":
    sim = run_simulation()

    prices = sim["prices"]
    order_flow = sim["order_flow"]

    print("\n--- First 10 prices ---")
    print([round(p, 2) for p in prices[:10]])

    print("\n--- First 5 order flow steps ---")
    for i, orders in enumerate(order_flow[:5]):
        print(f"Step {i}: {len(orders)} orders -> {orders}")