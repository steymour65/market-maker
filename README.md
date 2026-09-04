# Market Making Simulator

A quantitative market making simulator that implements and compares three algorithmic quoting strategies — Fixed Spread, Inventory-Adjusted, and the Avellaneda-Stoikov optimal model — against a simulated order book. Built in Python with a real-time Flask dashboard.

## What market making is

A market maker quotes a bid and an ask simultaneously, standing ready to buy from sellers and sell to buyers. The profit comes from the spread between those two prices. The risk is inventory: every fill leaves the maker holding a position, and if the price moves against that position, the accumulated inventory loss can easily exceed the spread earned.

The central tension is that quoting tightly attracts fills but earns little per trade, while quoting widely earns more per trade but trades rarely. A market maker who quotes too wide simply doesn't trade. One who quotes too tight accumulates inventory faster than the spread compensates for. Everything in this project is about that tradeoff.

## The three strategies

**Fixed Spread.** Always quotes a constant $0.10 spread centered on the mid price, with no reference to the current position. This is the naive baseline — it fills constantly and accumulates inventory without any mechanism to shed it.

**Inventory-Adjusted.** Shifts both quotes in the direction that unwinds the current position: when long, quotes move down to attract sellers and discourage buyers; when short, they move up. The spread also widens in proportion to position size. This is a heuristic — it responds to inventory but has no notion of volatility or time.

**Avellaneda-Stoikov.** The optimal solution from Avellaneda & Stoikov (2008). It computes a reservation price — the price at which the maker is indifferent to holding the current position — and quotes symmetrically around that rather than around the mid.

reservation price: r = S - q * gamma * sigma^2 * (T - t)
optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)


where `S` is the mid price, `q` is inventory, `gamma` is risk aversion, `sigma` is volatility, `kappa` is the order arrival decay rate, and `T - t` is time remaining. The reservation price does the inventory hedging: holding a long position pushes `r` below mid, which makes the maker more eager to sell and less eager to buy. The spread term widens with volatility and with time remaining, since both increase the risk of holding a position.

## Fill model

Orders do not fill unconditionally. Following the same assumption Avellaneda-Stoikov is derived under, fill intensity decays exponentially with distance from mid:

P(fill) = exp(-kappa * delta)


where `delta` is how far the quote sits from the mid price. A quote at mid fills with certainty; a quote $1.00 away at `kappa = 1.5` fills about 22% of the time. This is what gives the model something to optimize — without it, the widest quote always wins, and the inventory skew has no effect on the position.

## Results

Single simulation, seed 42, 1000 steps:

| Strategy | Final PnL | Sharpe | Max Drawdown | Trades | Fill Rate | Final Inventory |
|---|---|---|---|---|---|---|
| Fixed Spread | −$780.75 | −0.17 | −$3,051 | 9,275 | 93% | −27 |
| Inventory-Adjusted | +$641.65 | 0.18 | −$927 | 9,177 | 92% | +7 |
| Avellaneda-Stoikov | +$2,077.15 | 0.90 | −$593 | 3,806 | 38% | −8 |

The headline is not the PnL ranking but the drawdown column. Avellaneda-Stoikov reduced maximum drawdown by 81% relative to the fixed baseline while trading 59% less. It achieves this by quoting wider and skewing aggressively against inventory — it passes on most order flow and takes only the trades that pay for the risk.

The fixed spread strategy demonstrates the failure mode directly: it captured $0.10 per fill across 9,275 fills but ended in a loss, because inventory accumulated faster than the spread compensated. Inventory-Adjusted sits in between, as expected — it skews against position but ignores volatility and time horizon.

**Caveat:** these are single-path results. One realization of a random price process is weak evidence about a strategy. `monte_carlo.py` runs the comparison across many independent simulations and reports distributions rather than point estimates.

## Simulation parameters

| Parameter | Value | Description |
|---|---|---|
| Initial price | $100 | Starting price |
| Annual drift | 5% | GBM mu |
| Annual volatility | 20% | GBM sigma |
| Order arrival rate | 10/step | Poisson lambda |
| Time steps | 1000 | Simulation length |
| Fixed spread | $0.10 | Baseline quote width |
| Transaction cost | $0.01/share | Cost per fill |
| Max inventory | 500 shares | Hard position limit |
| Gamma | 0.1 | A-S risk aversion |
| Kappa | 1.5 | A-S order arrival decay |

## Project structure

market-maker/
simulation.py # Geometric Brownian Motion price path + Poisson order flow
market_maker.py # Probabilistic fill logic, inventory tracking, PnL accounting
strategy.py # Three quoting strategies and the comparison harness
analytics.py # Sharpe, drawdown, hit rate, spread capture
monte_carlo.py # Runs the comparison across many seeds
main.py # Single-simulation run with charts
app.py # Flask dashboard with real-time animation
templates/
index.html # Live charts


## Running it

```bash
git clone https://github.com/steymour65/market-maker.git
cd market-maker
python -m venv market-env
source market-env/Scripts/activate
pip install -r requirements.txt
python main.py
```

For the distribution study:

```bash
python monte_carlo.py
```

For the dashboard:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` and press **Animate** to watch the simulation run.

## What this doesn't model

Worth stating plainly, since these are the assumptions that separate a simulator from a trading system:

- **No adverse selection.** Order flow is independent of the price path, so the maker never systematically trades against better-informed counterparties. In real markets this is the primary source of market maker losses.
- **No queue position or discrete order book.** Fills are drawn from a probability, not matched against a limit order book with priority rules.
- **No latency.** Quotes update instantaneously at every step.
- **Circular fill assumption.** The fill model uses the same exponential decay and the same `kappa` that Avellaneda-Stoikov assumes internally, so the strategy is being evaluated in exactly the world it was derived for. This flatters it. Testing under a mismatched fill process would be a more demanding evaluation.
- **Geometric Brownian Motion.** Real returns have fat tails, volatility clustering, and jumps that GBM does not produce.

## References

Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.