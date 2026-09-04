from simulation import run_simulation
from strategy import run_strategies
from analytics import run_analytics

def main():
    print("="*55)
    print("    MARKET MAKING SIMULATOR")
    print("="*55)

    sim = run_simulation()

    print("\nRunning all three strategies...")
    results = run_strategies(sim)

    print("\nRunning analytics...")
    run_analytics(results, sim["prices"])

    print("\n" + "="*55)
    print(" SIMULATION COMPLETE")
    print(" Chart saved to performance.png")
    print("="*55)

if __name__ == "__main__":
    main()