"""Financial return calculation with zero-price days."""
import pathlib
import sys

# Allow running the example without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    # Simulated daily closing prices with a trading halt day at zero
    prices = dbz.array([100, 105, 0, 110])
    returns = (prices[1:] - prices[:-1]) / prices[:-1]

    print("Prices:", prices.array)
    print("Daily returns:", returns.array)

    # Reconstruct information lost during the zero-price division
    reconstructed = returns.elevate()
    print("Reconstructed returns:", reconstructed.array)

    registry = dbz.get_registry()
    error = registry.retrieve(returns._error_id)
    print("Stored reduction type:", error.reduction_type)


if __name__ == "__main__":
    main()

