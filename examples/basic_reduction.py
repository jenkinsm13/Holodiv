"""Basic dimensional reduction demonstration."""
import pathlib
import sys

# Allow running the example without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    # Create a 2D array and divide by zero to trigger dimensional reduction
    arr = dbz.array([[1, 2, 3], [4, 5, 6]])
    reduced = arr / 0

    print("Original shape:", arr.shape)
    print("Reduced representation:", reduced.array)

    # Reconstruct the original dimensions
    reconstructed = reduced.elevate()
    print("Reconstructed shape:", reconstructed.shape)
    print("Reconstructed array:\n", reconstructed.array)

    # Inspect the stored error information
    registry = dbz.get_registry()
    error = registry.retrieve(reduced._error_id)
    print("Stored error reduction type:", error.reduction_type)
    print("Error tensor shape:", error.error_tensor.shape)


if __name__ == "__main__":
    main()
