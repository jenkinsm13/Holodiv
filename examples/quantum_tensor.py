"""Quantum tensor dimensional reduction example."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    # Construct a simple 3-qubit state
    data = [[[1, 0], [0, 1]],
            [[0, 1], [1, 0]]]
    qt = dbz.quantum.QuantumTensor(data, physical_dims=(2, 2, 2))

    print("Original tensor shape:", qt.data.shape)

    # Reduce tensor dimensions while preserving entanglement
    reduced = qt.reduce_dimension(target_dims=2)
    print("Reduced tensor shape:", reduced.data.shape)

    # Elevate back to demonstrate reconstruction
    elevated = reduced.elevate()
    print("Elevated tensor shape:", elevated.data.shape)


if __name__ == "__main__":
    main()
