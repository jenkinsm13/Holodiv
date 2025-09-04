"""Demonstrate NumPy compatibility of DivideByZero."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    a = dbz.array([[1, 2], [3, 4]])
    b = dbz.array([[2, 0], [0, 2]])

    # Use NumPy-style operations through the dbz namespace
    c = dbz.matmul(a, b)
    print("Matrix product:\n", c.array)

    # Apply element-wise functions
    s = dbz.sin(a)
    print("Sine of array:\n", s.array)

    # Safe division by zero with reconstruction
    reduced = b / 0
    print("Reduced after division by zero:\n", reduced.array)
    print("Elevated back:\n", reduced.elevate().array)

    # Example of a linear algebra routine
    logm_a = dbz.linalg.logm(a)
    print("Matrix logarithm:\n", logm_a.array)


if __name__ == "__main__":
    main()
