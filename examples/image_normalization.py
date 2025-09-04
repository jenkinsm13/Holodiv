"""Image normalization with zero-valued illumination."""
import pathlib
import sys

# Allow running the example without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    # Grayscale image and illumination map with dark pixels
    image = dbz.array([
        [0, 50, 100],
        [150, 200, 0],
    ])
    illumination = dbz.array([
        [1, 0, 2],
        [3, 4, 0],
    ])

    normalized = image / illumination

    print("Image:\n", image.array)
    print("Illumination:\n", illumination.array)
    print("Normalized image:\n", normalized.array)

    # Recover full information for pixels where illumination was zero
    elevated = normalized.elevate()
    print("Elevated normalized image:\n", elevated.array)


if __name__ == "__main__":
    main()

