"""PageRank transition matrix with dangling nodes."""
import pathlib
import sys

# Allow running the example without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import dividebyzero as dbz


def main() -> None:
    # Simple web graph: node 2 has no outgoing links
    adjacency = dbz.array([
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
    ])

    out_degree = adjacency.sum(axis=1)
    # Broadcast out-degree across columns to avoid invalid indexing
    out_matrix = dbz.array([[d, d, d] for d in out_degree.array])
    transition = adjacency / out_matrix

    print("Adjacency matrix:\n", adjacency.array)
    print("Transition matrix:\n", transition.array)

    # Reconstruct transition probabilities for dangling nodes
    elevated = transition.elevate()
    print("Elevated transition matrix:\n", elevated.array)


if __name__ == "__main__":
    main()

