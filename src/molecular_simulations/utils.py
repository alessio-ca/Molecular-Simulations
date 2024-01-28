import numpy as np
from numba import njit


def initialise_lattice(N: int, L: float) -> np.ndarray:
    """Initialise particles on a cubic lattice with side L

    Args:
        N (int): number of particles
        L (float): lattice size

    Returns:
        np.ndarray: array of initial positions
    """
    positions = np.zeros(shape=(N, 3))
    n_per_side = np.ceil(N ** (1 / 3))
    d = L / n_per_side
    for i in range(N):
        positions[i] = [
            (i % n_per_side),
            (i // n_per_side) % (n_per_side),
            (i // (n_per_side**2)),
        ]
        positions[i] *= d
    return positions


def initialise_velocities(N: int, T: float) -> np.ndarray:
    """Initialise velocities (random).
    Shift and rescale to satisfy mean T and 0-momentum

    Args:
        N (int): number of particles

    Returns:
        np.ndarray: array of initial velocities.
    """
    velocities = np.random.random((N, 3)) - 0.5

    cm_v = velocities.mean(axis=0)
    cm_v2 = (velocities**2).sum() / N
    scale = np.sqrt(3 * T / cm_v2)
    return (velocities - cm_v) * scale


@njit
def apply_bc(X: np.ndarray, L: float) -> np.ndarray:
    return X - L * np.rint(X / L)


@njit
def dist(positions: np.ndarray, particle: np.ndarray, L: float) -> np.ndarray:
    # Matrix of distances between all positions and particle
    # Apply image convention
    delta_pos = apply_bc(particle - positions, L)
    # Calculate euclidean squared distance
    return delta_pos, (delta_pos**2).sum(axis=1)


def writeXYZ(trajectory: np.ndarray, filename: str, atomname: str = "Ar"):
    f = open(filename, "w")

    if len(trajectory.shape) == 2:
        # This is a snapshot, add the third axis
        trajectory = trajectory[None, :]

    N = str(trajectory.shape[1])
    for step in trajectory:
        f.write(N + "\n\n")
        for mol in step:
            f.write(atomname + "\t")
            for dim in mol:
                f.write("%.4f    " % dim)
            f.write("\n")
    f.close()
