from mayavi import mlab
import matplotlib.pyplot as plt
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


def plot_snapshot(pos: np.ndarray, distance: float = 40):
    mlab.figure(size=(800, 700))
    p = mlab.points3d(
        pos[:, 0], pos[:, 1], pos[:, 2], np.ones(shape=(len(pos),)), scale_factor=1
    )
    mlab.outline()
    mlab.orientation_axes()
    mlab.view(focalpoint=[0, 0, 0], distance=distance)
    return p


def plot_samples(u: np.ndarray, label: str):
    _, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axs[0].plot(u)
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel(label)

    # add a 'best fit' line to the histogram
    u = np.sort(u)
    mu, sigma = u.mean(), u.std()
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * (1 / sigma * (np.sort(u) - mu)) ** 2
    )

    textstr = "\n".join([r"$\mu=%.2f$" % (mu,), r"$\sigma=%.2f$" % (sigma,)])
    props = dict(boxstyle="round", facecolor="w", alpha=0.5)

    axs[1].hist(u, density=True)
    axs[1].plot(u, y, "--")
    axs[1].set_xlabel(label)
    # place a text box in upper left in axes coords
    axs[1].text(
        0.05,
        0.95,
        textstr,
        transform=axs[1].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    plt.show()
