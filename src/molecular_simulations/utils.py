import numpy as np
from numba import njit
from typing import Tuple


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


class Accumulator:
    def __init__(self, steps: int) -> None:
        self.block_n = 2
        if steps < 2:
            self.max_block_operations_ = 1
        else:
            self.max_block_operations_ = (
                int(np.floor(np.emath.logn(self.block_n, steps))) + 1
            )
        # Floating point can hit hard on logn
        if self.block_n**self.max_block_operations_ == steps:
            self.max_block_operations_ += 1
        self.powers = np.array([0, 1, 2], dtype=int)
        self.reset()

    def reset(self):
        self.block_averages_ = np.zeros(
            shape=(self.max_block_operations_, 3), dtype=float
        )
        self.block_sums_ = np.zeros(shape=(self.max_block_operations_,), dtype=float)
        self.block_size_ = np.array(
            [self.block_n**i for i in range(self.max_block_operations_)], dtype=int
        )

    def _update_averages(self, value: float, i: int = 0):
        # Update mean and std of the block on the fly
        delta0 = value - self.block_averages_[i, 1]
        self.block_averages_[i, 0] += 1
        self.block_averages_[i, 1] += delta0 / self.block_averages_[i, 0]
        delta1 = value - self.block_averages_[i, 1]
        self.block_averages_[i, 2] += delta1 * delta0
        self.block_sums_[i] += value

    def accumulate(self, value: float):
        # Update averages & value of the lowest block
        self._update_averages(value)
        # Accumulate higher blocks with the new value
        update_i = np.argmax(self.count() % self.block_size_[::-1] == 0)
        for i in range(1, self.max_block_operations_ - update_i):
            # The new value is the averaged cumulative sum of the previous level
            value = self.block_sums_[i - 1] / self.block_n
            # Reset block sum
            self.block_sums_[i - 1] = 0
            self._update_averages(value, i)

    def count(self) -> int:
        return self.block_averages_[0, 0]

    def mean(self) -> float:
        return self.block_averages_[0, 1]

    def stdev(self) -> float:
        return np.sqrt(self.block_averages_[0, 2] / (self.block_averages_[0, 0] - 1))

    def _block_means(self) -> np.ndarray:
        # They should all be equal
        return self.block_averages_[:, 1]

    def _block_vars(self) -> np.ndarray:
        # They should all be equal
        return self.block_averages_[:-1, 2] / (
            self.block_averages_[:-1, 0] - 1
        )  # variance definition

    def block_vars_est(self) -> Tuple[np.ndarray, np.ndarray]:
        # The last block only contains a single element by definition
        # Therefore no variance can be estimated
        var_est = self._block_vars() / (self.block_averages_[:-1, 0] - 1)
        error = np.sqrt(2 * var_est**2 / (self.block_averages_[:-1, 0] - 1))
        # Return the variance estimate (for the ensemble average) and the error
        return var_est, error

    def uncorr_stdev(self) -> float:
        # Find the plateau (optimal block to report stdev)
        var, error = self.block_vars_est()
        arr_min = np.maximum.accumulate(var[::-1] - error[::-1])[::-1]
        arr_max = np.minimum.accumulate(var[::-1] + error[::-1])[::-1]

        opti = np.argmax((var > arr_min) & (var < arr_max))

        # Return stdev and factor sum for opti block
        if opti == len(self.block_averages_) - 1:
            print("Stdev of blocks did not plateau.")
        return var[opti], error[opti]
