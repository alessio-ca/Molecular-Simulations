from typing import Tuple, Callable, Union
from numba.experimental import jitclass
from numba import int64, int32, float32
import numpy as np


def has_method(o, name):
    return callable(getattr(o, name, None))


class Accumulator:
    """Accumulator sampler.

    Accumulates values on the fly.
    Performs mean and var estimates on the fly.
    Performs block-analysis to estimate uncorrelated variance.

    References:
     - H. Flyvbjerg and H. G. Petersen, "Error estimates on correlated
        data", J. Chem. Phys. 91, 461--466 (1989)
     - Welford, B. P., "Note on a method for calculating corrected sums
       of squares and products". Technometrics. 4 (3): 419--420 (1962)
    """

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

    def uncorr_stdev(self) -> Tuple[float, float]:
        # Find the plateau (optimal block to report stdev)
        var, error = self.block_vars_est()
        opti = np.argmin(np.abs(np.gradient(np.gradient(var))))

        # Return stdev and factor sum for opti block
        if opti == (len(self.block_averages_) - 1) or opti == 0:
            print("Stdev of blocks did not plateau.")
        return var[opti], error[opti]


class Correlator:
    """Correlator ensemble sampler.

    Each value fed to the correlator is a N-dimensional vector
    where N is the number of ensemble members.

    Accumulates values on the fly.
    Uses a 2-16 correlator ladder to coarse-grain values
    at increasingly large time intervals.

    References:
     - J. Ramirez et al., "Efficient on the fly calculation of time correlation
       functions in computer simulations", J. Chem. Phys. 133, 154103 (2010)
    """

    def __init__(self, steps: int, N: int = 1) -> None:
        self.N = N
        self.block_p = 16
        self.block_m = 2
        self.ratio = int(self.block_p / self.block_m)
        if steps < self.block_m:
            self.max_block_operations_ = 1
        else:
            self.max_block_operations_ = (
                int(np.floor(np.emath.logn(self.block_m, steps))) + 1
            )
        # Floating point can hit hard on logn
        if self.block_m**self.max_block_operations_ == steps:
            self.max_block_operations_ += 1
        self.powers = np.array([0, 1, 2], dtype=int)
        self.reset()

    def reset(self):
        self.block_counts_ = np.zeros(
            shape=(self.max_block_operations_,), dtype=np.int64
        )
        self.block_size_ = np.array(
            [self.block_m**i for i in range(self.max_block_operations_)],
            dtype=np.int64,
        )
        self.data_ = np.zeros(
            shape=(self.max_block_operations_, self.block_p, self.N), dtype=float
        )
        self.correlator_ = np.zeros(
            shape=(self.max_block_operations_, self.block_p), dtype=float
        )
        self.accumulator_ = np.zeros(
            shape=(self.max_block_operations_, self.N), dtype=float
        )

    def _update_data(self, value: np.ndarray, i: int = 0):
        # Update data level i
        # Push value to the first place in the array
        # Shift all the others
        self.data_[i, 1:, :] = self.data_[i, :-1, :]
        self.data_[i, 0, :] = value

    def _update_lowest_correlator(self):
        # Perform correlation between first data point
        # and the rest of the array
        self.correlator_[0] += np.matmul(self.data_[0, :, :], self.data_[0, 0, :])

    def _update_correlator(self, i: int = 1):
        # Perform correlation between first data point
        # and the rest of the array
        # Only update after 'ratio'
        self.correlator_[i, self.ratio :] += np.matmul(
            self.data_[i, self.ratio :, :], self.data_[i, 0, :]
        )

    def accumulate(self, value: np.ndarray):
        # Update count of the lowest level and accumulator
        # (which corresponds to the overall counter)
        self.block_counts_[0] += 1
        self.accumulator_[0, :] += value
        # Update data, correlator of the lowest block
        self._update_data(value)
        self._update_lowest_correlator()

        # Accumulate higher blocks with the new value
        update_i = np.argmax(self.count() % self.block_size_[::-1] == 0)
        for i in range(1, self.max_block_operations_ - update_i):
            # The new value is the averaged cumulative sum of the previous level
            # Update count and accumulator of new level
            value = self.accumulator_[i - 1]
            self.block_counts_[i] += 1
            self.accumulator_[i, :] += value
            value = self.accumulator_[i - 1]
            # Update data, correlator of the new block
            self._update_data(value, i)
            self._update_correlator(i)
            # Reset old accumulator
            self.accumulator_[i - 1, :] = 0

    def count(self, i: int = 0) -> int:
        return self.block_counts_[i]

    def _rebuild_levels(self, correlator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Rebuild all the levels
        # The correlator is complete until the level count is > p
        level_mask = self.block_counts_ >= self.block_p
        # Create 2D array of counter
        # For each row:
        # n_ij = block_count_i - j, j = 0...(block_p - 1)
        n = self.block_counts_[level_mask, None] - np.arange(self.block_p)
        res = correlator[level_mask] / n
        # Create 2D array of time
        # For each row:
        # t_ij = block_size * j, j = 0...(block_n - 2)
        t = self.block_size_[level_mask, None] * np.arange(self.block_p)
        # Mask valid times
        mask = np.ones(shape=res.shape, dtype=int)
        # From the second row onward, mask until the ratio element
        mask[1:, : self.ratio] = 0
        return t[mask == 1], res[mask == 1]

    def acf(self) -> np.ndarray:
        # Return the autocorrelation function
        # The levels of the correlator need to be divided by
        # block_size**2 (since each sample is a product of two block averages)
        t, res = self._rebuild_levels(self.correlator_ / self.block_size_[:, None] ** 2)
        return np.column_stack((t, res / self.N))


class MSDCorrelator(Correlator):
    """MSD Correlator sampler.
     Estimates MSD using velocities.

    Each value fed to the correlator is a N-dimensional vector
    where N is the number of ensemble members.

    Accumulates values on the fly.
    Uses a 2-16 correlator ladder to coarse-grain values
    at increasingly large time intervals.

    Extension of Correlator - MSD Correlator will return the
        velocity ACF in addition to the MSD.

    References:
     - J. Ramirez et al., "Efficient on the fly calculation of time correlation
       functions in computer simulations", J. Chem. Phys. 133, 154103 (2010)
     - D. Frenkel and B. Smit, "Understanding Molecular Simulation",
       Academic Press Inc. (2001)
    """

    def __init__(self, steps: int, N: int = 1) -> None:
        super().__init__(steps, N)
        self.msd_correlator_ = np.zeros(
            shape=(self.max_block_operations_, self.block_p), dtype=float
        )

    def _update_lowest_correlator(self):
        super()._update_lowest_correlator()
        # Perform correlation between first data point
        # and the rest of the array
        s = min(self.block_p, self.count())
        vsum = self.data_[0, :s, :].cumsum(axis=0)
        self.msd_correlator_[0, :s] += (vsum**2).sum(axis=1)

    def _update_correlator(self, i: int = 1):
        super()._update_correlator(i)
        # Perform correlation between first data point
        # and the rest of the array
        # Only update after 'ratio'
        s = min(self.block_p, self.count(i))
        vsum = self.data_[i, :s, :].cumsum(axis=0)
        self.msd_correlator_[i, self.ratio : s] += (vsum[self.ratio : s, :] ** 2).sum(
            axis=1
        )

    def msd(self) -> np.ndarray:
        # Return the autocorrelation function
        # The correlator for the msd does not need to be normalised
        t, res = self._rebuild_levels(self.msd_correlator_)
        # The MSD from the vacf starts from t=1,
        # so we shift all the values and add t=0, MSD=0
        res[1:] = res[:-1]
        res[0] = 0
        return np.column_stack((t, res / self.N))


class MSDFrenkelCorrelator(Correlator):
    """MSD Correlator sampler.
     Estimates MSD using velocities.

    Each value fed to the correlator is a N-dimensional vector
     where N is the number of ensemble members.

    Accumulates values on the fly.
    Uses a 2-16 correlator ladder to coarse-grain values
     at increasingly large time intervals.

    Original implementation of Frenkel & Smit.
    Does direct sampling of velocity sums.

    References:
     - J. Ramirez et al., "Efficient on the fly calculation of time correlation
       functions in computer simulations", J. Chem. Phys. 133, 154103 (2010)
     - D. Frenkel and B. Smit, "Understanding Molecular Simulation",
       Academic Press Inc. (2001)
    """

    def _update_data(self, value: float, i: int = 0):
        # Update data level i
        # Push value to the first place in the array
        # Shift all the others
        s = min(self.block_p, self.count(i))
        self.data_[i, 1:s, :] = self.data_[i, : s - 1, :] + value
        self.data_[i, 0, :] = value

    def _update_lowest_correlator(self):
        # Perform correlation between first data point
        # and the rest of the array
        self.correlator_[0] += (self.data_[0, :, :] ** 2).sum(axis=1)

    def _update_correlator(self, i: int = 1):
        # Perform correlation between first data point
        # and the rest of the array
        # Only update after 'ratio'
        self.correlator_[i, self.ratio :] += (self.data_[i, self.ratio :, :] ** 2).sum(
            axis=1
        )

    def msd(self) -> np.ndarray:
        # Return the autocorrelation function
        # The correlator for the msd does not need to be normalised
        t, res = self._rebuild_levels(self.correlator_)
        # The MSD from the vacf starts from t=1,
        # so we shift all the values and add t=0, MSD=0
        res[1:] = res[:-1]
        res[0] = 0
        return np.column_stack((t, res / self.N))


class DCorrelator:
    """DCorrelator sampler.
     Estimates correlations for a N-dimensional system.
     Performs a sum of the individual correlations on each dimension.
     Example: for a 3D system, the VACF is equal to Vx + Vy + Vz

    Each value fed to the correlator is a (Nxdim) matrix
     where N is the number of ensemble members and dim is the system dimensionality.

    Requires a `correlator` class constructor with methods `accumulate` and `acf`.
    Example: can be used with Correlator and MSDCorrelator

    """

    def __init__(
        self,
        steps: int,
        correlator: Callable,
        N: int = 1,
        dim: int = 3,
    ) -> None:
        self.steps = steps
        self.N = N
        self.d = dim
        self.corrs = [correlator(self.steps, self.N) for _ in range(self.d)]
        assert has_method(self.corrs[0], "accumulate")
        assert has_method(self.corrs[0], "acf")

    def accumulate(self, value: np.ndarray):
        for i, corr in enumerate(self.corrs):
            corr.accumulate(value[:, i])

    def acf(self) -> np.ndarray:
        # Return the autocorrelation function
        res = self.corrs[0].acf()
        for corr in self.corrs[1:]:
            res[:, 1] += corr.acf()[:, 1]
        return res

    def msd(self) -> Union[np.ndarray, None]:
        if has_method(self.corrs[0], "msd"):
            # Return the autocorrelation function
            res = self.corrs[0].msd()
            for corr in self.corrs[1:]:
                res[:, 1] += corr.msd()[:, 1]
            return res
        else:
            print('The correlator does not possess a method "msd".')


spec_radial = [
    ("L", float32),
    ("n_bins", int32),
    ("count", int32),
    ("delta", float32),
    ("g_array", int64[:]),
]


@jitclass(spec_radial)
class RadialDistribution:
    """Radial Distribution sampler.
     Estimates g(r) using a standard binning
     procedure.

    Accumulates values on the fly.
    Each value is a vector of squared ij distances
     sampled over all the system.
    """

    def __init__(self, L: float, n_bins: int = 100) -> None:
        self.L = L
        self.n_bins = n_bins
        self.count = 0
        self.delta = self.L / (2 * self.n_bins)
        self.g_array = np.zeros(shape=(n_bins,), dtype=np.int64)

    def accumulate(self, sq_distances: np.ndarray):
        self.count += 1
        distances = np.sqrt(sq_distances)
        mask = distances < self.L / 2
        raw_idx = (distances[mask] / self.delta).astype(int32)
        for i in raw_idx:
            self.g_array[i] += 2

    def rdf(self) -> np.ndarray:
        r = (np.arange(self.n_bins) + 0.5) * self.delta
        vb = np.diff(np.arange(self.n_bins + 1) ** 3) * self.delta**3
        nid = (4 / 3) * np.pi * vb
        return np.column_stack((r, self.g_array / (nid * self.count)))
