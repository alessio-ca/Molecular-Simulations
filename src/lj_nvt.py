from typing import Tuple
import numpy as np
from numba import njit


def lj_nvt(
    N: int,
    rho: float,
    T: float,
    delta: float,
    eqnum: int = 100,  # Equilibration steps
    snum: int = 1000,  # Sampling steps
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    # Reduced units:
    # epsilon (energy)
    # sigma (length)
    # m (mass)

    # U* = U * epsilon
    # T* = kb * T / epsilon
    # P* = P * sigma^3 / epsilon
    # rho* = rho * sigma^3

    V = N / rho
    L = V ** (1 / 3)
    cutoff = L / 2  # cutoff
    cutoff_sq = cutoff**2
    # Tail of the energy contribution due to truncation
    tail_e = (8 * np.pi * rho) / 3 * ((1 / cutoff) ** 9 / 3 - (1 / cutoff) ** 3)
    # Tail for the pressure contribution due to truncation
    tail_p = (
        (16 * np.pi * rho**2) / 3 * (2 * (1 / cutoff) ** 9 / 3 - (1 / cutoff) ** 3)
    )

    def init(N: int, L: float) -> np.ndarray:
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

    @njit
    def apply_bc(X: np.ndarray) -> np.ndarray:
        X[X > L] -= L
        X[X < 0] += L
        return X

    @njit
    def dist(positions: np.ndarray, particle: np.ndarray) -> np.ndarray:
        # Matrix of distances between all positions and particle
        dists = positions - particle
        # Apply image convention
        for i in range(dists.shape[1]):
            dists[:, i] = apply_bc(dists[:, i] + L / 2) - L / 2
        # Calculate euclidean squared distance
        return (dists**2).sum(axis=1)

    @njit
    def single_enrg(
        dists: np.ndarray,
        K: int,
        M: float,
    ) -> float:
        # Apply cutoff as boolean mask & calculate all single energy contributions
        rc = dists < cutoff_sq
        en = 4 * K * ((1 / dists[rc]) ** 6 - M * (1 / dists[rc]) ** 3)
        return en.sum()

    @njit
    def sample(positions: np.ndarray, K: int, M: float) -> float:
        u = 0
        for i in range(N):
            # Calculate the pairwise quantity u by summing over all pair interactions
            particle = positions[i]
            dists = dist(positions[i + 1 :], particle)
            u += single_enrg(dists, K, M)
        return u

    @njit
    def mcmove(
        positions: np.ndarray, mask: np.array, delta: float
    ) -> Tuple[np.ndarray, float, int]:
        # Random particle selection
        x = np.random.randint(0, high=N)
        pos = positions[x]
        # Set mask to exclude particle
        mask[:] = 1
        mask[x] = 0
        dists = dist(positions[mask == 1], pos)
        # Calculate initial energy contribution of particle
        eno = single_enrg(dists, 1, 1)
        # Displace particle
        delta_v = delta * (np.random.random(size=(3,)) - 0.5)
        new_pos = apply_bc(pos + delta_v)
        ndists = dist(positions[mask == 1], new_pos)
        # Calculate new energy contribution of particle
        enn = single_enrg(ndists, 1, 1)
        if enn <= eno:
            # Move is always accepted
            positions[x] = new_pos
            return enn - eno, 1
        elif np.random.random() < np.exp((eno - enn) / T):
            # Move is accepted with Metropolis rule
            positions[x] = new_pos
            return enn - eno, 1
        else:
            return 0, 0

    def adapt_delta(delta: float, frac: float) -> float:
        # Adapt delta to roughly reach 0.5 acceptance ratio
        k = max(0.5, min(1.5, 2 * frac))
        delta = min(k * delta, L / 2)
        return delta

    # Tracking arrays
    positions = init(N, L)
    mask = np.zeros(shape=(N,), dtype=int)
    u = np.zeros(shape=(snum,))
    vr = np.zeros(shape=(snum,))

    # Equilibration loop
    for i in range(eqnum):
        acc = 0
        # Trial move consists of N single moves
        for _ in range(N):
            _, moved = mcmove(positions, mask, delta)
            acc += moved

        print(
            "Equi Step: {0}/{1} Step: {2:.2f} Acc: {3:.2f}".format(
                i + 1, eqnum, delta, acc / N
            ),
            end="\r",
        )
        # During equilibration, adapt delta
        delta = adapt_delta(delta, acc / N)
    print("")

    # Sampling loop
    en = sample(positions, 1, 1)
    for i in range(snum):
        acc = 0
        # Trial move consists of N single moves
        for _ in range(N):
            delta_e, moved = mcmove(positions, mask, delta)
            en += delta_e
            acc += moved

        # Energy & virial sampling
        u[i] = en
        vr[i] = sample(positions, 12, 0.5)

        print(
            "\rStep: {0}/{1} Energy: {2:.6f} Acc: {3:.2f}".format(
                i + 1, snum, en / N, acc / N
            ),
            end="\r",
        )
    # Returns positions, energies per particle and virial
    # Estimates have long tail corrections
    return positions, u / N + tail_e, vr + tail_p
