import numpy as np
from numpy.typing import ArrayLike

# J. Kolafa, I. Nezbeda / Fluid Phase Equilibria 100 (1994) 1-34
# LJ fluid EOS for:
# - Helmoltz Free Energy
# - Pressure
# - Energy


def gammaBH(_):
    return 1.92907278


def ALJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Helmoltz Free Energy
    eta = np.pi / 6.0 * rho * (dC(T)) ** 3
    return (
        np.log(rho) + betaAHS(eta) + rho * BC(T) / np.exp(gammaBH(T) * rho**2)
    ) * T + DALJ(T, rho)


def ALJres(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Helmoltz Free Energy (no ideal term)
    eta = np.pi / 6.0 * rho * (dC(T)) ** 3
    return (betaAHS(eta) + rho * BC(T) / np.exp(gammaBH(T) * rho**2)) * T + DALJ(
        T, rho
    )


def PLJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Pressure
    eta = np.pi / 6.0 * rho * (dC(T)) ** 3
    return (
        (
            zHS(eta)
            + BC(T)
            / np.exp(gammaBH(T) * rho**2)
            * rho
            * (1 - 2 * gammaBH(T) * rho**2)
        )
        * T
        + DPLJ(T, rho)
    ) * rho


def ULJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Internal Energy
    dBHdT = dCdT(T)
    dB2BHdT = BCdT(T)
    d = dC(T)
    eta = np.pi / 6.0 * rho * d**3

    return (
        3 * (zHS(eta) - 1) * dBHdT / d
        + rho * dB2BHdT / np.exp(gammaBH(T) * rho**2)
        + DULJ(T, rho)
    )


def zHS(eta: ArrayLike) -> ArrayLike:
    # Hard Sphere pressure
    return (1 + eta * (1 + eta * (1 - eta / 1.5 * (1 + eta)))) / (1 - eta) ** 3


def betaAHS(eta: ArrayLike) -> ArrayLike:
    # Hard Sphere free energy
    return (
        np.log(1 - eta) / 0.6
        + eta * ((4.0 / 6 * eta - 33.0 / 6) * eta + 34.0 / 6) / (1.0 - eta) ** 2
    )


def dLJ(T: ArrayLike) -> ArrayLike:
    # Lennard Jones diameter (see paper)
    isT = 1 / np.sqrt(T)
    return (
        (
            ((0.011117524191338 * isT - 0.076383859168060) * isT) * isT
            + 0.000693129033539
        )
        / isT
        + 1.080142247540047
        + 0.127841935018828 * np.log(isT)
    )


def dC(T: ArrayLike) -> ArrayLike:
    # Hard Sphere diameter (see paper)
    sT = np.sqrt(T)
    return (
        -0.063920968 * np.log(T)
        + 0.011117524 / T
        - 0.076383859 / sT
        + 1.080142248
        + 0.000693129 * sT
    )


def dCdT(T: ArrayLike) -> ArrayLike:
    # Derivative of HS diameter on T (see paper)
    sT = np.sqrt(T)
    return (
        0.063920968 * T
        + 0.011117524
        + (-0.5 * 0.076383859 - 0.5 * 0.000693129 * T) * sT
    )


def BC(T: ArrayLike) -> ArrayLike:
    # Virial term (see paper)
    isT = 1 / np.sqrt(T)
    return (
        (
            (((-0.58544978 * isT + 0.43102052) * isT + 0.87361369) * isT - 4.13749995)
            * isT
            + 2.90616279
        )
        * isT
        - 7.02181962
    ) / T + 0.02459877


def BCdT(T):
    # Derivative of virial on T  (see paper)
    isT = 1 / np.sqrt(T)
    return (
        (
            ((-0.58544978 * 3.5 * isT + 0.43102052 * 3) * isT + 0.87361369 * 2.5) * isT
            - 4.13749995 * 2
        )
        * isT
        + 2.90616279 * 1.5
    ) * isT - 7.02181962


def DALJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Summation term in A (see paper)
    return (
        (
            (
                +2.01546797
                + rho * (-28.17881636 + rho * (+28.28313847 + rho * (-10.42402873)))
            )
            + (
                -19.58371655
                + rho
                * (
                    75.62340289
                    + rho
                    * ((-120.70586598) + rho * (93.92740328 + rho * (-27.37737354)))
                )
            )
            / np.sqrt(T)
            + (
                (
                    29.34470520
                    + rho
                    * (
                        (-112.35356937)
                        + rho
                        * (+170.64908980 + rho * ((-123.06669187) + rho * 34.42288969))
                    )
                )
                + (
                    -13.37031968
                    + rho
                    * (
                        65.38059570
                        + rho
                        * ((-115.09233113) + rho * (88.91973082 + rho * (-25.62099890)))
                    )
                )
                / T
            )
            / T
        )
        * rho
        * rho
    )


def DPLJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Summation term in P (see paper)
    return (
        (
            2.01546797 * 2
            + rho
            * (
                +(-28.17881636) * 3
                + rho * (+28.28313847 * 4 + rho * +(-10.42402873) * 5)
            )
        )
        + (
            (-19.58371655) * 2
            + rho
            * (
                +75.62340289 * 3
                + rho
                * (
                    +(-120.70586598) * 4
                    + rho * (+93.92740328 * 5 + rho * +(-27.37737354) * 6)
                )
            )
        )
        / np.sqrt(T)
        + (
            (
                29.34470520 * 2
                + rho
                * (
                    +(-112.35356937) * 3
                    + rho
                    * (
                        +170.64908980 * 4
                        + rho * (+(-123.06669187) * 5 + rho * +34.42288969 * 6)
                    )
                )
            )
            + (
                (-13.37031968) * 2
                + rho
                * (
                    +65.38059570 * 3
                    + rho
                    * (
                        +(-115.09233113) * 4
                        + rho * (+88.91973082 * 5 + rho * +(-25.62099890) * 6)
                    )
                )
            )
            / T
        )
        / T
    ) * rho**2


def DULJ(T: ArrayLike, rho: ArrayLike) -> ArrayLike:
    # Summation term in U (see paper)
    return (
        (
            (
                2.01546797
                + rho * (+(-28.17881636) + rho * (+28.28313847 + rho * +(-10.42402873)))
            )
            + +(
                -19.58371655 * 1.5
                + rho
                * (
                    +75.62340289 * 1.5
                    + rho
                    * (
                        +(-120.70586598) * 1.5
                        + rho * (+93.92740328 * 1.5 + rho * +(-27.37737354) * 1.5)
                    )
                )
            )
            / np.sqrt(T)
            + +(
                (
                    29.34470520 * 2
                    + rho
                    * (
                        +-112.35356937 * 2
                        + rho
                        * (
                            +170.64908980 * 2
                            + rho * (+-123.06669187 * 2 + rho * +34.42288969 * 2)
                        )
                    )
                )
                + +(
                    -13.37031968 * 3
                    + rho
                    * (
                        +65.38059570 * 3
                        + rho
                        * (
                            +-115.09233113 * 3
                            + rho * (+88.91973082 * 3 + rho * +(-25.62099890) * 3)
                        )
                    )
                )
                / T
            )
            / T
        )
        * rho
        * rho
    )
