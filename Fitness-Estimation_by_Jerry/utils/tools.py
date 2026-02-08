import numpy as np
from numba import njit
from numpy.typing import NDArray
from utils.numba_utils import numba_switchable

@numba_switchable(cache=True)
def dirichlet_multinomial_drift(
    counts: NDArray[np.float64],  # 原始频率或数量, shape=(G,)
    Ne: float,                    # 有效群体大小
    seed: int,
    counter: int
) -> NDArray[np.float64]:
    """
    Numba-friendly Dirichlet version of continuous multinomial genetic drift,
    safe for zero-frequency genotypes.

    This function expects `counts` to represent raw counts or frequencies; it
    normalizes them internally before computing Dirichlet/Gamma draws and
    returns a probability vector (sums to 1) when successful.
    """
    rng_seed = seed + counter
    np.random.seed(rng_seed)

    G = counts.size
    total = counts.sum()

    if total <= 0.0:
        return np.zeros_like(counts)

    freqs = counts / total
    alpha = freqs * Ne
    sampled = np.zeros(G, dtype=np.float64)

    for i in range(G):
        if alpha[i] > 0.0:
            sampled[i] = np.random.gamma(alpha[i], 1.0)
        else:
            sampled[i] = 0.0

    sampled_sum = sampled.sum()
    if sampled_sum > 0.0:
        sampled /= sampled_sum  # normalize to 1
    else:
        sampled[:] = 0.0

    return sampled
