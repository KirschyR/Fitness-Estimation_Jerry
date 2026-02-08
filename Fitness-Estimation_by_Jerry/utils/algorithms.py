"""Algorithm helpers used by population simulation code.

This module provides NumPy-based, vectorized helper functions for computing
mating/sperm matrices, updating sperm storage and occupancy, generating
offspring distributions, and other population genetics operations. All
functions are written to be shape-defensive and to integrate with the
`PopulationState` data structures.

Docstring style: Google style (Args, Returns, Raises, Example).
"""
from typing import Tuple, Annotated, Optional, Union

import numpy as np
from numpy.typing import NDArray
from utils.numba_utils import numba_switchable
from utils.tools import dirichlet_multinomial_drift

# 1. Prepare male gamete pool
@numba_switchable(cache=True)
def compute_mating_probability_matrix(
    sexual_selection_matrix: Annotated[NDArray[np.float64], "shape=(g,g)"], 
    male_counts: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int
) -> Annotated[NDArray[np.float64], "shape=(g,g)"]:
    """Compute a row-normalized mating probability matrix.

    The function computes A = alpha * diag(M) (implemented as column-wise
    scaling) and returns a row-normalized matrix P where each row sums to 1.

    Args:
        sexual_selection_matrix: Preference weights with shape ``(g, g)``.
            Rows correspond to female genotypes, columns to male genotypes.
        male_counts: Male counts vector with shape ``(g,)``.
        n_genotypes: Number of genotypes ``g`` used for shape validation.

    Returns:
        np.ndarray: Row-normalized mating probability matrix ``P`` with shape
            ``(g, g)``. Any zero rows in the intermediate matrix are preserved
            as zero rows in the output.
    """
    A = np.asarray(sexual_selection_matrix)
    M = np.asarray(male_counts)
    g = n_genotypes

    assert A.shape == (g, g)
    assert M.shape == (g,)

    # Multiply columns of alpha by male_counts (equivalent to alpha @ diag(M))
    weighted = A * M[None, :]  # shape (g,g)

    # Row-normalize weighted matrix
    row_sums = weighted.sum(axis=1).reshape(-1, 1)  # shape (g,1)
    # avoid division by zero: leave zero rows as zeros
    # Vectorized handling: replace zero row sums with 1.0 and compute P without a Python loop
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    P = weighted / row_sums
    return P

@numba_switchable(cache=True)
def compute_new_sperm_pool(
    mating_probability_matrix: Annotated[NDArray[np.float64], "shape=(g,g)"], 
    male_fecundity_fitness: Annotated[NDArray[np.float64], "shape=(g,)"], 
    male_gamete_production_matrix: Annotated[NDArray[np.float64], "shape=(g,h*l)"],  # 压缩形式
    n_genotypes: int,
    n_haplogenotypes: int,
    n_glabs: int = 1
) -> Annotated[NDArray[np.float64], "shape=(g,h*l)"]:  # 返回压缩形式
    """Compute the expected sperm distribution received by each female genotype.

    The function computes S_new = P @ (diag(phi_m) @ Gm) efficiently using
    broadcasting.

    Args:
        mating_probability_matrix: ``P`` with shape ``(g, g)``.
        male_fecundity_fitness: ``phi_m`` vector with shape ``(g,)``.
        male_gamete_production_matrix: ``Gm`` with compressed shape ``(g, h*l)``.
        n_genotypes: Number of genotypes ``g``.
        n_haplogenotypes: Number of haplogenotypes ``h``.
        n_glabs: Number of gamete labels per haplogenotype (default ``1``).

    Returns:
        np.ndarray: ``S_new`` with shape ``(g, h*l)`` (compressed representation).
    """
    P = np.asarray(mating_probability_matrix)
    phi_m = np.asarray(male_fecundity_fitness)
    Gm_arr = np.asarray(male_gamete_production_matrix)
    
    assert P.shape == (n_genotypes, n_genotypes)
    assert phi_m.shape == (n_genotypes,)

    # male_gamete_production_matrix 已经是压缩形式 (g, h*l)
    assert Gm_arr.shape == (n_genotypes, n_haplogenotypes * n_glabs)

    # compute diag(phi) @ Gm efficiently as phi[:,None] * Gm
    effective_sperm_per_genotype = phi_m[:, None] * Gm_arr  # shape (g, h*l)
    S_new = P @ effective_sperm_per_genotype  # shape (g, h*l)

    return S_new  # 返回压缩形式

# 2. Update female adults' sperm storage and occupancy
@numba_switchable(cache=True)
def update_sperm_and_occupancy(
    sperm_storage: Annotated[NDArray[np.float64], "shape=(A,g,hl)"],  # 压缩形式
    female_occupancy: Annotated[NDArray[np.float64], "shape=(A,g)"],
    new_sperm_pool: Annotated[NDArray[np.float64], "shape=(g,hl)"],  # 压缩形式
    adult_female_mating_rate: float,
    sperm_displacement_rate: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    n_glabs: int = 0,
) -> tuple[Annotated[NDArray[np.float64], "shape=(A,g,hl)"], Annotated[NDArray[np.float64], "shape=(A,g)"]]:  # 压缩形式
    """Vectorized update of adult females' sperm storage and occupancy.

    This updates sperm storage (compressed haplotype/gamete axis) and the
    female occupancy (non-virgin fraction) for adult age classes.

    Args:
        sperm_storage: Current sperm storage array with shape ``(A, g, hl)``.
        female_occupancy: Current occupancy array with shape ``(A, g)``.
        new_sperm_pool: New sperm distribution for this timestep with shape
            ``(g, hl)`` (compressed representation).
        adult_female_mating_rate: Fraction of adult females participating in mating.
        sperm_displacement_rate: Fractional displacement rate of existing sperm.
        adult_start_idx: Index of the first adult age (e.g., 2).
        n_ages: Total number of ages ``A``.
        n_genotypes: Number of genotypes ``g``.
        n_haplogenotypes: Number of haplogenotypes ``h``.
        n_glabs: Number of gamete labels per haplogenotype (default ``0`` meaning
            none; used to compute compressed ``hl`` dimension).

    Returns:
        tuple: ``(updated_sperm_storage, updated_female_occupancy)`` with the
            same shapes as the input arrays. Both outputs are copies (inputs not
            modified in-place by the caller of this helper).
    """
    S = np.asarray(sperm_storage)
    Q = np.asarray(female_occupancy)
    S_new = np.asarray(new_sperm_pool)
    
    hl = n_haplogenotypes * n_glabs if n_glabs > 0 else n_haplogenotypes
    assert S_new.shape == (n_genotypes, hl)
    assert Q.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, hl)

    # 直接使用压缩形式，不需要 reshape
    # S: (A, g, hl)
    # S_new: (g, hl)
    
    # 提取成年部分
    Q_adults = Q[adult_start_idx:]
    S_adults = S[adult_start_idx:]
    
    # 计算系数
    gamma_new = adult_female_mating_rate * ((1 - Q_adults) + Q_adults * sperm_displacement_rate)
    gamma_old = 1.0 - adult_female_mating_rate * sperm_displacement_rate
    
    # 更新精子库 (直接使用压缩维度)
    S[adult_start_idx:] = (
        gamma_new[..., None] * S_new[None, :, :] + 
        gamma_old * S_adults
    )
    
    # 更新占位率
    Q[adult_start_idx:] += adult_female_mating_rate * (1.0 - Q_adults)
    
    # 直接返回压缩形式，不需要 reshape
    return S, Q

# 3. Generate offspring
@numba_switchable(cache=True)
def generate_offspring_distribution(
    population_females: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_storage: Annotated[NDArray[np.float64], "shape=(A,g,h,l)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    meiosis_f: Annotated[NDArray[np.float64], "shape=(g,h,l)"],
    haplo_to_genotype_map: Annotated[NDArray[np.float64], "shape=(h,l,h,l,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    n_glabs: int = 1,
    sex_ratio: float = 0.5
) -> Annotated[NDArray[np.float64], "shape=(g,)"]:
    """Compute offspring genotype distributions produced by all mothers.

    The routine uses compressed haplotype/gamete dimensions and einsum-based
    contractions to compute the expected numbers of offspring genotypes
    produced by adult females.

    Args:
        population_females: Female counts by age and genotype with shape
            ``(A, g)``.
        sperm_storage: Sperm storage tensor (compressed) with shape ``(A, g, hl)``.
        fertility_f: Fertility modifier per female genotype with shape ``(g,)``.
        meiosis_f: Meiosis/gamete production matrix with shape ``(g, hl)``.
        haplo_to_genotype_map: Mapping tensor from haplotype pairs to offspring
            genotype probabilities with appropriate compressed shape.
        average_eggs_per_wt_female: Baseline eggs produced by a wild-type female.
        adult_start_idx: Index where adult ages begin.
        n_ages: Total number of ages ``A``.
        n_genotypes: Number of genotypes ``g``.
        n_haplogenotypes: Number of haplogenotypes ``h``.
        n_glabs: Number of gamete labels per haplogenotype (default ``1``).
        sex_ratio: Proportion of offspring that are female (default ``0.5``).

    Returns:
        tuple: ``(n_0_female, n_0_male)`` where each is an array of expected
        counts per genotype with shape ``(g,)``.
    """
    F_tensor = np.asarray(population_females)
    S_tensor = np.asarray(sperm_storage)
    phi_f = np.asarray(fertility_f)
    G_f = np.asarray(meiosis_f)
    H = np.asarray(haplo_to_genotype_map)
    
    # 压缩形式: h*l
    hl = n_haplogenotypes * n_glabs
    
    # 更新形状检查（使用压缩形式）
    assert F_tensor.shape == (n_ages, n_genotypes)
    assert S_tensor.shape == (n_ages, n_genotypes, hl)  # 压缩形式
    assert phi_f.shape == (n_genotypes,)
    assert G_f.shape == (n_genotypes, hl)  # 压缩形式
    assert H.shape == (hl, hl, n_genotypes)  # 压缩形式
    
    # 提取成年切片
    F_adults = F_tensor[adult_start_idx:]
    S_adults = S_tensor[adult_start_idx:]
    
    # TODO: 这里把原本的 np.einsum 改成手动累加，需要进一步确认一致性
    # 数据已经是压缩形式，无需 reshape
    # G_f: (g, hl)
    # S_adults: (A_adult, g, hl)
    # H: (hl, hl, g)

    # (A, G)
    Fw = F_adults * phi_f

    # 手动累加掉 A 维（避免 3D 广播）
    temp_gs = np.zeros((n_genotypes, hl), dtype=np.float64)
    for a in range(F_adults.shape[0]):
        temp_gs += Fw[a, :, None] * S_adults[a]

    # (E, S)
    W = np.dot(G_f.T, temp_gs)

    # (G,)
    n_raw = np.dot(W.ravel(), H.reshape(hl * hl, n_genotypes))
    
    # 计算最终结果
    n_0_both_sex = n_raw * average_eggs_per_wt_female
    n_0_female = n_0_both_sex * sex_ratio
    n_0_male = n_0_both_sex * (1 - sex_ratio)
    n_0 = (n_0_female, n_0_male)
    
    return n_0

# 4. Post-processing: recruitment and genetic drift
@numba_switchable(cache=True)
def recruit_juveniles(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    carrying_capacity: float,
    n_genotypes: int
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Apply carrying-capacity recruitment scaling to age-0 juveniles.

    If the total number of age-0 juveniles exceeds the carrying capacity ``K``,
    both female and male vectors are scaled by the same factor ``K / total``.

    Args:
        age_0_juvenile_counts: Tuple ``(female, male)`` of arrays with shape ``(g,)``.
        carrying_capacity: Carrying capacity ``K``.
        n_genotypes: Number of genotypes ``g`` used for shape validation.

    Returns:
        Tuple of arrays ``(female_recruited, male_recruited)`` each with shape ``(g,)``.
    """
    female_0, male_0 = age_0_juvenile_counts
    female_0 = np.asarray(female_0)
    male_0 = np.asarray(male_0)
    assert female_0.shape == (n_genotypes,)
    assert male_0.shape == (n_genotypes,)

    total_n_0 = float(female_0.sum() + male_0.sum())
    # print(f"Total age 0 juveniles before recruitment: {total_n_0}, Carrying capacity: {carrying_capacity}")

    if total_n_0 <= carrying_capacity:
        return female_0.copy(), male_0.copy()
    else:
        scaling_factor = carrying_capacity / total_n_0
        return female_0 * scaling_factor, male_0 * scaling_factor

@numba_switchable(cache=True)
def recruit_juveniles_given_scaling_factor(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    scaling_factor: float,
    n_genotypes: int
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Apply carrying-capacity recruitment scaling to age-0 juveniles.

    If the total number of age-0 juveniles exceeds the carrying capacity ``K``,
    both female and male vectors are scaled by the same factor ``K / total``.

    Args:
        age_0_juvenile_counts: Tuple ``(female, male)`` of arrays with shape ``(g,)``.
        scaling_factor: Scaling factor.
        n_genotypes: Number of genotypes ``g`` used for shape validation.

    Returns:
        Tuple of arrays ``(female_recruited, male_recruited)`` each with shape ``(g,)``.
    """
    female_0, male_0 = age_0_juvenile_counts
    female_0 = np.asarray(female_0)
    male_0 = np.asarray(male_0)
    assert female_0.shape == (n_genotypes,)
    assert male_0.shape == (n_genotypes,)

    return female_0 * scaling_factor, male_0 * scaling_factor

@numba_switchable(cache=True)
def recruit_juveniles_logistic(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    age_1_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    age_1_carrying_capacity: float,
    expected_num_adult_females: float,
    eggs_per_wt_female: float,
    relative_competition_ability: float, # age 1 vs age 0
    low_density_growth_rate: float,
    n_genotypes: int
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Apply logistic recruitment scaling to age-0 juveniles."""
    # Ensures that when the population is at the equilibrium, the scaling factor makes 
    # the total number of age-0 juveniles equal to the carrying capacity (of age 1).

    expected_total_age_0 = expected_num_adult_females * eggs_per_wt_female
    expected_total_age_1 = age_1_carrying_capacity
    expected_survival_rate = expected_total_age_1 / expected_total_age_0

    total_age_0 = float(age_0_juvenile_counts[0].sum() + age_0_juvenile_counts[1].sum())
    total_age_1 = float(age_1_juvenile_counts[0].sum() + age_1_juvenile_counts[1].sum())

    expected_competition_strength = expected_total_age_0 + expected_total_age_1 * relative_competition_ability
    actual_competition_strength = total_age_0 + total_age_1 * relative_competition_ability
    competition_ratio = actual_competition_strength / expected_competition_strength

    # Logistic growth: linear relationship between competition strength and growth rate
    actual_growth_rate = max(0, - competition_ratio * (low_density_growth_rate - 1) + low_density_growth_rate)

    actual_survival_rate = actual_growth_rate * expected_survival_rate

    return recruit_juveniles_given_scaling_factor(
        age_0_juvenile_counts,
        actual_survival_rate,
        n_genotypes
    )

@numba_switchable(cache=True)
def apply_genetic_drift(
    age_1_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    effective_population_size: float,
    n_genotypes: int,
    seed: int,
    counter: int
) -> Annotated[NDArray[np.int64], "shape=(g,)"]:
    """Apply genetic drift via multinomial sampling to age-1 juveniles.

    The routine flattens the sex-by-genotype counts to a single category vector,
    samples ``n_e`` draws from a multinomial with those frequencies, and then
    rescales back to the original total.

    Args:
        age_1_juvenile_counts: Tuple of arrays ``(female, male)`` each shape ``(g,)``.
        effective_population_size: Effective population size ``N_e`` used for sampling.
        n_genotypes: Number of genotypes ``g`` used for shape validation.
        random_state: Optional NumPy random Generator.

    Returns:
        np.ndarray: Drifted counts with the same shape as the input (2, g),
        rescaled to the original total. Values may be non-integers due to
        scaling after multinomial sampling.
    """
    female_1, male_1 = age_1_juvenile_counts
    assert female_1.shape == (n_genotypes,)
    assert male_1.shape == (n_genotypes,)
    
    n_1 = np.empty((2, n_genotypes), dtype=female_1.dtype)
    n_1[0, :] = female_1
    n_1[1, :] = male_1
    
    total_n_1: float = n_1.sum()
    if total_n_1 <= 0:
        return np.zeros_like(n_1, dtype=np.float64)
    
    # n_1 在性别维度上铺平再归一化
    freqs = n_1.ravel() / total_n_1

    # Round the effective population size to nearest integer
    n_e = int(round(effective_population_size))

    # n_e_drifted = n_e
    # dirichlet_multinomial_drift 返回的是归一化的概率（和为 1）
    drifted_freqs = dirichlet_multinomial_drift(freqs, n_e, seed, counter)

    # 用漂变后的频率乘以原始总数得到新计数
    n_drifted = drifted_freqs * total_n_1

    # 恢复性别维度
    n_drifted = n_drifted.reshape(n_1.shape)
    
    return n_drifted

@numba_switchable(cache=True)
def apply_age_based_survival(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]] ,
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    n_genotypes: int,
    n_ages: int,
) -> Annotated[NDArray[np.float64], "shape=(A,g)"]:
    """Apply age-specific survival rates to a population tensor.

    Args:
        population: Tuple ``(female, male)`` each with shape ``(A, g)``.
        female_survival_rates: Array of length ``A`` for female survival per age.
        male_survival_rates: Array of length ``A`` for male survival per age.
        n_genotypes: Number of genotypes ``g`` for validation.
        n_ages: Number of ages ``A`` for validation.

    Returns:
        np.ndarray: The population tuple after applying survival rates (same shape).
    """
    female, male = population
    assert female.shape == (n_ages, n_genotypes)
    assert male.shape == (n_ages, n_genotypes)
    
    pop = np.empty((2, n_ages, n_genotypes), dtype=female.dtype)
    pop[0, :, :] = female
    pop[1, :, :] = male

    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)
    
    assert pop.shape == (2, n_ages, n_genotypes)
    assert s_f.shape == (n_ages,)
    assert s_m.shape == (n_ages,)

    # 广播乘法应用存活率
    pop[0] *= s_f[:, None]  # 雌性
    pop[1] *= s_m[:, None]  # 雄性
    
    return pop

@numba_switchable(cache=True)
def apply_viability(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]] ,
    female_viability_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    male_viability_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    n_genotypes: int,
    n_ages: int,
    target_age: int,
) -> Annotated[NDArray[np.float64], "shape=(A,g)"]:
    """Apply genotype-specific viability to the population at a target age.

    Args:
        population: Tuple ``(female, male)`` each with shape ``(A, g)``.
        female_viability_rates: Per-genotype female viability factors (shape ``(g,)``).
        male_viability_rates: Per-genotype male viability factors (shape ``(g,)``).
        n_genotypes: Number of genotypes ``g`` for validation.
        n_ages: Number of ages ``A`` for validation.
        target_age: Age index to which viability should be applied.

    Returns:
        np.ndarray: Population tuple with viability applied at ``target_age``.
    """
    female, male = population
    assert female.shape == (n_ages, n_genotypes)
    assert male.shape == (n_ages, n_genotypes)
    
    pop = np.empty((2, n_ages, n_genotypes), dtype=female.dtype)
    pop[0, :, :] = female
    pop[1, :, :] = male
    
    v_f = np.asarray(female_viability_rates)
    v_m = np.asarray(male_viability_rates)
    target_pop = pop[:, target_age, :]  # shape (2, g)
    
    assert pop.shape == (2, n_ages, n_genotypes)
    assert target_pop.shape == (2, n_genotypes)
    assert v_f.shape == (n_genotypes,)
    assert v_m.shape == (n_genotypes,)

    # 广播乘法应用 viability
    target_pop[0] *= v_f  # 雌性
    target_pop[1] *= v_m  # 雄性
    
    return pop

"""
algorithms
==========
Defines computational-intensive algorithms for population genetics simulations.
These algorithms are designed to be independent of specific data structures, thus can be accelerated by Numba.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Any, Optional
from utils.numba_utils import numba_switchable


# ============================================================================
# Recombination Haplotype Computation (Abstract as 01 sequence problem)
# ============================================================================

# @numba_switchable
def compute_recombinant_haplotypes(
    n_loci: int,
    recombination_rates: np.ndarray,
    start_maternal: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all possible recombinant haplotype patterns and their frequencies.
    
    Abstract problem: Given a sequence of loci [0, 1, 2, ..., n_loci-1] with
    recombination rates between adjacent loci, enumerate all crossover patterns
    and produce the resulting haplotype pattern (which chain at each locus).
    
    Args:
        n_loci: Number of loci (>= 1)
        recombination_rates: Shape (n_loci - 1,). recombination_rates[i] = rate between locus i and i+1
        start_maternal: If True, start from maternal chain (0); else paternal (1)
        use_numba: If True, use Numba-accelerated version; else pure Python (added by decorator)
    
    Returns:
        haplotype_patterns: Shape (2^(n_loci-1), n_loci). Each row is 01 sequence:
                            0=maternal allele at that locus, 1=paternal allele
        frequencies: Shape (2^(n_loci-1),). Frequency of each pattern.
    
    Example:
        >>> n_loci = 3
        >>> recomb_rates = np.array([0.1, 0.2])  # rate between 0-1 and 1-2
        >>> patterns, freqs = compute_recombinant_haplotypes(n_loci, recomb_rates, True)
        >>> patterns
        array([[0, 0, 0],   # No crossovers: all maternal
               [0, 0, 1],   # Crossover after locus 1: mat, mat, pat
               [0, 1, 1],   # Crossover after locus 0: mat, pat, pat
               [0, 1, 0]], dtype=int32)  # Two crossovers: mat, pat, mat
        >>> freqs
        array([0.72, 0.02, 0.18, 0.08])  # 0.9*0.8, 0.9*0.2, 0.1*0.8, 0.1*0.2
    """
    if n_loci < 1:
        raise ValueError("n_loci must be >= 1")
    
    if n_loci == 1:
        patterns = np.array([[int(not start_maternal)]], dtype=np.int32)
        frequencies = np.array([1.0], dtype=np.float64)
        return patterns, frequencies
    
    n_boundaries = n_loci - 1
    n_patterns = 2 ** n_boundaries
    
    patterns = np.zeros((n_patterns, n_loci), dtype=np.int32)
    frequencies = np.zeros(n_patterns, dtype=np.float64)
    
    for pattern_idx in range(n_patterns):
        current_chain = 0 if start_maternal else 1
        frequency = 1.0
        patterns[pattern_idx, 0] = current_chain
        
        for boundary_idx in range(n_boundaries):
            has_crossover = (pattern_idx >> boundary_idx) & 1
            recomb_rate = recombination_rates[boundary_idx]
            
            if has_crossover:
                frequency *= recomb_rate
                current_chain = 1 - current_chain
            else:
                frequency *= (1.0 - recomb_rate)
            
            patterns[pattern_idx, boundary_idx + 1] = current_chain
        
        frequencies[pattern_idx] = frequency
    
    return patterns, frequencies


def compute_recombinant_haplotypes_with_alleles(
    maternal_alleles: List[str],
    paternal_alleles: List[str],
    recombination_rates: np.ndarray,
    start_maternal: bool = True,
    use_numba: bool = True
) -> Dict[str, float]:
    """
    Compute recombinant haplotypes with actual allele symbols.
    
    Given maternal and paternal allele sequences, compute all recombinant
    haplotypes considering recombination rates, and return them as strings
    mapped to their frequencies.
    
    Args:
        maternal_alleles: List of allele symbols at each locus (maternal chain)
        paternal_alleles: List of allele symbols at each locus (paternal chain)
        recombination_rates: Recombination rates between adjacent loci
        start_maternal: Start from maternal (Arue) or paternal (False)
        use_numba: If True, use Numba-accelerated version; else pure Python
    
    Returns:
        Dict mapping haplotype string (e.g., "A1/a2/A3") to frequency
    """
    n_loci = len(maternal_alleles)
    if len(paternal_alleles) != n_loci:
        raise ValueError("maternal_alleles and paternal_alleles must have same length")
    
    # Compute patterns (auto-selects Numba or Python)
    patterns, frequencies = compute_recombinant_haplotypes(
        n_loci, recombination_rates, start_maternal, use_numba=use_numba
    )
    
    # Convert patterns to haplotype strings
    result = {}
    for pattern_idx, pattern in enumerate(patterns):
        alleles = [
            maternal_alleles[i] if chain == 0 else paternal_alleles[i]
            for i, chain in enumerate(pattern)
        ]
        result["/".join(alleles)] = frequencies[pattern_idx]
    
    return result