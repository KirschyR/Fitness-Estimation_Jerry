"""
Fitness parameter estimation of a mosquito model
Python implementation with performance optimizations

Original R code: ml_overl.Rmd
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings
import os

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import pandas as pd

# Optional: Numba for JIT compilation (speedup)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SimulationParams:
    """Parameters for population simulation"""
    remate_chance: float = 0.05
    effective_population_size: int = 0
    male_viability_fitness: float = 1.0
    female_viability_fitness: float = 1.0
    male_fecundity_fitness: float = 1.0
    female_fecundity_fitness: float = 1.0
    both_fecundity_fitness: float = 1.0
    male_mating_success: float = 1.0
    max_week: int = 317
    eggs_per_female_that_reproduce: float = 25.0
    relative_competition_factor: float = 5.0
    low_density_growth_rate: float = 6.0
    drive_efficiency: float = 0.5  # Drive conversion efficiency: 0.5 = standard Mendelian, 0.7 = gene drive


# Row indices for population matrix
FEMALE_DD, FEMALE_DW, FEMALE_WW = 0, 1, 2
MALE_DD, MALE_DW, MALE_WW = 3, 4, 5
STORED_DD, STORED_DW, STORED_WW = 6, 7, 8

# Age-based survival rates (constants)
MALE_SURVIVAL_RATES = np.array([2/3, 1/2, 0, 0, 0, 0])
FEMALE_SURVIVAL_RATES = np.array([5/6, 4/5, 3/4, 2/3, 1/2, 0])


# =============================================================================
# Core simulation function
# =============================================================================

def evolve_overlapping(
    initial_genotype_freq: np.ndarray,
    params: SimulationParams
) -> List[np.ndarray]:
    """
    Simulate genotype trajectories for given parameters.
    
    Parameters
    ----------
    initial_genotype_freq : np.ndarray
        Initial genotype frequencies, shape (6, 8) or (9, 8)
    params : SimulationParams
        Simulation parameters
        
    Returns
    -------
    List[np.ndarray]
        Population statistics for each week
    """
    # Initialize population matrix (9 rows x 8 cols)
    population = np.zeros((9, 8))
    population[0:6, :] = initial_genotype_freq[0:6, :]
    
    # Record population statistic every week
    population_statistic = [population.copy()]
    
    for week in range(params.max_week):
        population = population_statistic[week].copy()
        
        # Get adult genotype frequencies
        female_adults_total = np.sum(population[0:3, 2:8])
        male_adults_total = np.sum(population[3:6, 2:5])
        
        if female_adults_total > 0:
            freq_adult_female = np.sum(population[0:3, 2:8], axis=1) / female_adults_total
        else:
            freq_adult_female = np.array([1/3, 1/3, 1/3])
            
        if male_adults_total > 0:
            freq_adult_male = np.sum(population[3:6, 2:5], axis=1) / male_adults_total
        else:
            freq_adult_male = np.array([1/3, 1/3, 1/3])
        
        # Sexual selection - mate chances
        sexual_selection = np.array([
            params.male_mating_success * freq_adult_male[0],
            freq_adult_male[1],
            freq_adult_male[2]
        ])
        sexual_selection_sum = np.sum(sexual_selection)
        
        if sexual_selection_sum > 0:
            prob_male_chosen = sexual_selection / sexual_selection_sum
        else:
            prob_male_chosen = np.array([1/3, 1/3, 1/3])
        
        # Update stored sperm
        # For females of age 2, they always select males
        population[6:9, 2] = prob_male_chosen
        
        # For females of age 3-7
        if week == 0:
            population[6:9, 3:8] = prob_male_chosen[:, np.newaxis]
        else:
            prev_stored = population_statistic[week - 1][6:9, 2:7]
            population[6:9, 3:8] = (
                (1 - params.remate_chance) * prev_stored + 
                params.remate_chance * prob_male_chosen[:, np.newaxis]
            )
        
        # Normalize stored sperm
        for col in range(2, 8):
            col_sum = np.sum(population[6:9, col])
            if col_sum > 0:
                population[6:9, col] /= col_sum
        
        # Fecundity selection - calculate offspring
        eggs = params.eggs_per_female_that_reproduce
        f_m = params.male_fecundity_fitness
        f_f = params.female_fecundity_fitness
        f_b = params.both_fecundity_fitness
        
        # Drive conversion efficiency
        # e = drive_efficiency: probability that a dw heterozygote produces a 'd' gamete
        # Standard Mendelian: e = 0.5
        # With gene drive: e > 0.5 (e.g., 0.7 as in test_mosquito_population)
        e = params.drive_efficiency
        h_to_w = 1 - e  # probability of producing 'w' gamete from dw
        
        # Female populations by age (columns 2-7)
        fem_dd = population[FEMALE_DD, 2:8]
        fem_dw = population[FEMALE_DW, 2:8]
        fem_ww = population[FEMALE_WW, 2:8]
        
        # Stored sperm frequencies by age
        stored_dd = population[STORED_DD, 2:8]
        stored_dw = population[STORED_DW, 2:8]
        stored_ww = population[STORED_WW, 2:8]
        
        offspring_dd = eggs * (
            f_b * np.sum(fem_dd * stored_dd) +                    # dd × dd → 100% dd
            f_f * np.sum(fem_dd * stored_dw) * e +                # dd × dw → e dd (drive in sperm)
            f_m * np.sum(fem_dw * stored_dd) * e +                # dw × dd → e dd (drive in egg)
            np.sum(fem_dw * stored_dw) * e * e                    # dw × dw → e² dd
        )
        
        offspring_dw = eggs * (
            f_f * np.sum(fem_dd * stored_dw) * h_to_w +            # dd × dw → (1-e) dw
            f_f * np.sum(fem_dd * stored_ww) +                    # dd × ww → 100% dw
            f_m * np.sum(fem_dw * stored_dd) * h_to_w +            # dw × dd → (1-e) dw
            np.sum(fem_dw * stored_dw) * 2 * e * h_to_w +          # dw × dw → 2e(1-e) dw
            np.sum(fem_dw * stored_ww) * e +                      # dw × ww → e dw (drive in egg)
            f_m * np.sum(fem_ww * stored_dd) +                    # ww × dd → 100% dw
            np.sum(fem_ww * stored_dw) * e                        # ww × dw → e dw (drive in sperm)
        )
        
        offspring_ww = eggs * (
            np.sum(fem_ww * stored_ww) +                          # ww × ww → 100% ww
            np.sum(fem_ww * stored_dw) * h_to_w +                  # ww × dw → (1-e) ww
            np.sum(fem_dw * stored_ww) * h_to_w +                  # dw × ww → (1-e) ww
            np.sum(fem_dw * stored_dw) * h_to_w * h_to_w          # dw × dw → (1-e)² ww
        )
        
        # Sex differentiation (1:1 ratio)
        population[0:3, 0] = np.array([offspring_dd, offspring_dw, offspring_ww]) / 2
        population[3:6, 0] = np.array([offspring_dd, offspring_dw, offspring_ww]) / 2
        
        # Update current week's population
        population_statistic[week] = population.copy()
        
        # Initialize next week
        new_population = population.copy()
        
        # Age-based survival for adults (age 2+ to next age)
        for j in range(5):
            new_population[0:3, j+3] = population[0:3, j+2] * FEMALE_SURVIVAL_RATES[j]
            new_population[3:6, j+3] = population[3:6, j+2] * MALE_SURVIVAL_RATES[j]
        
        # Viability selection (age 1 to age 2)
        new_adult_females = np.array([
            population[FEMALE_DD, 1] * params.female_viability_fitness,
            population[FEMALE_DW, 1],
            population[FEMALE_WW, 1]
        ])
        new_adult_males = np.array([
            population[MALE_DD, 1] * params.male_viability_fitness,
            population[MALE_DW, 1],
            population[MALE_WW, 1]
        ])
        
        # Apply genetic drift if effective population size > 0
        if params.effective_population_size > 0:
            ne_half = params.effective_population_size // 2
            
            # Multinomial sampling for genetic drift
            female_probs = new_adult_females / np.sum(new_adult_females) if np.sum(new_adult_females) > 0 else np.array([1/3, 1/3, 1/3])
            male_probs = new_adult_males / np.sum(new_adult_males) if np.sum(new_adult_males) > 0 else np.array([1/3, 1/3, 1/3])
            
            sampled_females = np.random.multinomial(ne_half, female_probs)
            sampled_males = np.random.multinomial(ne_half, male_probs)
            
            new_population[0:3, 2] = (sampled_females / ne_half) * np.sum(new_adult_females)
            new_population[3:6, 2] = (sampled_males / ne_half) * np.sum(new_adult_males)
        else:
            new_population[0:3, 2] = new_adult_females
            new_population[3:6, 2] = new_adult_males
        
        # Density-dependent growth (age 0 to age 1)
        expected_competition = params.eggs_per_female_that_reproduce + params.relative_competition_factor * 2 * 2/7
        competition_ratio = (
            np.sum(population[0:6, 0] + params.relative_competition_factor * population[0:6, 1]) / 
            expected_competition
        )
        relative_growth_rate = params.low_density_growth_rate - competition_ratio * (params.low_density_growth_rate - 1)
        new_larvae_survival_rate = 2 * 2/7 * relative_growth_rate / params.eggs_per_female_that_reproduce
        
        new_population[0:6, 1] = population[0:6, 0] * new_larvae_survival_rate
        
        # Reset stored sperm for new week
        new_population[6:9, :] = 0
        
        population_statistic.append(new_population.copy())
    
    return population_statistic


# =============================================================================
# Likelihood functions
# =============================================================================

def rho(observed: np.ndarray, expected: np.ndarray, n: int) -> float:
    """
    Calculate log probability density of observed genotype frequencies
    based on expected genotype frequencies using multinomial distribution.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed genotype frequencies (3 values summing to 1)
    expected : np.ndarray
        Expected genotype frequencies (3 values summing to 1)
    n : int
        Effective sample size
        
    Returns
    -------
    float
        Log probability density
    """
    # Counts
    x1 = observed[0] * n
    x2 = observed[1] * n
    x3 = observed[2] * n
    
    # Expected probabilities
    p_x = np.clip(expected[0], 1e-10, 1 - 1e-10)
    p_y = np.clip(expected[1], 1e-10, 1 - 1e-10)
    p_z = 1 - p_x - p_y
    p_z = np.clip(p_z, 1e-10, 1 - 1e-10)
    
    # Normalization correction term
    term1 = (p_x**n + p_y**n + p_z**n) / 6
    term2 = ((1-p_x)**n + (1-p_y)**n + (p_x+p_y)**n) / 2
    denom = 1 + term1 - term2
    
    if denom <= 0:
        log_a = 2 * np.log(n)
    else:
        log_a = 2 * np.log(n) - np.log(denom)
    
    # Multinomial log probability
    result = (
        log_a +
        gammaln(n + 1) - gammaln(x1 + 1) - gammaln(x2 + 1) - gammaln(x3 + 1) +
        x1 * np.log(p_x) + x2 * np.log(p_y) + x3 * np.log(p_z)
    )
    
    return result


def log_likelihood(
    f: np.ndarray,
    population_statistic: List[np.ndarray],
    mode: int,
    return_details: bool = False
) -> float:
    """
    Calculate log-likelihood of observed data given parameters.
    
    Parameters
    ----------
    f : np.ndarray
        Optimization parameters (depends on mode)
    population_statistic : List[np.ndarray]
        Observed population data
    mode : int
        Model type (1-8)
    return_details : bool
        If True, return additional diagnostic information
        
    Returns
    -------
    float or tuple
        Log-likelihood value, or tuple with additional info if return_details=True
    """
    # Default parameter values
    male_viability_fitness = 1.0
    female_viability_fitness = 1.0
    male_fecundity_fitness = 1.0
    female_fecundity_fitness = 1.0
    both_fecundity_fitness = 1.0
    male_mating_success = 1.0
    remate_chance = 0.05
    drive_efficiency = 0.5  # Default: standard Mendelian
    
    # Parse parameters based on mode
    if mode == 1:  # Full model
        effective_population_size = f[0]
        male_viability_fitness = f[1]
        female_viability_fitness = f[2]
        male_fecundity_fitness = f[3]
        female_fecundity_fitness = f[4]
        both_fecundity_fitness = f[5]
        male_mating_success = f[6]
    elif mode == 2:  # Sexual selection only
        effective_population_size = f[0]
        male_mating_success = f[1]
    elif mode == 3:  # Viability selection only
        effective_population_size = f[0]
        male_viability_fitness = f[1]
        female_viability_fitness = f[2]
    elif mode == 4:  # Fecundity selection only
        effective_population_size = f[0]
        male_fecundity_fitness = f[1]
        female_fecundity_fitness = f[2]
        both_fecundity_fitness = f[3]
    elif mode == 5:  # Minimal model 1
        effective_population_size = f[0]
        male_viability_fitness = f[1]
        female_viability_fitness = male_viability_fitness
        male_mating_success = f[2]
    elif mode == 6:  # Minimal model 2
        effective_population_size = f[0]
        male_fecundity_fitness = f[1]
        female_fecundity_fitness = male_fecundity_fitness
        both_fecundity_fitness = male_fecundity_fitness ** 2
        male_mating_success = f[2]
    elif mode == 7:  # Simplistic fecundity model
        effective_population_size = f[0]
        male_fecundity_fitness = f[1]
        female_fecundity_fitness = male_fecundity_fitness
        both_fecundity_fitness = male_fecundity_fitness ** 2
    elif mode == 8:  # Multiplicative fecundity selection
        effective_population_size = f[0]
        male_fecundity_fitness = f[1]
        female_fecundity_fitness = f[2]
        both_fecundity_fitness = male_fecundity_fitness * female_fecundity_fitness
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    n_weeks = len(population_statistic)
    
    # Make a copy to avoid modifying original data
    pop_stat = [p.copy() for p in population_statistic]
    
    # Initialize observed new adult genotype frequencies
    observed_new_adult = np.zeros((n_weeks, 6))
    
    # Initialize first row
    female_adults = np.sum(pop_stat[0][0:3, 2:8], axis=1)
    male_adults = np.sum(pop_stat[0][3:6, 2:5], axis=1)
    
    female_sum = np.sum(female_adults)
    male_sum = np.sum(male_adults)
    
    observed_new_adult[0, 0:3] = female_adults / female_sum if female_sum > 0 else np.array([1/3, 1/3, 1/3])
    observed_new_adult[0, 3:6] = male_adults / male_sum if male_sum > 0 else np.array([1/3, 1/3, 1/3])
    
    # Process each week to infer new adult genotype frequencies
    for t in range(1, n_weeks):
        prev = t - 1
        population = pop_stat[prev].copy()
        
        # Calculate adult frequencies
        female_adults_total = np.sum(population[0:3, 2:8])
        male_adults_total = np.sum(population[3:6, 2:5])
        
        if female_adults_total > 0:
            freq_adult_female = np.sum(population[0:3, 2:8], axis=1) / female_adults_total
        else:
            freq_adult_female = np.array([1/3, 1/3, 1/3])
            
        if male_adults_total > 0:
            freq_adult_male = np.sum(population[3:6, 2:5], axis=1) / male_adults_total
        else:
            freq_adult_male = np.array([1/3, 1/3, 1/3])
        
        # Sexual selection
        sexual_selection = np.array([
            male_mating_success * freq_adult_male[0],
            freq_adult_male[1],
            freq_adult_male[2]
        ])
        ss_sum = np.sum(sexual_selection)
        prob_male_chosen = sexual_selection / ss_sum if ss_sum > 0 else np.array([1/3, 1/3, 1/3])
        
        # Update stored sperm
        population[6:9, 2] = prob_male_chosen
        
        if prev == 0:
            population[6:9, 3:8] = prob_male_chosen[:, np.newaxis]
        else:
            prev_stored = pop_stat[prev - 1][6:9, 2:7]
            population[6:9, 3:8] = (
                (1 - remate_chance) * prev_stored +
                remate_chance * prob_male_chosen[:, np.newaxis]
            )
        
        # Normalize stored sperm
        for col in range(2, 8):
            col_sum = np.sum(population[6:9, col])
            if col_sum > 0:
                population[6:9, col] /= col_sum
        
        pop_stat[prev] = population
        
        # Predict next week's old adults (after death)
        expected_next = population.copy()
        for j in range(5):
            expected_next[0:3, j+3] = population[0:3, j+2] * FEMALE_SURVIVAL_RATES[j]
            expected_next[3:6, j+3] = population[3:6, j+2] * MALE_SURVIVAL_RATES[j]
        expected_next[0:6, 2] = 0
        
        # Expected old adult population by genotype
        expected_female_by_genotype = np.sum(expected_next[0:3, 3:8], axis=1)
        expected_male_by_genotype = np.sum(expected_next[3:6, 3:8], axis=1)
        
        # Observed total population by genotype
        observed_female_by_genotype = np.sum(pop_stat[t][0:3, 2:8], axis=1)
        observed_male_by_genotype = np.sum(pop_stat[t][3:6, 2:8], axis=1)
        
        # Infer new adults (observed - expected old)
        inferred_female_age2 = observed_female_by_genotype - expected_female_by_genotype
        inferred_male_age2 = observed_male_by_genotype - expected_male_by_genotype
        
        # Handle negative values
        inferred_female_age2 = np.maximum(inferred_female_age2, 0)
        inferred_male_age2 = np.maximum(inferred_male_age2, 0)
        
        # Update population statistic
        pop_stat[t][0:3, 3:8] = expected_next[0:3, 3:8]
        pop_stat[t][3:6, 3:8] = expected_next[3:6, 3:8]
        pop_stat[t][0:3, 2] = inferred_female_age2
        pop_stat[t][3:6, 2] = inferred_male_age2
        
        # Normalize and store
        female_sum = np.sum(inferred_female_age2)
        male_sum = np.sum(inferred_male_age2)
        
        observed_new_adult[t, 0:3] = inferred_female_age2 / female_sum if female_sum > 0 else np.array([1/3, 1/3, 1/3])
        observed_new_adult[t, 3:6] = inferred_male_age2 / male_sum if male_sum > 0 else np.array([1/3, 1/3, 1/3])
    
    # Calculate expected new adult genotype frequencies
    expected_new_adult = np.zeros((n_weeks, 6))
    log_likelihood_value = 0.0
    
    for t in range(2, n_weeks):
        prev = t - 2
        population = pop_stat[prev]
        
        # Female populations by age
        fem_dd = population[FEMALE_DD, 2:8]
        fem_dw = population[FEMALE_DW, 2:8]
        fem_ww = population[FEMALE_WW, 2:8]
        
        # Stored sperm frequencies
        stored_dd = population[STORED_DD, 2:8]
        stored_dw = population[STORED_DW, 2:8]
        stored_ww = population[STORED_WW, 2:8]
        
        # Drive conversion parameters
        e = drive_efficiency
        h_to_w = 1 - e
        
        # Calculate newborn genotypes (WITH DRIVE CONVERSION)
        newborn_dd = (
            both_fecundity_fitness * np.sum(fem_dd * stored_dd) +       # dd × dd → 100% dd
            female_fecundity_fitness * np.sum(fem_dd * stored_dw) * e + # dd × dw → e dd
            male_fecundity_fitness * np.sum(fem_dw * stored_dd) * e +   # dw × dd → e dd
            np.sum(fem_dw * stored_dw) * e * e                          # dw × dw → e² dd
        )
        
        newborn_dw = (
            female_fecundity_fitness * np.sum(fem_dd * stored_dw) * h_to_w +  # dd × dw → (1-e) dw
            female_fecundity_fitness * np.sum(fem_dd * stored_ww) +           # dd × ww → 100% dw
            male_fecundity_fitness * np.sum(fem_dw * stored_dd) * h_to_w +    # dw × dd → (1-e) dw
            np.sum(fem_dw * stored_dw) * 2 * e * h_to_w +                     # dw × dw → 2e(1-e) dw
            np.sum(fem_dw * stored_ww) * e +                                  # dw × ww → e dw
            male_fecundity_fitness * np.sum(fem_ww * stored_dd) +             # ww × dd → 100% dw
            np.sum(fem_ww * stored_dw) * e                                    # ww × dw → e dw
        )
        
        newborn_ww = (
            np.sum(fem_ww * stored_ww) +                      # ww × ww → 100% ww
            np.sum(fem_ww * stored_dw) * h_to_w +             # ww × dw → (1-e) ww
            np.sum(fem_dw * stored_ww) * h_to_w +             # dw × ww → (1-e) ww
            np.sum(fem_dw * stored_dw) * h_to_w * h_to_w     # dw × dw → (1-e)² ww
        )
        
        # Sex differentiation
        newborn_female = np.array([newborn_dd, newborn_dw, newborn_ww]) / 2
        newborn_male = np.array([newborn_dd, newborn_dw, newborn_ww]) / 2
        
        # Viability selection
        new_adult_females = np.array([
            newborn_female[0] * female_viability_fitness,
            newborn_female[1],
            newborn_female[2]
        ])
        new_adult_males = np.array([
            newborn_male[0] * male_viability_fitness,
            newborn_male[1],
            newborn_male[2]
        ])
        
        # Normalize
        female_sum = np.sum(new_adult_females)
        male_sum = np.sum(new_adult_males)
        
        expected_new_adult[t, 0:3] = new_adult_females / female_sum if female_sum > 0 else np.array([1/3, 1/3, 1/3])
        expected_new_adult[t, 3:6] = new_adult_males / male_sum if male_sum > 0 else np.array([1/3, 1/3, 1/3])
        
        # Calculate log-likelihood
        ne_half = int(effective_population_size / 2)
        if ne_half > 0:
            log_likelihood_value += rho(
                observed_new_adult[t, 0:3],
                expected_new_adult[t, 0:3],
                ne_half
            )
            log_likelihood_value += rho(
                observed_new_adult[t, 3:6],
                expected_new_adult[t, 3:6],
                ne_half
            )
    
    if return_details:
        # Calculate chi-square statistic
        valid_idx = slice(2, n_weeks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi_sq = np.nansum(
                (observed_new_adult[valid_idx] - expected_new_adult[valid_idx])**2 /
                np.where(expected_new_adult[valid_idx] > 0, expected_new_adult[valid_idx], np.inf)
            )
        return log_likelihood_value, pop_stat, expected_new_adult, observed_new_adult, chi_sq
    
    return log_likelihood_value


def neg_log_likelihood(f: np.ndarray, population_statistic: List[np.ndarray], mode: int) -> float:
    """Negative log-likelihood for minimization."""
    return -log_likelihood(f, population_statistic, mode)


def chi_square_fit(f: np.ndarray, population_statistic: List[np.ndarray], mode: int) -> float:
    """Chi-square goodness of fit statistic."""
    f_with_ne = np.concatenate([[3000], f])
    _, _, expected, observed, chi_sq = log_likelihood(f_with_ne, population_statistic, mode, return_details=True)
    return chi_sq


# =============================================================================
# MLE estimation
# =============================================================================

def estimate_mle(
    population_statistic: List[np.ndarray],
    mode: int,
    initial_params: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> Dict:
    """
    Estimate maximum likelihood parameters.
    
    Parameters
    ----------
    population_statistic : List[np.ndarray]
        Observed population data
    mode : int
        Model type (1-8)
    initial_params : np.ndarray, optional
        Initial parameter values
    bounds : list of tuples, optional
        Parameter bounds
        
    Returns
    -------
    dict
        Optimization results including estimated parameters and log-likelihood
    """
    # Default bounds and initial values by mode
    default_configs = {
        1: {'start': [1000, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1)]},
        2: {'start': [1000, 0.5],
            'bounds': [(500, 2000), (0.01, 1)]},
        3: {'start': [1000, 0.5, 0.5],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1)]},
        4: {'start': [1000, 0.5, 0.5, 0.25],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1), (0.01, 1)]},
        5: {'start': [1000, 0.5, 0.5],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1)]},
        6: {'start': [1000, 0.5, 0.5],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1)]},
        7: {'start': [1000, 0.5],
            'bounds': [(500, 2000), (0.01, 1)]},
        8: {'start': [1000, 0.5, 0.5],
            'bounds': [(500, 2000), (0.01, 1), (0.01, 1)]},
    }
    
    config = default_configs.get(mode, default_configs[1])
    
    if initial_params is None:
        initial_params = np.array(config['start'])
    if bounds is None:
        bounds = config['bounds']
    
    # Optimize using L-BFGS-B
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(population_statistic, mode),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    return {
        'params': result.x,
        'log_likelihood': -result.fun,
        'success': result.success,
        'message': result.message,
        'n_iterations': result.nit
    }


# =============================================================================
# Parallel simulation utilities (for batch runs)
# =============================================================================

def run_batch_simulations(
    initial_genotype_freq: np.ndarray,
    params: SimulationParams,
    n_simulations: int = 100,
    n_jobs: int = -1
) -> List[List[np.ndarray]]:
    """
    Run multiple simulations in parallel.
    
    Parameters
    ----------
    initial_genotype_freq : np.ndarray
        Initial genotype frequencies
    params : SimulationParams
        Simulation parameters
    n_simulations : int
        Number of simulations to run
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
        
    Returns
    -------
    List[List[np.ndarray]]
        Results from all simulations
    """
    try:
        from joblib import Parallel, delayed
        
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count()
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(evolve_overlapping)(initial_genotype_freq, params)
            for _ in range(n_simulations)
        )
        return results
    except ImportError:
        # Fallback to sequential execution
        print("Warning: joblib not installed, running sequentially")
        return [evolve_overlapping(initial_genotype_freq, params) for _ in range(n_simulations)]


# =============================================================================
# Numba-optimized core functions (optional, significant speedup)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _fast_offspring_calculation(
        fem_dd: np.ndarray, fem_dw: np.ndarray, fem_ww: np.ndarray,
        stored_dd: np.ndarray, stored_dw: np.ndarray, stored_ww: np.ndarray,
        f_m: float, f_f: float, f_b: float, eggs: float,
        drive_efficiency: float = 0.5
    ) -> Tuple[float, float, float]:
        """Numba-optimized offspring calculation with drive conversion."""
        e = drive_efficiency
        h_to_w = 1.0 - e
        
        offspring_dd = eggs * (
            f_b * np.sum(fem_dd * stored_dd) +
            f_f * np.sum(fem_dd * stored_dw) * e +
            f_m * np.sum(fem_dw * stored_dd) * e +
            np.sum(fem_dw * stored_dw) * e * e
        )
        
        offspring_dw = eggs * (
            f_f * np.sum(fem_dd * stored_dw) * h_to_w +
            f_f * np.sum(fem_dd * stored_ww) +
            f_m * np.sum(fem_dw * stored_dd) * h_to_w +
            np.sum(fem_dw * stored_dw) * 2 * e * h_to_w +
            np.sum(fem_dw * stored_ww) * e +
            f_m * np.sum(fem_ww * stored_dd) +
            np.sum(fem_ww * stored_dw) * e
        )
        
        offspring_ww = eggs * (
            np.sum(fem_ww * stored_ww) +
            np.sum(fem_ww * stored_dw) * h_to_w +
            np.sum(fem_dw * stored_ww) * h_to_w +
            np.sum(fem_dw * stored_dw) * h_to_w * h_to_w
        )
        
        return offspring_dd, offspring_dw, offspring_ww


# =============================================================================
# Visualization functions
# =============================================================================

def get_d_allele_freq(pop_stat: List[np.ndarray]) -> np.ndarray:
    """
    Extract d allele frequency over time from population statistics.
    
    Parameters
    ----------
    pop_stat : List[np.ndarray]
        Population statistics from simulation
        
    Returns
    -------
    np.ndarray
        Array of d allele frequencies for each week
    """
    freqs = []
    for pop in pop_stat:
        total_dd = np.sum(pop[[FEMALE_DD, MALE_DD], 2:8])
        total_dw = np.sum(pop[[FEMALE_DW, MALE_DW], 2:8])
        total_ww = np.sum(pop[[FEMALE_WW, MALE_WW], 2:8])
        total = total_dd + total_dw + total_ww
        if total > 0:
            freqs.append((2 * total_dd + total_dw) / (2 * total))
        else:
            freqs.append(np.nan)
    return np.array(freqs)


def plot_drive_comparison(
    result_no_drive: List[np.ndarray],
    result_with_drive: List[np.ndarray],
    output_path: str = "drive_comparison.pdf"
) -> None:
    """
    Plot comparison of d allele frequency with and without drive conversion.
    
    Parameters
    ----------
    result_no_drive : List[np.ndarray]
        Simulation result without drive (e=0.5)
    result_with_drive : List[np.ndarray]
        Simulation result with drive (e>0.5)
    output_path : str
        Output PDF file path
    """
    freq_no_drive = get_d_allele_freq(result_no_drive)
    freq_with_drive = get_d_allele_freq(result_with_drive)
    
    weeks = np.arange(len(freq_no_drive))
    
    plt.figure(figsize=(6, 4))
    plt.plot(weeks, freq_no_drive, 'b-', label='No drive (e=0.5)')
    plt.plot(weeks, freq_with_drive, 'r-', label='With drive (e=0.7)')
    plt.xlabel('Week')
    plt.ylabel('Drive allele (d) frequency')
    plt.title('Gene Drive vs Standard Mendelian Inheritance')
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_genotype_trajectory(
    expected_new_adult: np.ndarray,
    start_week: int = 3,
    output_path: str = "track.pdf",
    title: str = ""
) -> None:
    """
    Plot genotype frequency trajectory over time (like R's track_1/2/3.pdf).
    
    Parameters
    ----------
    expected_new_adult : np.ndarray
        Expected new adult genotype frequencies from log_likelihood
    start_week : int
        Starting week to plot
    output_path : str
        Output PDF file path
    title : str
        Plot title
    """
    genotype_names = ['female_dd', 'female_dw', 'female_ww', 
                      'male_dd', 'male_dw', 'male_ww']
    
    # Extract data from start_week onwards
    data = expected_new_adult[start_week:, :]
    weeks = np.arange(start_week, start_week + len(data))
    
    plt.figure(figsize=(5, 3))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    for i, (name, marker, color) in enumerate(zip(genotype_names, markers, colors)):
        values = data[:, i]
        # Avoid log(0)
        values = np.maximum(values, 1e-10)
        plt.semilogy(weeks, values, marker=marker, color=color, 
                    label=name, markersize=4, linewidth=1)
    
    plt.xlabel('Weeks')
    plt.ylabel('Genotype frequency')
    plt.title(title)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    output_path: str,
    cmap_low: str = "green",
    cmap_high: str = "white",
    x_label: str = None,
    y_label: str = None,
    value_label: str = None
) -> None:
    """
    Create a heatmap from batch simulation results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with x, y, and value columns
    x_col, y_col : str
        Column names for x and y axes
    value_col : str
        Column name for heatmap values
    output_path : str
        Output PDF file path
    cmap_low, cmap_high : str
        Colors for gradient (low to high)
    x_label, y_label, value_label : str
        Axis labels
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create pivot table for heatmap
    pivot = results_df.pivot(index=y_col, columns=x_col, values=value_col)
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", [cmap_low, cmap_high])
    
    plt.figure(figsize=(3, 3))
    plt.imshow(pivot.values, aspect='equal', origin='lower', cmap=cmap)
    plt.colorbar(label=value_label or value_col)
    
    # Set tick labels
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    plt.xticks(range(len(x_vals)), [f'{v:.2f}' for v in x_vals], fontsize=6, rotation=45)
    plt.yticks(range(len(y_vals)), [f'{v:.2f}' for v in y_vals], fontsize=6)
    
    plt.xlabel(x_label or x_col)
    plt.ylabel(y_label or y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def run_batch_mle_test(
    initial_genotype_freq: np.ndarray,
    fitness_values: np.ndarray,
    selection_type: str = "viability",
    n_simulations: int = 3,
    n_e: int = 1000,
    max_week: int = 10,
    output_dir: str = "."
) -> pd.DataFrame:
    """
    Run batch MLE tests over a grid of fitness values (like R's 26x26 tests).
    
    Parameters
    ----------
    initial_genotype_freq : np.ndarray
        Initial genotype frequencies
    fitness_values : np.ndarray
        Array of fitness values to test
    selection_type : str
        "viability" or "fecundity"
    n_simulations : int
        Number of simulations per parameter combination
    n_e : int
        Effective population size
    max_week : int
        Maximum weeks to simulate
    output_dir : str
        Output directory for CSV and PDF files
        
    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    results = []
    
    mode = 3 if selection_type == "viability" else 8
    
    total_combos = len(fitness_values) ** 2
    combo_count = 0
    
    for male_fitness in fitness_values:
        for female_fitness in fitness_values:
            combo_count += 1
            
            MLE_values = []
            MLE_Ne = []
            MLE_male_fitness = []
            MLE_female_fitness = []
            fit_values = []
            fit_male_fitness = []
            fit_female_fitness = []
            
            for _ in range(n_simulations):
                if selection_type == "viability":
                    params = SimulationParams(
                        effective_population_size=n_e,
                        male_viability_fitness=male_fitness,
                        female_viability_fitness=female_fitness,
                        max_week=max_week
                    )
                else:
                    params = SimulationParams(
                        effective_population_size=n_e,
                        male_fecundity_fitness=male_fitness,
                        female_fecundity_fitness=female_fitness,
                        both_fecundity_fitness=male_fitness * female_fitness,
                        max_week=max_week
                    )
                
                res = evolve_overlapping(initial_genotype_freq, params)
                
                # MLE estimation
                mle_result = estimate_mle(res, mode=mode)
                MLE_values.append(mle_result['log_likelihood'])
                MLE_Ne.append(mle_result['params'][0])
                MLE_male_fitness.append(mle_result['params'][1])
                MLE_female_fitness.append(mle_result['params'][2])
                
                # Chi-square fit
                try:
                    chi_result = minimize(
                        chi_square_fit,
                        np.array([0.5, 0.5]),
                        args=(res, mode),
                        method='L-BFGS-B',
                        bounds=[(0.01, 1), (0.01, 1)]
                    )
                    fit_values.append(chi_result.fun)
                    fit_male_fitness.append(chi_result.x[0])
                    fit_female_fitness.append(chi_result.x[1])
                except:
                    fit_values.append(np.nan)
                    fit_male_fitness.append(np.nan)
                    fit_female_fitness.append(np.nan)
            
            results.append({
                f'male_{selection_type}_fitness': male_fitness,
                f'female_{selection_type}_fitness': female_fitness,
                'mean_MLE': np.mean(MLE_values),
                'mean_MLE_Ne': np.mean(MLE_Ne),
                f'mean_MLE_male_{selection_type}_fitness': np.mean(MLE_male_fitness),
                f'mean_MLE_female_{selection_type}_fitness': np.mean(MLE_female_fitness),
                'mean_fit': np.mean(fit_values),
                f'mean_fit_male_{selection_type}_fitness': np.mean(fit_male_fitness),
                f'mean_fit_female_{selection_type}_fitness': np.mean(fit_female_fitness),
            })
    
    df = pd.DataFrame(results)
    
    # Calculate Euclidean distances
    male_col = f'male_{selection_type}_fitness'
    female_col = f'female_{selection_type}_fitness'
    mle_male_col = f'mean_MLE_male_{selection_type}_fitness'
    mle_female_col = f'mean_MLE_female_{selection_type}_fitness'
    fit_male_col = f'mean_fit_male_{selection_type}_fitness'
    fit_female_col = f'mean_fit_female_{selection_type}_fitness'
    
    df['distance'] = np.sqrt(
        (df[male_col] - df[mle_male_col])**2 + 
        (df[female_col] - df[mle_female_col])**2
    )
    df['distance_fit'] = np.sqrt(
        (df[male_col] - df[fit_male_col])**2 + 
        (df[female_col] - df[fit_female_col])**2
    )
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'results_{selection_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Generate heatmaps
    plot_heatmap(
        df, male_col, female_col, 'distance',
        os.path.join(output_dir, f'{selection_type}_heatmap.pdf'),
        cmap_low='green', cmap_high='white',
        x_label=f'Male {selection_type.title()} Fitness',
        y_label=f'Female {selection_type.title()} Fitness',
        value_label='Distance'
    )
    
    plot_heatmap(
        df, male_col, female_col, 'distance_fit',
        os.path.join(output_dir, f'{selection_type}_heatmap_fit.pdf'),
        cmap_low='red', cmap_high='white',
        x_label=f'Male {selection_type.title()} Fitness',
        y_label=f'Female {selection_type.title()} Fitness',
        value_label='Distance'
    )
    
    plot_heatmap(
        df, male_col, female_col, 'mean_MLE_Ne',
        os.path.join(output_dir, f'Ne_{selection_type[0]}.pdf'),
        cmap_low='yellow', cmap_high='blue',
        x_label=f'Male {selection_type.title()} Fitness',
        y_label=f'Female {selection_type.title()} Fitness',
        value_label='N_e'
    )
    
    return df


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mosquito fitness MLE estimation')
    parser.add_argument('--batch', action='store_true', help='Run batch MLE tests with heatmaps')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for PDFs')
    args = parser.parse_args()
    
    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initial genotype frequency matrix
    initial_genotype_freq = np.array([
        [0, 0.085714286, 0, 0, 0, 0, 0, 0],                                    # female_dd
        [0, 0, 0, 0, 0, 0, 0, 0],                                              # female_dw
        [0, 0.2, 0.285714286, 0.238095238, 0.19047619, 0.142857143, 0.095238095, 0.047619048],  # female_ww
        [0, 0.085714286, 0, 0, 0, 0, 0, 0],                                    # male_dd
        [0, 0, 0, 0, 0, 0, 0, 0],                                              # male_dw
        [0, 0.2, 0.285714286, 0.19047619, 0.095238095, 0, 0, 0],               # male_ww
    ])
    
    np.random.seed(42)
    
    # =========================================================================
    # 1. Drive comparison simulation and plot
    # =========================================================================
    print("=" * 60)
    print("1. Gene Drive Comparison")
    print("=" * 60)
    
    params_no_drive = SimulationParams(
        effective_population_size=300,
        male_viability_fitness=0.8,
        female_viability_fitness=0.8,
        max_week=32,
        drive_efficiency=0.5  # Standard Mendelian
    )
    
    params_with_drive = SimulationParams(
        effective_population_size=300,
        male_viability_fitness=0.8,
        female_viability_fitness=0.8,
        max_week=32,
        drive_efficiency=0.7  # Gene drive
    )
    
    result_no_drive = evolve_overlapping(initial_genotype_freq, params_no_drive)
    result_with_drive = evolve_overlapping(initial_genotype_freq, params_with_drive)
    
    # Plot drive comparison (like R's plot in chunk 7)
    plot_drive_comparison(
        result_no_drive, result_with_drive,
        os.path.join(output_dir, "drive_comparison.pdf")
    )
    
    # Print brief summary (like R output)
    freq_no_drive = get_d_allele_freq(result_no_drive)
    freq_with_drive = get_d_allele_freq(result_with_drive)
    print(f"Week 30: No drive={freq_no_drive[30]:.3f}, With drive={freq_with_drive[30]:.3f}")
    
    # =========================================================================
    # 2. Genotype trajectory plots (track_1, track_2, track_3)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Genotype Trajectory Plots")
    print("=" * 60)
    
    # Viability selection only (track_1)
    params1 = SimulationParams(
        effective_population_size=0,  # Deterministic
        male_viability_fitness=0.6,
        female_viability_fitness=0.8,
        max_week=10
    )
    res1 = evolve_overlapping(initial_genotype_freq, params1)
    _, _, expected1, _, _ = log_likelihood(np.array([300, 0.6, 0.8]), res1, mode=3, return_details=True)
    plot_genotype_trajectory(expected1, start_week=3, 
                            output_path=os.path.join(output_dir, "track_1.pdf"),
                            title="Viability Selection")
    
    # Fecundity selection only (track_2)
    params2 = SimulationParams(
        effective_population_size=0,
        male_fecundity_fitness=0.6,
        female_fecundity_fitness=0.8,
        both_fecundity_fitness=0.48,
        max_week=10
    )
    res2 = evolve_overlapping(initial_genotype_freq, params2)
    _, _, expected2, _, _ = log_likelihood(np.array([300, 0.6, 0.8]), res2, mode=8, return_details=True)
    plot_genotype_trajectory(expected2, start_week=3,
                            output_path=os.path.join(output_dir, "track_2.pdf"),
                            title="Fecundity Selection")
    
    # Sexual selection only (track_3)
    params3 = SimulationParams(
        effective_population_size=0,
        male_mating_success=0.6,
        max_week=10
    )
    res3 = evolve_overlapping(initial_genotype_freq, params3)
    _, _, expected3, _, _ = log_likelihood(np.array([300, 0.6]), res3, mode=2, return_details=True)
    plot_genotype_trajectory(expected3, start_week=3,
                            output_path=os.path.join(output_dir, "track_3.pdf"),
                            title="Sexual Selection")
    
    # =========================================================================
    # 3. MLE estimation test
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. MLE Estimation Test")
    print("=" * 60)
    
    mle_result = estimate_mle(result_no_drive, mode=3)
    print(f"Estimated: Ne={mle_result['params'][0]:.0f}, v.m={mle_result['params'][1]:.3f}, v.f={mle_result['params'][2]:.3f}")
    print(f"True:      Ne=300, v.m=0.800, v.f=0.800")
    print(f"LogL: {mle_result['log_likelihood']:.2f}")
    
    # =========================================================================
    # 4. Batch MLE tests with heatmaps (optional, slow)
    # =========================================================================
    if args.batch:
        print("\n" + "=" * 60)
        print("4. Batch MLE Tests (generating heatmaps)")
        print("=" * 60)
        
        # Use smaller grid for faster demo
        fitness_values = np.arange(0.5, 1.01, 0.1)
        
        print("\nViability selection tests...")
        df_viability = run_batch_mle_test(
            initial_genotype_freq, fitness_values,
            selection_type="viability",
            n_simulations=3,
            output_dir=output_dir
        )
        print(f"Mean distance: {df_viability['distance'].mean():.3f}")
        
        print("\nFecundity selection tests...")
        df_fecundity = run_batch_mle_test(
            initial_genotype_freq, fitness_values,
            selection_type="fecundity",
            n_simulations=3,
            output_dir=output_dir
        )
        print(f"Mean distance: {df_fecundity['distance'].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("Done! Check output directory for PDF files.")
    print("=" * 60)
