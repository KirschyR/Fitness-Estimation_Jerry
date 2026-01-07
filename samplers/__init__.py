"""Samplers module - MCMC and particle filtering methods for population models."""

from samplers.helpers import particle_filter, systematic_resampling
from samplers.observation import ObservationFilter, apply_rule
from samplers.pmcmc import (
    # 结果容器
    PMCMCResult,
    # 核心纯函数
    gaussian_obs_loglik,
    mh_accept,
    random_walk_proposal,
    # 先验分布
    log_uniform_prior,
    log_normal_prior,
    # 遗传模型接口
    make_init_sampler,
    make_transition_fn,
    make_obs_loglik_fn,
    # 主函数
    run_pmcmc,
    # 类
    PMCMC,
    # 便利函数
    create_pmcmc_from_population,
    make_fitness_modifier,
)

__all__ = [
    # helpers
    'particle_filter', 'systematic_resampling',
    # observation
    'ObservationFilter', 'apply_rule',
    # pmcmc
    'PMCMCResult',
    'gaussian_obs_loglik', 'mh_accept', 'random_walk_proposal',
    'log_uniform_prior', 'log_normal_prior',
    'make_init_sampler', 'make_transition_fn', 'make_obs_loglik_fn',
    'run_pmcmc', 'PMCMC',
    'create_pmcmc_from_population', 'make_fitness_modifier',
]
