"""Particle MCMC (PMCMC) 采样器
================================

本模块实现 Particle Marginal Metropolis-Hastings (PMMH) 算法，用于对具有
隐马尔可夫结构的状态空间模型进行贝叶斯参数推断。

PMMH 算法概述
-------------
PMMH 是一种将粒子滤波（Bootstrap Particle Filter）嵌入到 MCMC 框架中的算法。
它利用粒子滤波产生的似然估计（虽然是有偏的）来近似目标后验分布 p(θ|y)。

核心思想是：用 N 个粒子运行序贯蒙特卡洛（SMC），得到边际似然的无偏估计：

    p̂(y₁:T | θ) = ∏_{t=1}^{T} [ (1/N) ∑_{i=1}^{N} w_t^{(i)} ]

然后在 MH 框架中使用这个估计。Andrieu et al. (2010) 证明了即使使用有限粒子数，
PMMH 仍然以精确后验 p(θ|y) 为平稳分布。

模块结构
--------
本模块采用分层设计，兼顾易用性和性能：

1. **核心计算层** (Numba 友好):
   - `gaussian_obs_loglik()`: 高斯观测对数似然
   - `mh_accept()`: Metropolis-Hastings 接受判断
   - 这些函数用 `@numba_switchable` 装饰，可选 JIT 编译

2. **算法层** (纯函数):
   - `run_pmcmc()`: PMMH 算法的完整实现
   - `random_walk_proposal()`: 随机游走提议分布
   - `log_uniform_prior()`, `log_normal_prior()`: 先验分布

3. **接口层** (工厂函数):
   - `make_init_sampler()`: 创建初始状态采样器
   - `make_transition_fn()`: 创建状态转移函数
   - `make_obs_loglik_fn()`: 创建观测似然函数

4. **便利层** (类封装):
   - `PMCMC`: 封装所有组件，提供简洁的用户接口
   - `create_pmcmc_from_population()`: 从 Population 对象快速创建

使用示例
--------
方式 1: 使用纯函数 (更灵活，适合定制)::

    from samplers.pmcmc import run_pmcmc, make_init_sampler, make_transition_fn

    # 设置组件
    init_sampler = make_init_sampler(initial_state, perturbation=0.1)
    transition_fn = make_transition_fn(config, shapes, rng)
    obs_loglik_fn = make_obs_loglik_fn(sigma=1.0, obs_rule, apply_rule)

    # 运行采样
    result = run_pmcmc(
        observations=observations,
        n_particles=200,
        init_sampler=init_sampler,
        transition_fn=transition_fn,
        obs_loglik_fn=obs_loglik_fn,
        theta_init=np.array([0.1, 0.2]),
        n_iter=5000,
        step_sizes=np.array([0.01, 0.01]),
        log_prior_fn=lambda theta: log_uniform_prior(theta, bounds),
        burnin=1000
    )

方式 2: 使用类封装 (更简洁，快速原型)::

    from samplers.pmcmc import create_pmcmc_from_population

    sampler = create_pmcmc_from_population(pop, observations)
    result = sampler.run(theta_init, n_iter=5000, step_sizes, log_prior_fn)

    # 访问结果
    print(f"接受率: {result.acceptance_rate:.3f}")
    posterior_mean = result.theta_chain.mean(axis=0)

参考文献
# TODO: 核查真实性
--------
.. [1] Andrieu, C., Doucet, A., & Holenstein, R. (2010). 
       Particle Markov chain Monte Carlo methods. 
       Journal of the Royal Statistical Society: Series B, 72(3), 269-342.

.. [2] Doucet, A., & Johansen, A. M. (2009). 
       A tutorial on particle filtering and smoothing: Fifteen years later.
       Handbook of nonlinear filtering, 12(656-704), 3.

.. [3] Roberts, G. O., & Rosenthal, J. S. (2009). 
       Examples of adaptive MCMC. 
       Journal of Computational and Graphical Statistics, 18(2), 349-367.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any, NamedTuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from samplers.helpers import particle_filter
from utils.numba_utils import numba_switchable
from utils import simulation_kernels as sk

if TYPE_CHECKING:
    from utils.nonWF_population import AgeStructuredPopulation
    from utils.index_core import IndexCore
    from utils.genetic_structures import Species


# =============================================================================
# 结果容器
# =============================================================================

class PMCMCResult(NamedTuple):
    """PMCMC 采样结果的命名元组容器。
    
    该类封装了 PMMH 算法的所有输出，包括参数链、对数似然链、
    先验链以及接受率统计。使用 NamedTuple 确保不可变性和内存效率。
    
    Attributes:
        theta_chain (NDArray): 参数样本链。
            形状为 (n_samples, n_params)，其中 n_samples = (n_iter - burnin) / thin。
            每一行是一个参数向量样本。可用于计算后验统计量。
        loglik_chain (NDArray): 对数似然链。
            形状为 (n_samples,)。每个值是对应参数下粒子滤波估计的 log p̂(y|θ)。
            注意这是似然的蒙特卡洛估计，存在一定方差。
        logprior_chain (NDArray): 对数先验链。
            形状为 (n_samples,)。每个值是对应参数的 log p(θ)。
        accepted (NDArray): 接受标记数组。
            形状为 (n_samples,)，dtype=bool。True 表示该迭代接受了提议。
            可用于诊断采样器混合性。
        acceptance_rate (float): 总体接受率。
            范围 [0, 1]。最优值通常在 0.2-0.3 附近（对于高维问题）。
            过高（>0.5）表示步长太小，过低（<0.1）表示步长太大。
    
    Example:
        >>> result = run_pmcmc(...)
        >>> # 计算后验均值和标准差
        >>> posterior_mean = result.theta_chain.mean(axis=0)
        >>> posterior_std = result.theta_chain.std(axis=0)
        >>> # 检查收敛性
        >>> print(f"接受率: {result.acceptance_rate:.3f}")
        >>> # 可视化参数轨迹
        >>> plt.plot(result.theta_chain[:, 0], label='param_0')
    """
    theta_chain: NDArray      # (n_samples, n_params) 参数样本链
    loglik_chain: NDArray     # (n_samples,) 对数似然链
    logprior_chain: NDArray   # (n_samples,) 对数先验链
    accepted: NDArray         # (n_samples,) bool 接受标记
    acceptance_rate: float    # 总体接受率


# =============================================================================
# 核心纯函数（Numba 友好）
# =============================================================================

@numba_switchable
def gaussian_obs_loglik(
    obs: NDArray, 
    predicted: NDArray, 
    sigma: float
) -> float:
    """计算高斯（正态）观测模型的对数似然。
    
    假设观测误差服从独立同分布的正态分布：
    
        y_i = μ_i + ε_i,  ε_i ~ N(0, σ²)
    
    则联合对数似然为：
    
        log p(y | μ, σ) = -n/2 * log(2πσ²) - 1/(2σ²) * Σ(y_i - μ_i)²
    
    该函数用 @numba_switchable 装饰，根据配置决定是否启用 JIT 编译。
    
    Args:
        obs (NDArray): 实际观测值向量。
            可以是任意形状，内部会展平为一维。
        predicted (NDArray): 模型预测值向量。
            形状应与 obs 相同或可广播兼容。
        sigma (float): 观测标准差。
            必须为正数。较小的 sigma 意味着观测更精确。
    
    Returns:
        float: 对数似然值 log p(obs | predicted, σ)。
            值域为 (-∞, 0]。完美预测时返回 -n/2 * log(2πσ²)。
    
    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> pred = np.array([1.1, 1.9, 3.1])  # 小误差
        >>> loglik = gaussian_obs_loglik(obs, pred, sigma=0.5)
        >>> print(f"log p = {loglik:.4f}")
    
    Notes:
        - 该函数假设各维度观测误差独立（对角协方差矩阵）
        - 对于异方差情况，需要扩展为每维不同的 sigma
        - Numba 编译会在首次调用时产生编译开销
    """
    # 展平数组以统一处理不同形状的输入
    diff = obs.ravel() - predicted.ravel()
    n = len(diff)
    
    # 对数似然: -n/2 * log(2πσ²) - 1/(2σ²) * ||diff||²
    # = -n/2 * log(2π) - n/2 * log(σ²) - SSE/(2σ²)
    return -0.5 * n * np.log(2.0 * np.pi * sigma**2) - 0.5 * np.sum(diff**2) / sigma**2


@numba_switchable
def mh_accept(
    loglik_prop: float, logprior_prop: float,
    loglik_curr: float, logprior_curr: float,
    log_u: float
) -> bool:
    """Metropolis-Hastings 接受/拒绝判断。
    
    实现标准的 MH 接受准则。对于对称提议分布 q(θ'|θ) = q(θ|θ')，
    接受概率为：
    
        α(θ' | θ) = min(1, [p(y|θ') * p(θ')] / [p(y|θ) * p(θ)])
    
    在对数空间中实现以避免数值下溢：
    
        log α = [log p(y|θ') + log p(θ')] - [log p(y|θ) + log p(θ)]
        
    接受条件：log(u) < log α，其中 u ~ Uniform(0, 1)
    
    该函数用 @numba_switchable 装饰，是 MCMC 内循环中的热点函数。
    
    Args:
        loglik_prop (float): 提议参数的对数似然 log p(y|θ')。
            通常由粒子滤波估计得到。
        logprior_prop (float): 提议参数的对数先验 log p(θ')。
        loglik_curr (float): 当前参数的对数似然 log p(y|θ)。
        logprior_curr (float): 当前参数的对数先验 log p(θ)。
        log_u (float): 均匀随机数的对数 log(u)，u ~ Uniform(0,1)。
            传入 log(u) 而非 u 以避免重复计算对数。
    
    Returns:
        bool: True 表示接受提议，False 表示拒绝。
    
    Example:
        >>> # 提议比当前好得多
        >>> accept = mh_accept(-100, -1, -200, -1, np.log(0.5))
        >>> print(accept)  # True
        >>> # 提议比当前差很多
        >>> accept = mh_accept(-200, -1, -100, -1, np.log(0.5))
        >>> print(accept)  # False（几乎一定拒绝）
    
    Notes:
        - 如果 loglik_prop 或 logprior_prop 为 -inf 或 nan，直接拒绝
        - 对于非对称提议，需要额外加上 log q(θ|θ') - log q(θ'|θ) 修正
        - 在 PMMH 中，log p(y|θ) 是粒子滤波的估计，有随机性
    """
    # 快速检查：如果提议落在先验支撑外或似然无效，直接拒绝
    if not np.isfinite(loglik_prop) or not np.isfinite(logprior_prop):
        return False
    
    # 计算对数接受率 (log-Hastings ratio)
    # log α = log[p(y|θ') p(θ')] - log[p(y|θ) p(θ)]
    log_alpha = (loglik_prop + logprior_prop) - (loglik_curr + logprior_curr)
    
    # 接受条件: log(u) < log(α)，等价于 u < α
    return log_u < log_alpha


def random_walk_proposal(
    theta: NDArray, 
    step_sizes: NDArray, 
    rng: np.random.Generator,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> NDArray:
    """生成随机游走 Metropolis 提议。
    
    从以当前点为中心的正态分布中采样：
    
        θ' ~ N(θ, Σ),  Σ = diag(step_sizes²)
    
    这是对称提议分布，因此 MH 比率中的 q(θ|θ')/q(θ'|θ) = 1。
    
    当指定边界时，使用反射边界条件（reflecting boundary）：
    如果提议落在边界外，将其"反弹"回来。这保持了提议的对称性，
    避免了在边界附近引入偏差。
    
    Args:
        theta (NDArray): 当前参数向量，形状为 (n_params,)。
        step_sizes (NDArray): 每个参数的步长（标准差），形状为 (n_params,)。
            步长的选择非常重要：
            - 太小：接受率高但混合慢
            - 太大：接受率低
            - 理想接受率约 23.4%（高维）到 44%（一维）
        rng (np.random.Generator): NumPy 随机数生成器。
            使用 Generator 而非传统 RandomState 以获得更好的随机性。
        bounds (List[Tuple[float, float]], optional): 参数边界列表。
            每个元素是 (lower, upper) 元组。如果为 None，则无边界约束。
    
    Returns:
        NDArray: 提议的新参数向量 θ'，形状与 theta 相同。
    
    Example:
        >>> rng = np.random.default_rng(42)
        >>> theta = np.array([0.5, 1.0])
        >>> step_sizes = np.array([0.1, 0.1])
        >>> bounds = [(0.0, 1.0), (0.0, 2.0)]
        >>> theta_new = random_walk_proposal(theta, step_sizes, rng, bounds)
        >>> print(theta_new)  # 在边界内的新值
    
    Notes:
        - 反射边界可能在极端情况下需要多次反弹，此处使用 clip 作为安全措施
        - 对于硬边界约束，也可考虑变换到无约束空间（如 logit 变换）
        - 步长自适应（见 run_pmcmc 中的 adapt 参数）可自动调整步长
    """
    # 从正态分布采样增量: Δθ ~ N(0, diag(step_sizes²))
    theta_new = theta + rng.normal(0, step_sizes)
    
    # 处理边界约束（如果指定）
    if bounds is not None:
        for i, (lo, hi) in enumerate(bounds):
            # 反射边界：将越界值"弹回"
            # 例如: 如果 theta_new[i] = lo - 0.1，反射到 lo + 0.1
            if theta_new[i] < lo:
                theta_new[i] = lo + (lo - theta_new[i])
            elif theta_new[i] > hi:
                theta_new[i] = hi - (theta_new[i] - hi)
            # 安全裁剪：防止反射后仍越界（步长过大时）
            theta_new[i] = np.clip(theta_new[i], lo, hi)
    
    return theta_new


# =============================================================================
# 先验分布
# =============================================================================

def log_uniform_prior(theta: NDArray, bounds: List[Tuple[float, float]]) -> float:
    """计算均匀（无信息）先验的对数密度。
    
    均匀先验表示在给定范围内没有偏好，是一种典型的"无信息"先验。
    对于第 i 个参数在范围 [a_i, b_i] 上的均匀分布：
    
        p(θ_i) = 1/(b_i - a_i) 如果 a_i ≤ θ_i ≤ b_i，否则 0
    
    假设各参数独立，则联合先验：
    
        p(θ) = ∏_i p(θ_i) = 1 / ∏_i (b_i - a_i)
        
        log p(θ) = -Σ_i log(b_i - a_i)
    
    Args:
        theta (NDArray): 参数向量，形状为 (n_params,)。
        bounds (List[Tuple[float, float]]): 参数边界列表。
            每个元素 (lower, upper) 定义一个参数的有效范围。
            列表长度必须等于 theta 的维度。
    
    Returns:
        float: 对数先验密度 log p(θ)。
            - 如果所有参数在边界内，返回 -Σ log(b_i - a_i)
            - 如果任意参数越界，返回 -inf
    
    Example:
        >>> theta = np.array([0.5, 1.0])
        >>> bounds = [(0.0, 1.0), (0.0, 2.0)]  # 第一个在 [0,1]，第二个在 [0,2]
        >>> log_p = log_uniform_prior(theta, bounds)
        >>> print(f"log p = {log_p:.4f}")  # = -log(1) - log(2) = -log(2)
        >>> # 越界情况
        >>> log_p = log_uniform_prior(np.array([1.5, 1.0]), bounds)
        >>> print(log_p)  # -inf
    
    Notes:
        - 均匀先验对应于最大熔原则下的无信息先验
        - 对于位置参数合理，对于尺度参数可考虑对数均匀先验
        - 边界选择影响后验，应基于领域知识合理设定
    """
    # 检查所有参数是否在边界内
    for i, (lo, hi) in enumerate(bounds):
        if theta[i] < lo or theta[i] > hi:
            return -np.inf  # 越界，先验密度为 0
    
    # 计算标准化常数的对数: log(1/V) = -log(V) = -Σ log(Δ_i)
    return -sum(np.log(hi - lo) for lo, hi in bounds)


def log_normal_prior(
    theta: NDArray, 
    means: NDArray, 
    stds: NDArray,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> float:
    """计算正态（高斯）先验的对数密度。
    
    正态先验表示对参数有某种"软"的先验信念，以平均值为中心，
    其不确定性由标准差度量。每个参数独立，所以：
    
        θ_i ~ N(μ_i, σ_i²)
        
        log p(θ_i) = -1/2 * [(θ_i - μ_i)/σ_i]² - 1/2 * log(2π) - log(σ_i)
    
    联合先验（假设独立）：
    
        log p(θ) = Σ_i log p(θ_i)
                 = -1/2 * Σ_i z_i² - n/2 * log(2π) - Σ_i log(σ_i)
    
    其中 z_i = (θ_i - μ_i)/σ_i 是标准化值。
    
    可选地，可以结合边界约束形成"截断正态"先验，
    在边界外返回 -inf。
    
    Args:
        theta (NDArray): 参数向量，形状为 (n_params,)。
        means (NDArray): 先验平均值向量，形状为 (n_params,)。
            表示对参数的最佳估计或先验信念中心。
        stds (NDArray): 先验标准差向量，形状为 (n_params,)。
            表示对参数的不确定性。较大的标准差表示更"弱"的先验。
        bounds (List[Tuple[float, float]], optional): 参数边界列表。
            如果指定，形成截断正态先验。默认 None（无截断）。
    
    Returns:
        float: 对数先验密度 log p(θ)。
            - 如果在边界内，返回完整的正态对数密度
            - 如果越界，返回 -inf
    
    Example:
        >>> theta = np.array([0.0, 0.0])
        >>> means = np.array([0.0, 0.0])
        >>> stds = np.array([1.0, 1.0])
        >>> log_p = log_normal_prior(theta, means, stds)
        >>> print(f"log p = {log_p:.4f}")  # = -log(2π) ≈ -1.8379
        >>> # 带边界
        >>> bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        >>> log_p = log_normal_prior(theta, means, stds, bounds)
    
    Notes:
        - 这里实现的是独立先验（对角协方差）
        - 对于相关先验，需要扩展为多元正态分布
        - 截断正态的标准化常数未包含（用于 MCMC 时不影响结果）
    """
    # 如果有边界，先检查是否越界
    if bounds is not None:
        for i, (lo, hi) in enumerate(bounds):
            if theta[i] < lo or theta[i] > hi:
                return -np.inf  # 越界，先验密度为 0
    
    # 计算标准化值: z_i = (θ_i - μ_i) / σ_i
    z = (theta - means) / stds
    
    # log p = -0.5 * Σ z_i² - 0.5 * n * log(2π) - Σ log(σ_i)
    return -0.5 * np.sum(z**2) - 0.5 * len(theta) * np.log(2*np.pi) - np.sum(np.log(stds))


# =============================================================================
# 遗传模型接口函数
# =============================================================================

def make_init_sampler(
    initial_state: Tuple, 
    perturbation: float = 0.0,
    seed: int = 42
) -> Callable[[int], NDArray]:
    """创建粒子滤波的初始状态采样器。
    
    粒子滤波需要一个函数来生成 N 个初始粒子。本函数基于给定的
    初始状态创建该采样器，可选地添加一些扰动以增加多样性。
    
    初始状态使用种群动态模型的完整状态格式（来自 simulation_kernels）：
    - individual_count: 个体计数数组 (n_sexes, n_ages, n_genotypes)
    - sperm_storage: 精子储存数组 (n_ages, n_genotypes, n_haplotypes * n_glabs)
    - female_occupancy: 雌性占据数组 (n_ages, n_genotypes)
    - tick: 时间步（整数）
    
    采样器将完整状态展平为一维向量作为粒子状态，格式为：
        [ind_count.ravel(), sperm_storage.ravel(), female_occupancy.ravel(), tick]
    
    随机数生成
    ----------
    为了 Numba 兼容性，本函数使用 seed + counter 的模式来生成随机数，
    而非 np.random.Generator。每次调用采样器时，counter 会递增，
    确保不同调用产生不同的随机序列，同时保持可重复性。
    
    Args:
        initial_state (Tuple): 初始状态元组 (ind_count, sperm, fem_occ, tick)。
            通常由 sk.export_state(population) 获得。
        perturbation (float, optional): 扰动比例。默认 0.0（无扰动）。
            当 > 0 时，对 individual_count 部分添加噪声。
            扰动仅应用于 individual_count，其他状态保持不变。
        seed (int, optional): 随机数种子。默认 42。
            用于生成可重复的随机扰动。
    
    Returns:
        Callable[[int], NDArray]: 初始状态采样函数。
            输入: N (粒子数)
            输出: (N, state_dim) 数组，每行是一个完整粒子状态的展平表示
    
    Example:
        >>> state, config, _ = sk.export_state(population)
        >>> # 无扰动：所有粒子相同
        >>> sampler = make_init_sampler(state, perturbation=0.0)
        >>> particles = sampler(100)  # (100, state_dim)
        >>> # 带扰动：粒子有差异（仅 individual_count 部分）
        >>> sampler = make_init_sampler(state, perturbation=0.1, seed=123)
        >>> particles = sampler(100)
    
    Notes:
        - 扰动仅应用于 individual_count，sperm_storage 和 female_occupancy 保持原值
        - 计数裁剪为非负（种群数不能为负）
        - 使用 seed + counter 模式保证 Numba 兼容性和可重复性
    """
    # 提取各状态分量并展平
    ind_count = initial_state[sk.STATE_INDIVIDUAL_COUNT].ravel().astype(np.float64)
    sperm_storage = initial_state[sk.STATE_SPERM_STORAGE].ravel().astype(np.float64)
    female_occ = initial_state[sk.STATE_FEMALE_OCC].ravel().astype(np.float64)
    tick = float(initial_state[sk.STATE_TICK])
    
    # 拼接为完整的展平状态向量
    # 格式: [ind_count, sperm_storage, female_occ, tick]
    base = np.concatenate([ind_count, sperm_storage, female_occ, [tick]])
    
    # 记录各分量的长度，用于后续重建
    ind_len = len(ind_count)
    
    # 使用列表包装 counter 以便在闭包中修改
    counter = [0]
    
    def sampler(N: int) -> NDArray:
        """采样 N 个初始粒子状态（完整状态）"""
        # 复制基础状态 N 次
        states = np.tile(base, (N, 1))
        
        # 如果需要扰动，仅对 individual_count 部分添加正态噪声
        if perturbation > 0:
            # 使用 seed + counter 生成随机数（Numba 友好）
            rng_seed = seed + counter[0]
            np.random.seed(rng_seed)
            counter[0] += 1
            
            # 噪声标准差与计数的平方根成比例
            noise_std = perturbation * np.sqrt(np.maximum(ind_count, 1))
            noise = np.random.normal(0, noise_std, (N, ind_len))
            # 仅更新 individual_count 部分
            states[:, :ind_len] = np.maximum(0, states[:, :ind_len] + noise)
        
        return states
    
    return sampler


def make_transition_fn(
    config: Tuple,
    shapes: Tuple[Tuple, Tuple, Tuple],  # (ind_count_shape, sperm_shape, fem_occ_shape)
    seed: int = 42,
    config_modifier: Optional[Callable[[NDArray, Tuple], Tuple]] = None
) -> Callable[[NDArray, Any], NDArray]:
    """创建粒子滤波的状态转移函数。
    
    转移函数接收当前时刻的粒子状态（完整状态的展平表示），返回下一时刻的状态。
    内部调用 simulation_kernels.run_tick() 执行一步种群动态模拟：
    
        繁殖 → 存活 → 老化
    
    由于 run_tick 是纯函数，支持 Numba JIT 编译，
    可以在大规模粒子滤波中获得显著加速。
    
    参数依赖模型
    -----------
    当提供 config_modifier 时，转移函数会根据当前参数 theta 动态修改配置。
    这允许在 PMCMC 中估计影响状态转移的参数（如适应度、繁殖力等）。
    
    使用 make_fitness_modifier() 创建修改器：
    
        modifier = make_fitness_modifier(['drive_viability'], {'drive_viability': [1,2,3]})
        transition = make_transition_fn(config, shapes, config_modifier=modifier)
    
    状态格式
    --------
    展平的粒子状态格式为：
        [ind_count.ravel(), sperm_storage.ravel(), female_occ.ravel(), tick]
    
    转移函数会将其重建为完整的状态元组传递给 run_tick。
    
    随机数生成
    ----------
    为了 Numba 兼容性，本函数使用 seed + counter 的模式来生成随机数，
    而非 np.random.Generator。每次调用转移函数时，counter 会递增，
    确保不同调用和不同粒子产生不同的随机序列，同时保持可重复性。
    
    Args:
        config (Tuple): 模拟配置元组。
            由 sk.export_state(population) 返回，包含适应度、死亡率等参数。
        shapes (Tuple[Tuple, Tuple, Tuple]): 状态数组形状元组。
            (ind_count_shape, sperm_storage_shape, female_occupancy_shape)
            用于将展平的粒子状态重新塑形为原始形状。
        seed (int, optional): 基础随机数种子。默认 42。
            用于生成可重复的随机模拟。
        config_modifier (Callable, optional): 配置修改函数。
            签名: modifier(theta, base_config) -> modified_config
            当提供时，转移函数会根据 theta 动态修改 config。
            可使用 make_fitness_modifier() 创建。默认 None（使用固定 config）。
    
    Returns:
        Callable[[NDArray, Any], NDArray]: 状态转移函数。
            签名: transition(prev_states, theta) -> next_states
            - prev_states: (N, state_dim) 当前粒子状态（完整状态展平）
            - theta: 参数向量，当 config_modifier 存在时用于修改配置
            - next_states: (N, state_dim) 下一时刻粒子状态
    
    Example:
        >>> state, config, _ = sk.export_state(population)
        >>> shapes = (state[0].shape, state[1].shape, state[2].shape)
        >>> # 基本用法（固定配置）
        >>> transition = make_transition_fn(config, shapes, seed=42)
        >>> next_states = transition(prev_states, None)
        >>> 
        >>> # 参数依赖用法
        >>> modifier = make_fitness_modifier(['drive_viability'], {'drive_viability': [1,2]})
        >>> transition = make_transition_fn(config, shapes, seed=42, config_modifier=modifier)
        >>> next_states = transition(prev_states, np.array([0.8]))  # 生存力=0.8
    
    Notes:
        - 每个粒子独立模拟，故可并行化（未来可考虑 numba.prange）
        - 使用完整状态元组确保 run_tick 获得所有必要信息
        - config_modifier 在每次调用时应用，允许参数依赖的转移
        - 使用 seed + counter 模式保证 Numba 兼容性和可重复性
    """
    # 解包状态形状
    ind_shape, sperm_shape, fem_shape = shapes
    
    # 计算各分量的展平长度
    ind_len = int(np.prod(ind_shape))
    sperm_len = int(np.prod(sperm_shape))
    fem_len = int(np.prod(fem_shape))
    
    # 计算各分量在展平向量中的起止位置
    ind_end = ind_len
    sperm_end = ind_end + sperm_len
    fem_end = sperm_end + fem_len
    # tick 是最后一个元素
    
    # 使用列表包装 call_counter 以便在闭包中修改
    call_counter = [0]
    
    def transition(prev_states: NDArray, theta: Any) -> NDArray:
        """将所有粒子前进一步（使用完整状态）"""
        N, dim = prev_states.shape
        next_states = np.zeros_like(prev_states)
        
        # 获取当前调用的 counter 值，然后递增
        current_call = call_counter[0]
        call_counter[0] += 1
        
        # 如果有 config_modifier 且 theta 不为 None，则动态修改配置
        if config_modifier is not None and theta is not None:
            current_config = config_modifier(theta, config)
        else:
            current_config = config
        
        # 对每个粒子独立模拟一步
        for i in range(N):
            # 计算该粒子的 counter 值
            particle_counter = current_call * 10000 + i
            
            # 从展平向量重建完整状态元组
            flat = prev_states[i]
            ind_count = flat[:ind_end].reshape(ind_shape).copy()
            sperm_storage = flat[ind_end:sperm_end].reshape(sperm_shape).copy()
            female_occ = flat[sperm_end:fem_end].reshape(fem_shape).copy()
            tick = int(flat[fem_end])  # 最后一个元素是 tick
            
            state_t = (ind_count, sperm_storage, female_occ, tick)
            
            # 调用纯函数执行一步模拟（使用可能已修改的配置）
            new_state = sk.run_tick(
                state_t, current_config, 
                seed=seed,
                counter=particle_counter
            )
            
            # 将新状态重新展平
            new_ind = new_state[sk.STATE_INDIVIDUAL_COUNT].ravel()
            new_sperm = new_state[sk.STATE_SPERM_STORAGE].ravel()
            new_fem = new_state[sk.STATE_FEMALE_OCC].ravel()
            new_tick = float(new_state[sk.STATE_TICK])
            
            next_states[i] = np.concatenate([new_ind, new_sperm, new_fem, [new_tick]])
        
        return next_states
    
    return transition


def make_obs_loglik_fn(
    sigma: float,
    obs_rule: NDArray,
    project_fn: Callable,
    shapes: Optional[Tuple[Tuple, Tuple, Tuple]] = None
) -> Callable[[NDArray, NDArray, Any], NDArray]:
    """创建粒子滤波的观测似然函数。
    
    该函数将隐状态（粒子）映射到观测空间，并计算观测似然。
    使用高斯观测模型：
    
        y ~ N(H(x), σ² I)
    
    其中 H 是观测函数（由 obs_rule 和 project_fn 定义），
    将种群状态映射为可观测量（如各基因型计数）。
    
    状态格式
    --------
    展平的粒子状态格式为：
        [ind_count.ravel(), sperm_storage.ravel(), female_occ.ravel(), tick]
    
    本函数仅使用 individual_count 部分计算观测似然。
    
    Args:
        sigma (float): 观测标准差。
            表示观测误差的大小。较小的 sigma 使得观测更"可信"，
            但如果太小可能导致粒子退化。
        obs_rule (NDArray): 观测规则矩阵。
            形状为 (n_groups, n_sexes, n_ages, n_genotypes) 或 (n_groups, n_sexes, n_genotypes)。
            定义如何将个体计数聚合为观测组。
        project_fn (Callable): 投影函数。
            签名: project_fn(ind_count, rule) -> observed_counts
            通常使用 samplers.observation.apply_rule。
        shapes (Tuple, optional): 状态数组形状元组。
            (ind_count_shape, sperm_storage_shape, female_occupancy_shape)
            用于从完整状态中正确提取 individual_count。
            必须提供以支持完整状态格式。
    
    Returns:
        Callable[[NDArray, NDArray, Any], NDArray]: 观测似然函数。
            签名: obs_loglik(x_obs, states, theta) -> log_likelihoods
            - x_obs: (obs_dim,) 实际观测
            - states: (N, state_dim) 粒子状态（完整状态展平）
            - theta: 参数（未使用）
            - log_likelihoods: (N,) 每个粒子的对数似然
    
    Example:
        >>> from samplers.observation import apply_rule
        >>> state, config, _ = sk.export_state(population)
        >>> shapes = (state[0].shape, state[1].shape, state[2].shape)
        >>> obs_loglik = make_obs_loglik_fn(sigma=5.0, obs_rule=obs_rule, 
        ...                                  project_fn=apply_rule, shapes=shapes)
        >>> logliks = obs_loglik(x_obs, states, None)
    
    Notes:
        - 观测模型假设独立高斯误差，可扩展为其他分布（如泊松）
        - obs_rule 的设计允许灵活的观测分组（按基因型、按表型等）
        - project_fn 应该是纯函数以便于性能优化
    """
    # 确定 individual_count 的形状和长度
    if shapes is not None:
        ind_shape = shapes[0]
        ind_len = int(np.prod(ind_shape))
    else:
        # 向后兼容：从 obs_rule 推断形状（假设整个状态就是 ind_count）
        if obs_rule.ndim == 4:
            _, n_sex, n_age, n_gen = obs_rule.shape
            ind_shape = (n_sex, n_age, n_gen)
        else:
            _, n_sex, n_gen = obs_rule.shape
            ind_shape = (n_sex, n_gen)
        ind_len = int(np.prod(ind_shape))
    
    def obs_loglik(x_obs: NDArray, states: NDArray, theta: Any) -> NDArray:
        """计算所有粒子的观测对数似然"""
        N = states.shape[0]
        log_liks = np.zeros(N)
        
        # 对每个粒子计算似然
        for i in range(N):
            # 从完整状态中提取 individual_count 部分
            ind_count = states[i, :ind_len].reshape(ind_shape)
            # 投影到观测空间
            predicted = project_fn(ind_count, obs_rule).ravel()
            # 计算高斯对数似然
            log_liks[i] = gaussian_obs_loglik(x_obs, predicted, sigma)
        
        return log_liks
    
    return obs_loglik


# =============================================================================
# PMCMC 主函数（纯函数版本）
# =============================================================================

def run_pmcmc(
    # --------------- 粒子滤波参数 ---------------
    observations: NDArray,
    n_particles: int,
    init_sampler: Callable[[int], NDArray],
    transition_fn: Callable[[NDArray, Any], NDArray],
    obs_loglik_fn: Callable[[NDArray, NDArray, Any], NDArray],
    # --------------- MCMC 参数 ---------------
    theta_init: NDArray,
    n_iter: int,
    step_sizes: NDArray,
    log_prior_fn: Callable[[NDArray], float],
    # --------------- 可选参数 ---------------
    bounds: Optional[List[Tuple[float, float]]] = None,
    theta_to_config_fn: Optional[Callable] = None,
    resample_threshold: float = 0.5,
    burnin: int = 0,
    thin: int = 1,
    adapt: bool = True,
    adapt_interval: int = 100,
    target_accept: float = 0.234,
    seed: Optional[int] = None,
    verbose: bool = True
) -> PMCMCResult:
    """运行 Particle Marginal Metropolis-Hastings (PMMH) 采样器。
    
    这是 PMCMC 的核心实现，采用纯函数形式以便于测试和组合。
    该算法将粒子滤波嵌入到 MCMC 框架中，用于对状态空间模型的
    静态参数进行贝叶斯推断。
    
    算法流程
    --------
    PMMH 算法的核心思想是使用粒子滤波来近似边际似然 p(y|θ)，
    然后在标准的 Metropolis-Hastings 框架中使用这个近似。
    
    具体步骤::
    
        1. 初始化:
           - 设置初始参数 θ₀
           - 运行粒子滤波得到 log p̂(y|θ₀)
           
        2. 对于 m = 1, 2, ..., M:
           a. 提议新参数: θ* ~ q(·|θ_{m-1}) = N(θ_{m-1}, Σ)
           b. 运行粒子滤波得到 log p̂(y|θ*)
           c. 计算接受概率:
              α = min(1, [p̂(y|θ*) × p(θ*)] / [p̂(y|θ) × p(θ)])
           d. 以概率 α 接受 θ*，否则保持 θ_{m-1}
           
        3. 返回参数链 {θ_m}
    
    理论保证
    --------
    Andrieu et al. (2010) 的关键结果：即使粒子滤波只提供了似然的
    有偏估计，PMMH 仍然以精确后验 p(θ|y) 为平稳分布。这是因为
    算法可以被解释为在扩展空间上的精确 MCMC。
    
    Args:
        observations (NDArray): 观测序列。
            形状为 (T, obs_dim) 或 (T,)，其中 T 是时间步数。
            每个时刻一个观测向量。
        n_particles (int): 粒子数量。
            更多粒子 → 更精确的似然估计，但计算成本更高。
            典型值: 100-1000。粒子数太少会导致接受率不稳定。
        init_sampler (Callable[[int], NDArray]): 初始状态采样器。
            签名: sampler(N) -> (N, state_dim) 数组。
            由 make_init_sampler() 创建。
        transition_fn (Callable): 状态转移函数。
            签名: transition(states, theta) -> next_states。
            由 make_transition_fn() 创建。
        obs_loglik_fn (Callable): 观测对数似然函数。
            签名: obs_loglik(obs, states, theta) -> log_liks。
            由 make_obs_loglik_fn() 创建。
        theta_init (NDArray): 初始参数值，形状为 (n_params,)。
            应该在先验的高概率区域内，否则可能启动失败。
        n_iter (int): 总迭代次数（包括 burnin）。
        step_sizes (NDArray): 随机游走步长，形状为 (n_params,)。
            每个参数的提议标准差。初始值影响收敛速度。
        log_prior_fn (Callable[[NDArray], float]): 对数先验函数。
            可使用 log_uniform_prior 或 log_normal_prior。
        bounds (List[Tuple[float, float]], optional): 参数边界。
            用于反射边界条件。默认 None（无边界）。
        theta_to_config_fn (Callable, optional): 参数到配置的映射。
            用于参数依赖的状态转移（未来扩展）。默认 None。
        resample_threshold (float, optional): 重采样阈值。
            当 ESS/N < threshold 时触发重采样。默认 0.5。
        burnin (int, optional): 热身（burn-in）迭代数。
            前 burnin 次迭代不保存，用于链达到平稳分布。默认 0。
        thin (int, optional): 稀疏化间隔。
            每 thin 次迭代保存一次样本，减少自相关。默认 1。
        adapt (bool, optional): 是否自适应调整步长。
            仅在 burnin 期间进行。默认 True。
        adapt_interval (int, optional): 自适应间隔。
            每隔 adapt_interval 次迭代调整一次步长。默认 100。
        target_accept (float, optional): 目标接受率。
            自适应算法的目标值。默认 0.234（高维最优值）。
        seed (int, optional): 随机数种子。用于可重复性。
        verbose (bool, optional): 是否打印进度信息。默认 True。
    
    Returns:
        PMCMCResult: 包含以下字段的命名元组:
            - theta_chain: (n_samples, n_params) 参数样本链
            - loglik_chain: (n_samples,) 对数似然链
            - logprior_chain: (n_samples,) 对数先验链
            - accepted: (n_samples,) 接受标记
            - acceptance_rate: 总体接受率
    
    Raises:
        ValueError: 如果初始参数导致无效的似然或先验。
    
    Example:
        基本使用::
        
            >>> from samplers.pmcmc import run_pmcmc, make_init_sampler
            >>> # 设置组件
            >>> init_sampler = make_init_sampler(initial_state)
            >>> transition_fn = make_transition_fn(config, shapes, rng)
            >>> obs_loglik_fn = make_obs_loglik_fn(1.0, obs_rule, apply_rule)
            >>> # 定义先验
            >>> bounds = [(0.0, 1.0), (0.0, 1.0)]
            >>> log_prior = lambda theta: log_uniform_prior(theta, bounds)
            >>> # 运行
            >>> result = run_pmcmc(
            ...     observations=obs_data,
            ...     n_particles=200,
            ...     init_sampler=init_sampler,
            ...     transition_fn=transition_fn,
            ...     obs_loglik_fn=obs_loglik_fn,
            ...     theta_init=np.array([0.5, 0.5]),
            ...     n_iter=5000,
            ...     step_sizes=np.array([0.05, 0.05]),
            ...     log_prior_fn=log_prior,
            ...     burnin=1000,
            ...     thin=5
            ... )
            >>> print(f"接受率: {result.acceptance_rate:.3f}")
            >>> posterior_mean = result.theta_chain.mean(axis=0)
    
    Notes:
        - 粒子数选择: 通常 N ≥ 100，对于高维状态空间可能需要更多
        - 步长调节: 如果接受率 < 0.1，减小步长；> 0.5，增大步长
        - 自适应仅在 burnin 期间进行，以保证链的理论性质
        - 该函数是纯函数，不修改任何输入参数
    
    References:
        .. [1] Andrieu, C., Doucet, A., & Holenstein, R. (2010).
               Particle Markov chain Monte Carlo methods. JRSSB.
        .. [2] Roberts, G. O., & Rosenthal, J. S. (2009).
               Examples of adaptive MCMC. JCGS.
    """
    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    n_params = len(theta_init)
    
    # 计算保存的样本数: (总迭代 - burnin) / thin
    n_saved = max(1, (n_iter - burnin) // thin)
    
    # 预分配存储数组
    theta_chain = np.zeros((n_saved, n_params))    # 参数链
    loglik_chain = np.zeros(n_saved)                # 对数似然链
    logprior_chain = np.zeros(n_saved)              # 对数先验链
    accepted_arr = np.zeros(n_saved, dtype=bool)    # 接受标记
    
    # 复制可变输入，避免修改原始数据
    theta = theta_init.copy()
    step_sizes = step_sizes.copy()
    
    # -------------------------------------------------------------------------
    # 运行初始粒子滤波
    # -------------------------------------------------------------------------
    if verbose:
        print(f"PMCMC: 初始化, n_iter={n_iter}, n_particles={n_particles}")
    
    # 使用粒子滤波估计初始参数的边际似然
    _, _, loglik = particle_filter(
        observations, n_particles, init_sampler, transition_fn, obs_loglik_fn,
        theta, resample_threshold=resample_threshold
    )
    
    # 计算初始先验
    logprior = log_prior_fn(theta)
    
    # 检查初始状态有效性
    if not np.isfinite(loglik) or not np.isfinite(logprior):
        raise ValueError(f"初始参数无效: loglik={loglik}, logprior={logprior}")
    
    # -------------------------------------------------------------------------
    # MCMC 主循环
    # -------------------------------------------------------------------------
    n_accept = 0           # 总接受次数
    n_accept_window = 0    # 当前窗口内接受次数（用于自适应）
    save_idx = 0           # 保存索引
    
    for m in range(n_iter):
        # ----- 步骤 1: 生成提议 -----
        # 使用随机游走 Metropolis 提议: θ* ~ N(θ, diag(step_sizes²))
        theta_prop = random_walk_proposal(theta, step_sizes, rng, bounds)
        
        # 计算提议的先验
        logprior_prop = log_prior_fn(theta_prop)
        
        # ----- 步骤 2: 评估提议 -----
        # 只有当先验有效时才运行（昂贵的）粒子滤波
        if np.isfinite(logprior_prop):
            # 运行粒子滤波估计边际似然 log p̂(y|θ*)
            _, _, loglik_prop = particle_filter(
                observations, n_particles, init_sampler, transition_fn, obs_loglik_fn,
                theta_prop, resample_threshold=resample_threshold
            )
        else:
            # 先验外的提议直接拒绝，无需运行粒子滤波
            loglik_prop = -np.inf
        
        # ----- 步骤 3: Metropolis-Hastings 接受/拒绝 -----
        # 生成均匀随机数用于接受判断
        log_u = np.log(rng.uniform())
        
        # 调用接受函数（可 Numba 加速）
        accept = mh_accept(loglik_prop, logprior_prop, loglik, logprior, log_u)
        
        # 如果接受，更新当前状态
        if accept:
            theta = theta_prop
            loglik = loglik_prop
            logprior = logprior_prop
            n_accept += 1
            n_accept_window += 1
        
        # ----- 步骤 4: 保存样本 -----
        # 只在 burnin 之后、按 thin 间隔保存
        if m >= burnin and (m - burnin) % thin == 0 and save_idx < n_saved:
            theta_chain[save_idx] = theta
            loglik_chain[save_idx] = loglik
            logprior_chain[save_idx] = logprior
            accepted_arr[save_idx] = accept
            save_idx += 1
        
        # ----- 步骤 5: 自适应步长调整（仅 burnin 期间） -----
        # 使用 Robbins-Monro 类型的自适应算法（Roberts & Rosenthal, 2009）
        # 目标：调整步长使接受率接近 target_accept
        if adapt and m < burnin and (m + 1) % adapt_interval == 0:
            # 计算当前窗口的接受率
            rate = n_accept_window / adapt_interval
            
            # 自适应衰减因子: γ_n → 0 以保证收敛
            # 这确保后期调整越来越小
            gamma = 1.0 / np.sqrt(1 + m / adapt_interval)
            
            # 对数尺度调整步长
            # 如果 rate > target: 增大步长; 如果 rate < target: 减小步长
            step_sizes *= np.exp(gamma * (rate - target_accept))
            
            # 限制步长在合理范围内，避免过大或过小
            # 参考参数的先验范围来设置上下界
            if bounds is not None:
                # 使用参数范围的 0.1% 到 20% 作为步长上下界
                # 更保守的上界避免步长过大导致MCMC"乱走"
                param_ranges = np.array([b[1] - b[0] for b in bounds])
                min_step = 0.001 * param_ranges   # 最小步长为范围的 0.1%
                max_step = 0.2 * param_ranges     # 最大步长为范围的 20%
                step_sizes = np.clip(step_sizes, min_step, max_step)
            else:
                # 没有 bounds 时使用绝对限制
                step_sizes = np.clip(step_sizes, 1e-6, 10.0)
            
            # 重置窗口计数器
            n_accept_window = 0
            
            if verbose:
                print(f"  [iter {m+1}] adapt: accept_rate={rate:.3f}, step_sizes={step_sizes}")
        
        # ----- 进度报告 -----
        if verbose and (m + 1) % 200 == 0:
            print(f"  [iter {m+1}/{n_iter}] accept={n_accept/(m+1):.3f}, loglik={loglik:.2f}")
    
    # -------------------------------------------------------------------------
    # 返回结果
    # -------------------------------------------------------------------------
    rate = n_accept / n_iter
    if verbose:
        print(f"PMCMC 完成: acceptance_rate={rate:.3f}, n_samples={save_idx}")
    
    # 返回命名元组结果
    return PMCMCResult(
        theta_chain=theta_chain[:save_idx],      # 截断到实际保存数量
        loglik_chain=loglik_chain[:save_idx],
        logprior_chain=logprior_chain[:save_idx],
        accepted=accepted_arr[:save_idx],
        acceptance_rate=rate
    )


# =============================================================================
# PMCMC 类（便利封装）
# =============================================================================

class PMCMC:
    """PMMH 采样器的面向对象封装。
    
    该类提供了 PMMH 算法的便捷接口，内部调用 run_pmcmc 纯函数。
    适合用于快速原型开发，无需手动构建各个组件函数。
    
    类封装了以下组件的创建:
    - 初始状态采样器 (init_sampler)
    - 状态转移函数 (transition_fn)
    - 观测似然函数 (obs_loglik_fn)
    
    主要优点:
    - 简化 API：只需传入种群状态和观测，自动构建内部组件
    - 状态管理：保存配置，支持多次调用 run()
    - 与 Population 类集成：可使用 create_pmcmc_from_population() 快速创建
    - 参数估计：通过 config_modifier 支持估计影响状态转移的参数
    
    Attributes:
        config (Tuple): 模拟配置元组。
        observations (NDArray): 观测序列。
        n_particles (int): 粒子数量。
        resample_threshold (float): 重采样阈值。
        verbose (bool): 是否打印进度。
        seed (int): 随机数种子。用于 seed + counter 模式生成随机数，保证 Numba 兼容性。
        init_sampler (Callable): 初始状态采样器。
        transition_fn (Callable): 状态转移函数。
        obs_loglik_fn (Callable): 观测似然函数。
    
    Example:
        基本使用（固定参数）::
        
            >>> from utils import simulation_kernels as sk
            >>> state, config, _ = sk.export_state(population)
            >>> # 构建观测规则
            >>> obs_rule = ...  # 由 ObservationFilter.build_filter() 生成
            >>> # 创建采样器
            >>> sampler = PMCMC(
            ...     initial_state=state,
            ...     config=config,
            ...     observations=obs_data,
            ...     observation_rule=obs_rule,
            ...     n_particles=200,
            ...     obs_sigma=5.0
            ... )
            >>> # 定义先验并运行
            >>> bounds = [(0.0, 1.0), (0.0, 1.0)]
            >>> log_prior = lambda theta: log_uniform_prior(theta, bounds)
            >>> result = sampler.run(
            ...     theta_init=np.array([0.5, 0.5]),
            ...     n_iter=5000,
            ...     step_sizes=np.array([0.05, 0.05]),
            ...     log_prior_fn=log_prior,
            ...     burnin=1000
            ... )
        
        估计适应度参数::
        
            >>> # 创建参数修改器：估计转基因的生存力
            >>> modifier = make_fitness_modifier(
            ...     param_names=['drive_viability'],
            ...     genotype_targets={'drive_viability': ["D|w", "D|D"]},
            ...     registry=pop.registry,
            ...     species=pop.species
            ... )
            >>> sampler = PMCMC(
            ...     initial_state=state,
            ...     config=config,
            ...     observations=obs_data,
            ...     observation_rule=obs_rule,
            ...     n_particles=200,
            ...     obs_sigma=5.0,
            ...     config_modifier=modifier  # 关键：传入修改器
            ... )
            >>> # 估计生存力参数
            >>> bounds = [(0.5, 1.0)]  # 生存力在 50%-100% 之间
            >>> log_prior = lambda theta: log_uniform_prior(theta, bounds)
            >>> result = sampler.run(
            ...     theta_init=np.array([0.9]),    # 初始猜测 90%
            ...     n_iter=2000,
            ...     step_sizes=np.array([0.05]),
            ...     log_prior_fn=log_prior,
            ...     bounds=bounds,
            ...     burnin=500
            ... )
            >>> print(f"后验均值: {result.theta_chain.mean():.3f}")
        
        使用便利函数::
        
            >>> sampler = create_pmcmc_from_population(pop, observations)
            >>> result = sampler.run(...)
    
    Notes:
        - 内部调用 run_pmcmc 纯函数，保持实现的模块化
        - 每次调用 run() 都是独立的采样过程
        - 使用 config_modifier 可估计影响状态转移的参数（生存力、繁殖力等）
        - 如需高度定制，建议直接使用 run_pmcmc 纯函数
    """
    
    def __init__(
        self,
        initial_state: Tuple,
        config: Tuple,
        observations: NDArray,
        observation_rule: NDArray,
        n_particles: int = 100,
        obs_sigma: float = 1.0,
        resample_threshold: float = 0.5,
        seed: Optional[int] = None,
        verbose: bool = True,
        config_modifier: Optional[Callable[[NDArray, Tuple], Tuple]] = None
    ):
        """初始化 PMCMC 采样器。
        
        Args:
            initial_state (Tuple): 初始种群状态元组。
                格式: (individual_count, sperm_storage, female_occupancy, tick)
                通常由 sk.export_state(population) 获得。
            config (Tuple): 模拟配置元组。
                包含适应度、死亡率等参数。由 sk.export_state() 返回。
            observations (NDArray): 观测序列。
                形状为 (T, obs_dim)，T 是时间步数。
            observation_rule (NDArray): 观测规则矩阵。
                定义如何将个体计数投影到观测空间。
                由 ObservationFilter.build_filter() 生成。
            n_particles (int, optional): 粒子数量。默认 100。
                更多粒子 = 更精确的似然估计。
            obs_sigma (float, optional): 观测标准差。默认 1.0。
                较小的 sigma 使观测更"可信"。
            resample_threshold (float, optional): 重采样阈值。默认 0.5。
                当 ESS/N < threshold 时触发重采样。
            seed (int, optional): 随机数种子。用于可重复性。
            verbose (bool, optional): 是否打印进度。默认 True。
            config_modifier (Callable, optional): 配置修改函数。
                签名: modifier(theta, base_config) -> modified_config
                当提供时，转移函数会根据 theta 动态修改 config。
                可使用 make_fitness_modifier() 创建。默认 None。
        """
        # 保存配置
        self.config = config
        self.observations = observations
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.verbose = verbose
        
        # 保存种子用于随机数生成
        # 使用 seed + counter 模式保证 Numba 兼容性
        self.seed = seed if seed is not None else 42
        
        # 形状信息
        ind = initial_state[sk.STATE_INDIVIDUAL_COUNT]
        sperm = initial_state[sk.STATE_SPERM_STORAGE]
        fem = initial_state[sk.STATE_FEMALE_OCC]
        shapes = (ind.shape, sperm.shape, fem.shape)
        
        # 构建函数
        # 使用 seed 参数而非 rng，保证 Numba 兼容性
        # 使用完整状态（ind_count + sperm_storage + female_occ + tick）
        from samplers.observation import apply_rule
        self.init_sampler = make_init_sampler(initial_state, seed=self.seed)
        self.transition_fn = make_transition_fn(
            config, shapes, seed=self.seed + 1000, 
            config_modifier=config_modifier
        )
        self.obs_loglik_fn = make_obs_loglik_fn(obs_sigma, observation_rule, apply_rule, shapes=shapes)
    
    def run(
        self,
        theta_init: NDArray,
        n_iter: int,
        step_sizes: NDArray,
        log_prior_fn: Callable[[NDArray], float],
        bounds: Optional[List[Tuple[float, float]]] = None,
        burnin: int = 0,
        thin: int = 1,
        adapt: bool = True,
        **kwargs
    ) -> PMCMCResult:
        """运行 PMMH 采样。
        
        这是采样器的主入口点。内部调用 run_pmcmc 纯函数执行实际采样。
        每次调用都是独立的采样过程。
        
        Args:
            theta_init (NDArray): 初始参数值，形状为 (n_params,)。
                应该在先验的高概率区域内。
            n_iter (int): 总迭代次数（包括 burnin）。
            step_sizes (NDArray): 随机游走步长，形状为 (n_params,)。
                每个参数的提议标准差。
            log_prior_fn (Callable): 对数先验函数。
                签名: log_prior(theta) -> float。
                可使用 log_uniform_prior 或 log_normal_prior。
            bounds (List[Tuple[float, float]], optional): 参数边界。
                用于反射边界条件。默认 None。
            burnin (int, optional): 热身迭代数。默认 0。
            thin (int, optional): 稀疏化间隔。默认 1。
            adapt (bool, optional): 是否自适应步长。默认 True。
            **kwargs: 传递给 run_pmcmc 的其他参数。
        
        Returns:
            PMCMCResult: 采样结果，包含参数链、似然链等。
        
        Example:
            >>> bounds = [(0.0, 1.0), (0.0, 1.0)]
            >>> log_prior = lambda theta: log_uniform_prior(theta, bounds)
            >>> result = sampler.run(
            ...     theta_init=np.array([0.5, 0.5]),
            ...     n_iter=5000,
            ...     step_sizes=np.array([0.05, 0.05]),
            ...     log_prior_fn=log_prior,
            ...     burnin=1000
            ... )
            >>> print(f"后验均值: {result.theta_chain.mean(axis=0)}")
        """
        return run_pmcmc(
            observations=self.observations,
            n_particles=self.n_particles,
            init_sampler=self.init_sampler,
            transition_fn=self.transition_fn,
            obs_loglik_fn=self.obs_loglik_fn,
            theta_init=theta_init,
            n_iter=n_iter,
            step_sizes=step_sizes,
            log_prior_fn=log_prior_fn,
            bounds=bounds,
            resample_threshold=self.resample_threshold,
            burnin=burnin,
            thin=thin,
            adapt=adapt,
            verbose=self.verbose,
            **kwargs
        )


# =============================================================================
# 便利函数
# =============================================================================

def create_pmcmc_from_population(
    pop: 'AgeStructuredPopulation',
    observations: NDArray,
    observation_groups: Optional[Any] = None,
    **kwargs
) -> PMCMC:
    """从 AgeStructuredPopulation 对象快速创建 PMCMC 采样器。
    
    这是一个便利函数，自动从种群对象导出状态和配置，
    并构建观测规则。适合快速原型开发。
    
    Args:
        pop (AgeStructuredPopulation): 种群对象。
            将从中导出初始状态和配置。
        observations (NDArray): 观测序列。
            形状为 (T, obs_dim)。
        observation_groups (Any, optional): 观测分组定义。
            传递给 ObservationFilter.build_filter()。
            如果为 None，使用默认分组（按基因型）。
        **kwargs: 传递给 PMCMC 构造函数的其他参数。
            如 n_particles, obs_sigma 等。
    
    Returns:
        PMCMC: 配置好的采样器实例。
    
    Example:
        >>> from utils.nonWF_population import AgeStructuredPopulation
        >>> pop = AgeStructuredPopulation(...)
        >>> # 创建采样器
        >>> sampler = create_pmcmc_from_population(
        ...     pop, 
        ...     observations=obs_data,
        ...     n_particles=200,
        ...     obs_sigma=5.0
        ... )
        >>> result = sampler.run(theta_init, n_iter=5000, ...)
    
    Notes:
        - 该函数内部调用 sk.export_state() 和 ObservationFilter
        - 对于高度定制的场景，建议直接使用 PMCMC 构造函数
    """
    from samplers.observation import ObservationFilter
    
    # 从种群导出状态和配置
    state, config, _ = sk.export_state(pop)
    
    # 构建观测规则
    obs_filter = ObservationFilter(pop.registry)
    obs_rule, _ = obs_filter.build_filter(
        pop, diploid_genotypes=pop.registry.index_to_genotype, groups=observation_groups
    )
    
    return PMCMC(state, config, observations, obs_rule, **kwargs)


def make_fitness_modifier(
    param_names: List[str],
    genotype_targets: dict,
    registry: 'IndexCore',
    species: Optional['Species'] = None
) -> Callable[[NDArray, Tuple], Tuple]:
    """创建适应度参数修改器。
    
    返回一个函数，将参数向量映射到修改后的配置元组。
    用于在 PMCMC 中根据当前参数值动态更新适应度设置。
    
    该函数支持修改以下适应度相关参数:
    - viability: 生存力（存活概率）
    - female_fecundity: 雌性繁殖力
    - male_fecundity: 雄性繁殖力
    - sexual_selection: 性选择强度
    
    基因型指定方式
    --------------
    genotype_targets 的值可以是以下类型:
    - Genotype 对象列表: 直接使用 registry.genotype_to_index 查询索引
    - 字符串列表: 需要提供 species 参数，使用 species.get_genotype_from_str 转换
    - 整数列表: 直接作为索引使用（向后兼容，但不推荐）
    
    Args:
        param_names (List[str]): 参数名称列表。
            每个名称应包含关键词以指示参数类型:
            - 'viability': 生存力参数（同时影响雌雄）
            - 'female_viability': 仅影响雌性生存力
            - 'male_viability': 仅影响雄性生存力
            - 'female_fecundity': 雌性繁殖力
            - 'male_fecundity' 或 'fecundity': 雄性/一般繁殖力
            - 'sexual_selection': 性选择
        genotype_targets (dict): 参数到目标基因型的映射。
            格式: {param_name: [genotypes]}
            genotypes 可以是 Genotype 对象、字符串或整数索引。
        registry (IndexCore): 索引注册表。
            用于将 Genotype 对象映射到整数索引。
            通常通过 population.registry 获取。
        species (Species, optional): 物种对象。
            当 genotype_targets 中包含字符串时必须提供。
            用于将字符串解析为 Genotype 对象。
    
    Returns:
        Callable[[NDArray, Tuple], Tuple]: 配置修改函数。
            签名: modifier(theta, base_config) -> modified_config
    
    Example:
        使用 Genotype 对象::
        
            >>> wt = species.get_genotype_from_str("w|w")
            >>> drive = species.get_genotype_from_str("D|w")
            >>> modifier = make_fitness_modifier(
            ...     param_names=['wt_viability', 'drive_viability'],
            ...     genotype_targets={
            ...         'wt_viability': [wt],
            ...         'drive_viability': [drive]
            ...     },
            ...     registry=pop.registry
            ... )
        
        使用基因型字符串（需要 species）::
        
            >>> modifier = make_fitness_modifier(
            ...     param_names=['wt_viability', 'drive_viability'],
            ...     genotype_targets={
            ...         'wt_viability': ["w|w"],
            ...         'drive_viability': ["D|w", "D|D"]
            ...     },
            ...     registry=pop.registry,
            ...     species=pop.species
            ... )
            >>> # 使用修改器
            >>> theta = np.array([1.0, 0.7])  # 野生型 100%，drive 70%
            >>> new_config = modifier(theta, base_config)
    
    Raises:
        ValueError: 如果 genotype_targets 包含字符串但未提供 species。
        KeyError: 如果基因型未在 registry 中注册。
    
    Notes:
        - 修改器返回配置的副本，不修改原始配置
        - 参数名中的关键词检测不区分大小写
        - 预先解析所有基因型索引，运行时只做数组修改，保证高效
    """
    # 预处理：将所有基因型转换为整数索引
    resolved_indices: Dict[str, List[int]] = {}
    
    for param_name in param_names:
        if param_name not in genotype_targets:
            raise ValueError(f"参数 '{param_name}' 未在 genotype_targets 中定义")
        
        targets = genotype_targets[param_name]
        indices = []
        
        for target in targets:
            if isinstance(target, int):
                # 直接使用索引（向后兼容）
                indices.append(target)
            elif isinstance(target, str):
                # 字符串形式，需要 species 解析
                if species is None:
                    raise ValueError(
                        f"genotype_targets 包含字符串 '{target}'，但未提供 species 参数。"
                        f"请传入 species=pop.species 以支持字符串解析。"
                    )
                genotype = species.get_genotype_from_str(target)
                idx = registry.genotype_to_index[genotype]
                indices.append(idx)
            else:
                # 假设是 Genotype 对象
                idx = registry.genotype_to_index[target]
                indices.append(idx)
        
        resolved_indices[param_name] = indices
    
    def modifier(theta: NDArray, base_config: Tuple) -> Tuple:
        """将参数向量应用到配置中"""
        # 创建配置列表的可变副本
        cfg = list(base_config)
        
        # 复制各适应度数组（避免修改原始数据）
        f_viab = cfg[sk.CFG_FEMALE_VIABILITY].copy()
        m_viab = cfg[sk.CFG_MALE_VIABILITY].copy()
        f_fec = cfg[sk.CFG_FEMALE_FECUNDITY].copy()
        m_fec = cfg[sk.CFG_MALE_FECUNDITY].copy()
        sex_sel = cfg[sk.CFG_SEXUAL_SELECTION].copy()
        
        # 遍历参数，根据名称更新相应的适应度值
        for i, name in enumerate(param_names):
            val = theta[i]
            indices = resolved_indices[name]
            name_lower = name.lower()
            
            # 根据参数名称关键词更新相应数组
            for idx in indices:
                # 生存力参数
                if 'viability' in name_lower:
                    if 'female_viability' in name_lower:
                        f_viab[idx] = val
                    elif 'male_viability' in name_lower:
                        m_viab[idx] = val
                    else:
                        # 通用 viability: 同时设置雌雄
                        f_viab[idx] = m_viab[idx] = val
                
                # 繁殖力参数
                if 'female_fecundity' in name_lower:
                    f_fec[idx] = val
                if 'male_fecundity' in name_lower or \
                   ('fecundity' in name_lower and 'female' not in name_lower):
                    m_fec[idx] = val
                
                # 性选择参数
                if 'sexual_selection' in name_lower:
                    sex_sel[idx] = val
        
        # 更新配置
        cfg[sk.CFG_FEMALE_VIABILITY] = f_viab
        cfg[sk.CFG_MALE_VIABILITY] = m_viab
        cfg[sk.CFG_FEMALE_FECUNDITY] = f_fec
        cfg[sk.CFG_MALE_FECUNDITY] = m_fec
        cfg[sk.CFG_SEXUAL_SELECTION] = sex_sel
        
        return tuple(cfg)
    
    return modifier
