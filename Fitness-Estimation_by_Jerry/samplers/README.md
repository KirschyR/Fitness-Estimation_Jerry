# samplers

此目录包含与观测/采样相关的轻量工具，主要文件：

- `samplers/observation.py` — `ObservationFilter` 与纯 NumPy 的 `apply_rule`。
- `samplers/helpers.py` — 粒子滤波器 `particle_filter` 和重采样函数。
- `samplers/pmcmc.py` — **Particle Marginal Metropolis-Hastings (PMMH)** 采样器。
- `test_observation.py` — 单元测试，位于仓库根目录。
- `test_pmcmc.py` — PMCMC 采样器单元测试。

## PMCMC 采样器

`samplers/pmcmc.py` 实现了 Particle Marginal Metropolis-Hastings (PMMH) 算法，用于对遗传模型的参数进行贝叶斯推断。

### 核心算法

PMMH 使用粒子滤波器作为 MH 接受率中边缘似然 p(y_{1:T}|θ) 的无偏估计器：

1. 给定当前参数 θ，运行粒子滤波得到边缘似然估计 p̂(y|θ)
2. 从提议分布 q(θ'|θ) 采样新参数 θ'
3. 运行粒子滤波得到 p̂(y|θ')
4. 以概率 min(1, A) 接受 θ'，其中
   A = [p̂(y|θ') × p(θ') × q(θ|θ')] / [p̂(y|θ) × p(θ) × q(θ'|θ)]

### 快速示例

```python
from utils.nonWF_population import AgeStructuredPopulation
from utils.simulation_kernels import export_state
from samplers.pmcmc import (
    PMCMC, 
    create_pmcmc_from_population,
    log_uniform_prior,
    make_fitness_config_modifier
)
import numpy as np

# 1. 创建种群模型
pop = AgeStructuredPopulation(species, initial_population_distribution, ...)

# 2. 方法一：使用便利函数创建采样器
sampler = create_pmcmc_from_population(
    pop, 
    observations=y_obs,  # shape (T, n_obs)
    observation_groups={'WT': {'genotype': [0]}, 'Mutant': {'genotype': [1,2,3]}},
    n_particles=100,
    obs_sigma=10.0
)

# 3. 方法二：手动创建（更多控制）
from samplers.observation import ObservationFilter, apply_rule

state, config, _ = export_state(pop)
obs_filter = ObservationFilter(pop.registry)
obs_rule, labels = obs_filter.build_filter(pop, diploid_genotypes=pop)

sampler = PMCMC(
    initial_state=state,
    config=config,
    observations=y_obs,
    observation_rule=obs_rule,
    n_particles=100,
    obs_sigma=10.0
)

# 4. 定义先验和参数映射
bounds = [(0.0, 2.0), (0.0, 2.0)]
def log_prior_fn(theta):
    return log_uniform_prior(theta, bounds)

# 可选：将参数映射到适应度
modifier = make_fitness_config_modifier(
    ['viability_drive', 'fecundity_drive'],
    genotype_indices={'viability_drive': [1, 2], 'fecundity_drive': [1, 2]}
)

# 5. 运行采样
result = sampler.run(
    theta_init=np.array([1.0, 1.0]),
    n_iter=1000,
    step_sizes=np.array([0.1, 0.1]),
    log_prior_fn=log_prior_fn,
    theta_to_config_fn=modifier,  # 可选
    bounds=bounds,
    burnin=200,
    thin=2
)

# 6. 分析结果
print(f"接受率: {result.acceptance_rate:.3f}")
print(f"后验均值: {result.theta_chain.mean(axis=0)}")
print(f"后验标准差: {result.theta_chain.std(axis=0)}")
```

### 设计特点

- **Numba 友好**: 使用纯函数 `simulation_kernels` 进行状态推进
- **复用初始化**: 只初始化一次 `AgeStructuredPopulation`，导出状态后使用纯函数运行
- **观测模型**: 支持小的正态观测误差
- **自适应步长**: 在 burnin 阶段使用 Robbins-Monro 方案自动调整步长

### 参考文献

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). 
  Particle Markov chain Monte Carlo methods. 
  Journal of the Royal Statistical Society: Series B, 72(3), 269-342.

---

## 观测模型 (observation.py)

目标

- 从用户友好的分组规范（按 age/sex/genotype/unordered）生成 NumPy mask。
- 将 mask 应用于 `PopulationState.individual_count`，得到按组聚合的计数。
- 保持简单、可测试，且易于与 Numba 加速函数配合使用。

快速入门示例

1. 基本（非年龄/年龄折叠/年龄结构均支持）

```python
from utils.index_core import IndexCore
from utils.population_state import PopulationState
from samplers.observation import ObservationFilter, apply_rule

ic = IndexCore()
obs = ObservationFilter(ic)

# 假设已有 `pop`（`BasePopulation` 的实例）和 diploid_genotypes（序列或 Species 或 BasePopulation）
mask, labels = obs.build_filter(
  pop,
  diploid_genotypes=diploid_genotypes,
  groups=[{"age": [[2,4]], "genotype": ["A|a"], "sex": "M", "unordered": True}],
  collapse_age=False,
)
observed = apply_rule(pop.state.individual_count, mask)
```

2. 与 `AgeStructuredPopulation` 一起使用

- `build_filter` 的 `diploid_genotypes` 可以直接传入 `pop`（`BasePopulation` 的实例）或 `pop.species`，函数会尽量从中提取二倍型列表。

```python
from utils.nonWF_population import AgeStructuredPopulation
from samplers.observation import ObservationFilter, apply_rule

pop = AgeStructuredPopulation(...)   # 已构建且初始化完毕
obs = ObservationFilter(pop.registry)
mask, labels = obs.build_filter(pop, diploid_genotypes=pop, groups=...)
observed = apply_rule(pop.state.individual_count, mask)
```

Numba 加速提示

- `ObservationFilter.build_filter` 返回的是纯数值 ndarray（mask），可直接传入 Numba 编译的函数。示例函数（位于本仓库注释中）有：
  - `apply_mask_age(mask, arr)` — 对 age-structured 的 `mask`(4D) 与 `arr`(3D)
  - `apply_mask_collapsed(mask, arr)` — mask 已折叠年龄
  - `apply_mask_nonage(mask, arr)` — 非年龄结构

测试

- 单元测试位于根目录 `test_observation.py`。运行：

```bash
pytest -q test_observation.py
```

实现与兼容性说明

- `ObservationFilter` 构造时接收一个 `IndexCore`（通常从 `pop.registry` 或 `pop._index_core` 获取）。
- `build_filter` 的 `diploid_genotypes` 参数可接受：
  - 序列（Genotype 对象或字符串），
  - `Species` 对象（会调用 `iter_genotypes()`），
  - `BasePopulation` 实例（优先调用 `_get_all_possible_diploid_genotypes()`，回退到 `pop.species.iter_genotypes()`）。
- 若只使用整数索引选择器，则可以省略 `diploid_genotypes`（但 `groups is None` 时仍需要以便枚举）。

进阶建议

- 若需自动分发到合适的 Numba 函数，可在调用端写个薄包装函数：根据 `state` 与 `mask` 的 ndim 选择对应实现。
- 若需要更复杂的分组语法或权重支持（非 0/1 掩码），可扩展 `build_filter` 以接受自定义权重。

联系方式与贡献

- 若要我把 Numba 函数或自动分发包装加入到 `samplers/` 中或把 README 中的示例转换为可执行示例脚本，请告诉我。