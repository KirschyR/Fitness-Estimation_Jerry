# PopGen-Sim（暂定名称）

**高性能遗传学模拟框架**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.21+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/numba-0.55+-orange.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

一个用于模拟种群遗传学动态的 Python 框架，支持年龄结构、精子存储、配子/合子修饰器以及 Numba 加速。

---

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
  - [遗传结构 (Genetic Structures)](#遗传结构-genetic-structures)
  - [遗传实体 (Genetic Entities)](#遗传实体-genetic-entities)
  - [种群模型 (Population Models)](#种群模型-population-models)
- [API 参考](#api-参考)
  - [Species](#species)
  - [Chromosome](#chromosome)
  - [Locus](#locus)
  - [AgeStructuredPopulation](#agestructuredpopulation)
  - [Hooks 系统](#hooks-系统)
  - [Modifiers](#modifiers)
- [高级用法](#高级用法)
  - [自定义配子修饰器](#自定义配子修饰器)
  - [自定义合子修饰器](#自定义合子修饰器)
  - [纯函数模拟核心](#纯函数模拟核心)
- [性能优化](#性能优化)
- [示例](#示例)
- [API 完整参考](#api-完整参考)
- [变更日志](#变更日志)

---

## 特性

- 🧬 **灵活的遗传架构**: 支持多染色体、多位点、多等位基因的遗传结构定义
- 👥 **年龄结构种群**: 支持世代重叠的年龄结构模型 (Non-Wright-Fisher)
- 🔄 **精子存储机制**: 模拟昆虫等物种的精子存储与置换
- ⚡ **Numba 加速**: 核心算法支持 JIT 编译，显著提升性能
- 🪝 **Hook 系统**: 灵活的事件钩子机制，支持在模拟生命周期各阶段注入自定义逻辑
- 🔧 **修饰器系统**: 支持配子和合子级别的遗传修饰（基因驱动、不育等）
- 📊 **纯函数核心**: 支持导出状态进行批量/并行 Monte Carlo 模拟

---

## 安装

### 依赖项

- Python >= 3.9
- NumPy >= 1.21
- Numba >= 0.55 (可选，用于加速)

---

## 快速开始

### 最小示例

```python
from utils.genetic_structures import Species, Chromosome, Locus
from utils.genetic_entities import Gene, Genotype
from utils.nonWF_population import AgeStructuredPopulation

# 1. 定义物种的遗传架构
species = Species("Drosophila")

# 添加常染色体
chr2 = species.add("Chr2")
locus_A = chr2.add("A")  # 位点 A
locus_A.add_alleles(["w", "D"])  # 野生型 w 和驱动型 D

# 2. 获取所有可能的基因型
genotypes = species.get_all_genotypes()
# 输出: [Genotype(w|w), Genotype(w|D), Genotype(D|w), Genotype(D|D)]

# 3. 定义初始种群
initial_pop = {
    "female": {
        "w|w": [0, 0, 100, 80, 60, 40, 20, 0],  # 各年龄的个体数
    },
    "male": {
        "w|w": [0, 0, 50, 40, 0, 0, 0, 0],
    }
}

# 4. 创建种群实例
pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=initial_pop,
    n_ages=8,
    offspring_per_female=10.0,
)

# 5. 运行模拟
pop.run(n_steps=100, record_every=10)

# 6. 查看结果
print(f"最终种群大小: {pop.get_total_count()}")
print(f"历史记录数: {len(pop.history)}")
```

---

## 核心概念

PopGen-Sim 框架基于清晰的分层设计：

```
Species (物种)
├── Chromosome (染色体)
│   ├── Locus (位点)
│   │   └── Gene (等位基因/实体)
│   └── Haplotype (单倍型/实体)
└── Genotype (基因型/实体)
```

### 遗传结构 (Genetic Structures)

**结构 (Structures)** 是静态的、模型级别的遗传蓝图，定义了遗传架构的拓扑结构。

| 类 | 描述 | 子结构 |
|---|---|---|
| `Species` | 物种定义，遗传架构的根 | Chromosome |
| `Chromosome` | 染色体（连锁群） | Locus |
| `Locus` | 基因位点 | - |

**结构特性:**
- 结构在同一 `Species` 内唯一（按名称缓存）
- 不可变设计，创建后不应修改
- 支持链式 API 快速定义

```python
# 链式定义遗传架构
species = Species("Example")
species.add("ChrX", sex_type="X").add("white").add_alleles(["w+", "w"])
species.add("Chr2").add("yellow").add_alleles(["y+", "y"])
```

### 遗传实体 (Genetic Entities)

**实体 (Entities)** 是具体的遗传对象，代表个体携带的实际遗传物质。

| 类 | 描述 | 绑定到 |
|---|---|---|
| `Gene` | 等位基因实例 | Locus |
| `Haplotype` | 单倍型（染色体上的等位基因组合） | Chromosome |
| `HaploidGenome` | 完整单倍体基因组 | Species |
| `Genotype` | 二倍体基因型（两个单倍体基因组） | Species |

```python
from utils.genetic_entities import Gene, Genotype

# 获取特定基因
w_plus = species.get_gene("w+")
w = species.get_gene("w")

# 从字符串解析基因型
genotype = species.get_genotype_from_str("w+|w")
print(genotype.maternal)  # HaploidGenome with w+
print(genotype.paternal)  # HaploidGenome with w
```

### 种群模型 (Population Models)

框架提供了基于 `BasePopulation` 的种群模型抽象，当前主要实现是 `AgeStructuredPopulation`。

**种群生命周期:**

```
                    [initialization]
                          │
    ┌─────────────────────▼─────────────────────┐
    │                                           │
    │  [first] → reproduction → [early]         │
    │     │                                     │
    │     └───── survival → [late] ─────────────┤
    │                         │                 │
    │                    age advance            │
    │                         │                 │
    └─────────────────────────┴─────────────────┘
                          │
                      [finish]
```

---

## API 参考

### Species

物种类，定义遗传架构的根节点。

```python
class Species(GeneticStructure):
    """物种遗传架构定义"""
```

#### 构造函数

```python
Species(name: str)
```

**参数:**
- `name` (str): 物种名称

#### 方法

| 方法 | 描述 |
|---|---|
| `add(name, sex_type=None)` | 添加染色体，返回 `Chromosome` |
| `get_all_genotypes()` | 获取所有可能的二倍体基因型列表 |
| `get_genotype_from_str(s)` | 从字符串解析基因型 |
| `get_gene(name)` | 按名称获取等位基因 |

#### 示例

```python
species = Species("Aedes aegypti")

# 添加常染色体和性染色体
species.add("Chr1")
species.add("ChrX", sex_type="X")
species.add("ChrY", sex_type="Y")

# 获取所有染色体
for chrom in species.chromosomes:
    print(f"{chrom.name}: {chrom.sex_type}")
```

---

### Chromosome

染色体（连锁群）定义，包含多个位点和重组信息。

```python
class Chromosome(GeneticStructure):
    """染色体结构定义"""
```

#### 构造函数

```python
Chromosome(
    name: str,
    sex_type: Optional[str | SexChromosomeType] = None,
    recombination_rates: Optional[List[float]] = None
)
```

**参数:**
- `name` (str): 染色体名称
- `sex_type` (str | SexChromosomeType): 性染色体类型
  - `None` 或 `'autosome'`: 常染色体（默认）
  - `'X'`: X 染色体
  - `'Y'`: Y 染色体（仅父本遗传）
  - `'Z'`: Z 染色体
  - `'W'`: W 染色体（仅母本遗传）
- `recombination_rates` (List[float]): 位点间重组率

#### 属性

| 属性 | 类型 | 描述 |
|---|---|---|
| `loci` | List[Locus] | 位点列表（按位置排序） |
| `sex_type` | SexChromosomeType | 性染色体类型 |
| `is_sex_chromosome` | bool | 是否为性染色体 |
| `recombination_map` | RecombinationMap | 重组图 |

#### 示例

```python
# 创建染色体并添加位点
chr2 = species.add("Chr2")
chr2.add("locus_A", position=0)
chr2.add("locus_B", position=50)

# 设置重组率
chr2.recombination_map[0] = 0.1  # A-B 间重组率 10%

# 性染色体
chr_x = species.add("ChrX", sex_type="X")
print(chr_x.is_sex_chromosome)  # True
print(chr_x.sex_system)  # "XY"
```

---

### Locus

基因位点定义。

```python
class Locus(GeneticStructure):
    """基因位点定义"""
```

#### 构造函数

```python
Locus(
    name: str,
    position: Optional[float] = None
)
```

**参数:**
- `name` (str): 位点名称
- `position` (float): 在染色体上的位置（用于计算重组）

#### 方法

| 方法 | 描述 |
|---|---|
| `add_alleles(names)` | 添加等位基因 |
| `alleles` | 获取所有等位基因列表 |

#### 类方法

```python
@classmethod
def with_alleles(cls, name: str, allele_names: List[str]) -> Locus
```

工厂方法，创建位点并同时添加等位基因。

#### 示例

```python
# 方式1: 分步添加
locus = chr2.add("A")
locus.add_alleles(["A1", "A2", "A3"])

# 方式2: 工厂方法
locus = Locus.with_alleles("A", ["A1", "A2", "A3"])

# 查看等位基因
for allele in locus.alleles:
    print(allele.name)
```

---

### AgeStructuredPopulation

年龄结构种群模型，支持世代重叠、精子存储等特性。

```python
class AgeStructuredPopulation(BasePopulation):
    """年龄结构种群模型"""
```

#### 构造函数

```python
AgeStructuredPopulation(
    species: Species,
    initial_population_distribution: Dict[str, Dict[str | Genotype, List[int] | Dict[int, int]]],
    n_ages: int = 8,
    name: str = "AgeStructuredPop",
    
    # 生存参数
    female_survival_rates: Optional[List[float] | Dict | Callable] = None,
    male_survival_rates: Optional[List[float] | Dict | Callable] = None,
    
    # 成年定义
    female_adult_ages: Optional[List[int]] = None,  # 默认 [2,3,4,5,6,7]
    male_adult_ages: Optional[List[int]] = None,    # 默认 [2,3,4]
    
    # 繁殖参数
    offspring_per_female: float = 2.0,
    sex_ratio: float = 0.5,  # 雌性比例
    
    # 精子存储
    use_sperm_storage: bool = True,
    sperm_displacement_rate: float = 0.05,
    adult_female_mating_rate: float = 1.0,
    
    # 幼虫招募
    juvenile_growth_mode: int = 2,
    recruitment_size: Optional[int] = None,
    
    # 遗传漂变
    effective_population_size: int = 0,
    
    # 修饰器
    gamete_modifiers: Optional[List[Tuple]] = None,
    zygote_modifiers: Optional[List[Tuple]] = None,
    
    # Hooks
    hooks: Optional[Dict[str, List[Tuple]]] = None,
    
    # 随机种子
    seed: Optional[int] = None,
)
```

**参数详解:**

##### `initial_population_distribution`

初始种群分布，格式为嵌套字典：

```python
{
    "female": {
        "genotype_string_or_object": [age0_count, age1_count, ...],
        # 或稀疏格式
        "genotype": {age: count, ...}
    },
    "male": {...}
}
```

##### `survival_rates`

生存率支持多种格式：

```python
# 列表（按年龄索引）
[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]

# 带哨兵的列表（用最后一个非 None 值填充）
[1.0, 0.8, None]  # → [1.0, 0.8, 0.8, 0.8, ...]

# 字典（缺省为 1.0）
{0: 1.0, 7: 0.0}

# 可调用对象
lambda age: max(0, 1 - age * 0.1)

# 常量
0.9  # 所有年龄相同
```

#### 核心属性

| 属性 | 类型 | 描述 |
|---|---|---|
| `state` | PopulationState | 当前种群状态 |
| `species` | Species | 遗传架构 |
| `tick` | int | 当前时间步 |
| `n_ages` | int | 年龄类别数 |
| `history` | List[Tuple] | 历史快照列表 |

#### 核心方法

| 方法 | 描述 |
|---|---|
| `run(n_steps, record_every=1, finish=False)` | 运行多步模拟 |
| `step()` | 执行单步 |
| `reset()` | 重置到初始状态 |
| `get_total_count()` | 获取总个体数 |
| `get_female_count()` | 获取雌性总数 |
| `get_male_count()` | 获取雄性总数 |
| `get_adult_count(sex)` | 获取成年个体数 |
| `create_snapshot()` | 创建状态快照 |
| `finish_simulation()` | 结束模拟 |

#### Fitness 设置方法

```python
# 设置存活力
pop.set_viability(genotype, value, sex=None)
pop.set_viability_batch({genotype: value, ...}, sex=None)

# 设置繁殖力
pop.set_fecundity(genotype, value, sex=None)
pop.set_fecundity_batch({genotype: value, ...})

# 设置性选择偏好
pop.set_sexual_selection(female_genotype, male_genotype, preference)
pop.set_sexual_selection_batch({(f_gt, m_gt): pref, ...})
```

#### 完整示例

```python
from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation

# 定义遗传架构
species = Species("TestSpecies")
chr1 = species.add("Chr1")
locus = chr1.add("A")
locus.add_alleles(["w", "d"])

# 初始种群
init_pop = {
    "female": {
        "w|w": {2: 100, 3: 80, 4: 60},  # 稀疏格式
    },
    "male": {
        "w|w": [0, 0, 50, 40],  # 密集格式
    }
}

# 创建种群
pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    female_survival_rates=[1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0],
    male_survival_rates=[1.0, 1.0, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0],
    offspring_per_female=10.0,
    use_sperm_storage=True,
    effective_population_size=1000,
)

# 设置基因型适应度
d_homo = species.get_genotype_from_str("d|d")
pop.set_viability(d_homo, 0.0)  # d|d 致死

# 运行模拟
pop.run(100, record_every=10)
```

---

### Hooks 系统

Hooks 允许在模拟生命周期的特定阶段注入自定义逻辑。

#### 支持的事件

| 事件 | 触发时机 |
|---|---|
| `initialization` | 种群初始化完成后 |
| `first` | 每个 tick 开始时 |
| `reproduction` | 繁殖阶段完成后 |
| `early` | 早期阶段（繁殖后、生存前） |
| `survival` | 生存阶段完成后 |
| `late` | 晚期阶段（tick 结束前） |
| `finish` | 模拟结束时 |

#### 注册 Hook

```python
# 方式1: 使用 set_hook 方法
def my_hook(pop):
    print(f"Tick {pop.tick}: 总数 {pop.get_total_count()}")

pop.set_hook("late", my_hook, hook_name="printer")

# 方式2: 在构造时传入
hooks = {
    "late": [
        (my_hook, "printer", 0),  # (func, name, priority)
    ],
    "finish": [
        (lambda p: print("Done!"),),
    ]
}
pop = AgeStructuredPopulation(..., hooks=hooks)
```

#### Hook 签名

所有 Hook 函数必须接受单个参数 `population`:

```python
def hook_func(population: BasePopulation) -> None:
    # 通过 population 对象访问所有状态
    tick = population.tick
    state = population.state
    # ...
```

#### 提前终止

Hook 可以调用 `finish_simulation()` 提前终止模拟：

```python
def extinction_check(pop):
    if pop.get_total_count() == 0:
        print("Population extinct!")
        pop.finish_simulation()

pop.set_hook("late", extinction_check)
```

#### 管理 Hooks

```python
# 获取某事件的所有 hooks
hooks = pop.get_hooks("late")

# 移除特定 hook
pop.remove_hook("late", hook_id=0)

# 手动触发事件
pop.trigger_event("early")
```

---

### Modifiers

修饰器系统允许自定义配子生成和合子形成规则，用于模拟基因驱动、不育等复杂遗传现象。

#### 配子修饰器 (Gamete Modifiers)

配子修饰器修改特定基因型产生配子的频率分布。

**签名:**

```python
def gamete_modifier(population) -> Dict[key, Dict[compressed_idx, freq]]:
    """
    返回格式:
    {
        "male": {genotype_key: {compressed_gamete_idx: frequency, ...}},
        "female": {...},
        # 或直接
        (sex_idx, genotype_key): {compressed_gamete_idx: frequency, ...},
    }
    """
```

**示例：基因驱动**

```python
def gene_drive_modifier(pop):
    """D|w 雄性的 D 配子比例提高到 90%"""
    Dw = pop.species.get_genotype_from_str("D|w")
    D_gamete = pop.registry.haplo_to_index[...]  # 获取 D 单倍型索引
    w_gamete = pop.registry.haplo_to_index[...]  # 获取 w 单倍型索引
    
    return {
        "male": {
            Dw: {
                D_gamete: 0.9,  # D 配子 90%
                w_gamete: 0.1,  # w 配子 10%
            }
        }
    }

pop = AgeStructuredPopulation(
    ...,
    gamete_modifiers=[(0, "gene_drive", gene_drive_modifier)]
)
```

#### 合子修饰器 (Zygote Modifiers)

合子修饰器修改配子组合产生特定基因型的概率。

**签名:**

```python
def zygote_modifier(population) -> Dict[key, replacement]:
    """
    key: (compressed_idx_female_gamete, compressed_idx_male_gamete)
    replacement: genotype_idx | {genotype_idx: probability, ...}
    """
```

**示例：细胞质不兼容**

```python
def ci_modifier(pop):
    """感染 Wolbachia 的雄性 × 未感染雌性 → 胚胎死亡"""
    infected_sperm = ...  # 感染雄性的精子索引
    uninfected_egg = ...  # 未感染雌性的卵子索引
    
    return {
        (uninfected_egg, infected_sperm): {}  # 空字典表示无后代
    }

pop = AgeStructuredPopulation(
    ...,
    zygote_modifiers=[(0, "CI", ci_modifier)]
)
```

#### 注册修饰器

```python
# 方式1: 构造时传入
pop = AgeStructuredPopulation(
    ...,
    gamete_modifiers=[
        (priority, name, callable),
        ...
    ],
    zygote_modifiers=[
        (priority, name, callable),
        ...
    ]
)

# 方式2: 动态添加
pop.set_gamete_modifier(modifier_func, hook_id=0, hook_name="my_modifier")
pop.set_zygote_modifier(modifier_func, hook_id=0, hook_name="my_modifier")
```

---

## 高级用法

### 纯函数模拟核心

`simulation_kernels` 模块提供了纯函数化的模拟核心，支持：

- 状态导出/导入
- 批量 Monte Carlo 模拟
- Numba 加速

```python
from utils.simulation_kernels import (
    export_state, import_state, 
    run_tick, run_ticks, batch_ticks
)

# 导出状态
state, config, history = export_state(pop)

# 运行 100 步
state, history = run_ticks(
    state, config, 
    n_steps=100, 
    rng=np.random.default_rng(),
    record_history=True
)

# 导入回种群对象
import_state(pop, state, history)

# 批量模拟（Monte Carlo）
particles = batch_ticks(
    state, config,
    n_particles=100,
    n_steps_per_particle=50,
    rng=np.random.default_rng()
)
```

### PopulationState 数据结构

`PopulationState` 是存储种群状态的数据容器：

```python
@dataclass
class PopulationState:
    # 个体计数: (n_sexes, n_ages, n_genotypes)
    individual_count: NDArray[np.float64]
    
    # 精子存储: (n_ages, n_genotypes, n_hg * n_glabs)
    sperm_storage: NDArray[np.float64]
    
    # 雌性占用率: (n_ages, n_genotypes)
    female_occupancy: NDArray[np.float64]
```

**访问方法:**

```python
from utils.type_def import make_indtype, Sex

# 创建个体类型标识
ind = make_indtype(Sex.FEMALE, age=2, genotype_index=0)

# 获取/设置计数
count = pop.state.get_count(ind)
pop.state.add_count(ind, delta=10)
```

### IndexCore 索引管理

`IndexCore` 提供实体到整数索引的映射：

```python
from utils.index_core import IndexCore

ic = pop.registry  # 或 pop._index_core

# 注册实体
gid = ic.register_genotype(genotype)
hid = ic.register_haplogenotype(haplotype)
glid = ic.register_gamete_label("label")

# 查询索引
idx = ic.genotype_index(genotype)
idx = ic.haplo_index(haplotype)

# 压缩索引（用于优化存储）
compressed = ic.compress_hg_glab(haplo_idx, glab_idx, n_glabs)
haplo_idx, glab_idx = ic.decompress_hg_glab(compressed, n_glabs)
```

---

## 性能优化

### Numba 加速

框架核心算法支持 Numba JIT 编译。使用 `@numba_switchable` 装饰器的函数可以动态切换加速模式：

```python
from utils.numba_utils import numba_switchable

@numba_switchable
def my_algorithm(x: np.ndarray) -> np.ndarray:
    # 纯 NumPy 实现
    return x * 2

# 使用 Numba 加速（默认）
result = my_algorithm(data)

# 禁用加速（调试用）
result = my_algorithm(data, use_numba=False)
```

### 大规模模拟建议

1. **使用纯函数核心**: 对于 Monte Carlo 模拟，使用 `batch_ticks` 避免对象创建开销
2. **减少历史记录**: 设置 `record_every` 为较大值或 0
3. **预编译**: 首次调用 Numba 函数会触发编译，可提前预热

```python
# 预热编译
_ = run_tick(state, config, rng, use_numba=True)

# 然后进行实际模拟
for _ in range(1000):
    state = run_tick(state, config, rng)
```

---

## 示例

### 基因驱动模拟

```python
"""模拟 suppression gene drive 在蚊子种群中的传播"""

from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation

# 定义物种
species = Species("Aedes aegypti")
chr2 = species.add("Chr2")
locus = chr2.add("drive_locus")
locus.add_alleles(["w", "D"])  # w: 野生型, D: 驱动等位基因

# 初始种群（仅野生型）
init_pop = {
    "female": {"w|w": [0, 0, 500, 400, 300, 200, 100, 0]},
    "male": {"w|w": [0, 0, 250, 200, 0, 0, 0, 0]},
}

# 创建种群
pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    offspring_per_female=8.0,
    female_survival_rates=[1.0, 1.0, 0.85, 0.7, 0.5, 0.3, 0.1, 0.0],
    male_survival_rates=[1.0, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0],
    use_sperm_storage=True,
    sperm_displacement_rate=0.1,
    effective_population_size=2000,
    seed=42,
)

# 设置适应度: D/D 雌性不育
Dw = species.get_genotype_from_str("D|w")
DD = species.get_genotype_from_str("D|D")
pop.set_fecundity(DD, 0.0, sex="female")  # D/D 雌性不育

# 基因驱动修饰器: D|w 杂合子的 D 配子比例 95%
def gene_drive(population):
    ic = population.registry
    Dw_idx = ic.genotype_index(Dw)
    
    # 获取 D 和 w 单倍型的压缩索引
    D_hap = population.species.get_haplotype_from_str("D")
    w_hap = population.species.get_haplotype_from_str("w")
    D_idx = ic.compress_hg_glab(ic.haplo_index(D_hap), 0, 1)
    w_idx = ic.compress_hg_glab(ic.haplo_index(w_hap), 0, 1)
    
    return {
        "male": {Dw_idx: {D_idx: 0.95, w_idx: 0.05}},
        "female": {Dw_idx: {D_idx: 0.95, w_idx: 0.05}},
    }

pop.set_gamete_modifier(gene_drive, hook_name="gene_drive")

# 释放携带驱动的雄性（在 tick 10）
def release_drive_males(population):
    if population.tick == 10:
        # 释放 100 只 D|w 杂合雄性（年龄 2）
        from utils.type_def import make_indtype, Sex
        Dw_idx = population.registry.genotype_index(Dw)
        ind = make_indtype(Sex.MALE, age=2, genotype_index=Dw_idx)
        population.state.add_count(ind, 100)
        print(f"Tick {population.tick}: Released 100 D|w males")

pop.set_hook("first", release_drive_males, hook_name="release")

# 监控钩子
history_data = []
def monitor(population):
    total = population.get_total_count()
    females = population.get_female_count()
    
    # 计算 D 等位基因频率
    ic = population.registry
    D_count = 0
    total_alleles = 0
    for gt in population.species.get_all_genotypes():
        gt_idx = ic.genotype_index(gt)
        count = population.state.individual_count[:, :, gt_idx].sum()
        # 计算该基因型中 D 的数量
        d_in_gt = str(gt).count('D')
        D_count += count * d_in_gt
        total_alleles += count * 2
    
    D_freq = D_count / total_alleles if total_alleles > 0 else 0
    history_data.append({
        'tick': population.tick,
        'total': total,
        'females': females,
        'D_freq': D_freq
    })

pop.set_hook("late", monitor, hook_name="monitor")

# 运行模拟
pop.run(200)

# 分析结果
import matplotlib.pyplot as plt

ticks = [h['tick'] for h in history_data]
totals = [h['total'] for h in history_data]
D_freqs = [h['D_freq'] for h in history_data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(ticks, totals)
ax1.axvline(x=10, color='r', linestyle='--', label='Release')
ax1.set_xlabel('Time (ticks)')
ax1.set_ylabel('Population Size')
ax1.set_title('Population Dynamics with Gene Drive')
ax1.legend()

ax2.plot(ticks, D_freqs, color='orange')
ax2.axvline(x=10, color='r', linestyle='--', label='Release')
ax2.set_xlabel('Time (ticks)')
ax2.set_ylabel('Drive Allele Frequency')
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig('gene_drive_simulation.png')
```

### 昆虫不育技术 (SIT) 模拟

```python
"""模拟释放不育雄性对野生种群的抑制效果"""

from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation
from utils.type_def import make_indtype, Sex

species = Species("Pest")
chr1 = species.add("Chr1")
locus = chr1.add("sterility")
locus.add_alleles(["W", "S"])  # W: 野生型, S: 不育标记

# 初始野生种群
init_pop = {
    "female": {"W|W": [0, 0, 200, 150, 100, 50, 0, 0]},
    "male": {"W|W": [0, 0, 100, 75, 0, 0, 0, 0]},
}

pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    offspring_per_female=6.0,
    use_sperm_storage=True,
    seed=123,
)

# S|S 和 W|S 的后代全部死亡（模拟辐射不育）
WS = species.get_genotype_from_str("W|S")
SS = species.get_genotype_from_str("S|S")

# 使用合子修饰器实现
def sterility_modifier(population):
    ic = population.registry
    result = {}
    
    # 获取所有涉及 S 等位基因的配子组合
    S_hap = population.species.get_haplotype_from_str("S")
    S_idx = ic.compress_hg_glab(ic.haplo_index(S_hap), 0, 1)
    
    # 任何涉及 S 精子的组合都不产生后代
    for egg_idx in range(ic.n_haplogenotypes):
        egg_compressed = ic.compress_hg_glab(egg_idx, 0, 1)
        result[(egg_compressed, S_idx)] = {}  # 空 = 无后代
    
    return result

pop.set_zygote_modifier(sterility_modifier, hook_name="sterility")

# 定期释放不育雄性
def periodic_release(population):
    if population.tick >= 20 and population.tick % 7 == 0:  # 每周释放
        SS_idx = population.registry.genotype_index(SS)
        ind = make_indtype(Sex.MALE, age=2, genotype_index=SS_idx)
        release_size = 50  # 每次释放 50 只
        population.state.add_count(ind, release_size)

pop.set_hook("first", periodic_release, hook_name="SIT_release")

# 运行
pop.run(150)
print(f"Final population: {pop.get_total_count()}")
```

### 多位点连锁模拟

```python
"""模拟两个连锁位点的遗传动态"""

from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation

species = Species("TwoLocus")
chr1 = species.add("Chr1")

# 添加两个连锁位点
locus_A = chr1.add("A", position=0)
locus_A.add_alleles(["A1", "A2"])

locus_B = chr1.add("B", position=50)
locus_B.add_alleles(["B1", "B2"])

# 设置重组率 (10%)
chr1.recombination_map[0] = 0.1

# 查看所有可能的单倍型
print("Haplotypes:")
for hap in chr1.get_all_haplotypes():
    print(f"  {hap}")
# 输出:
#   A1/B1
#   A1/B2
#   A2/B1
#   A2/B2

# 查看所有可能的基因型
print(f"\nTotal genotypes: {len(species.get_all_genotypes())}")

# 初始种群: 只有 A1/B1|A1/B1 和 A2/B2|A2/B2
init_pop = {
    "female": {
        "A1/B1|A1/B1": [0, 0, 100, 0, 0, 0, 0, 0],
        "A2/B2|A2/B2": [0, 0, 100, 0, 0, 0, 0, 0],
    },
    "male": {
        "A1/B1|A1/B1": [0, 0, 50, 0, 0, 0, 0, 0],
        "A2/B2|A2/B2": [0, 0, 50, 0, 0, 0, 0, 0],
    }
}

pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    offspring_per_female=4.0,
    seed=42,
)

# 追踪重组体频率
def track_recombinants(population):
    if population.tick % 10 == 0:
        ic = population.registry
        # 重组单倍型: A1/B2 和 A2/B1
        recomb_haps = ["A1/B2", "A2/B1"]
        recomb_count = 0
        total_haps = 0
        
        for gt in population.species.get_all_genotypes():
            gt_idx = ic.genotype_index(gt)
            count = population.state.individual_count[:, :, gt_idx].sum()
            if count > 0:
                # 检查基因型中的重组单倍型
                gt_str = str(gt)
                for rh in recomb_haps:
                    if rh in gt_str:
                        recomb_count += count
                total_haps += count * 2
        
        if total_haps > 0:
            print(f"Tick {population.tick}: Recombinant freq = {recomb_count/total_haps:.4f}")

pop.set_hook("late", track_recombinants)
pop.run(100)
```

### 性染色体遗传模拟

```python
"""模拟 X 连锁基因的遗传"""

from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation

species = Species("XLinked")

# 添加 X 和 Y 染色体
chr_x = species.add("ChrX", sex_type="X")
chr_y = species.add("ChrY", sex_type="Y")

# X 连锁位点
white_locus = chr_x.add("white")
white_locus.add_alleles(["w+", "w"])  # w+: 红眼, w: 白眼

# Y 染色体标记
y_marker = chr_y.add("Ymarker")
y_marker.add_alleles(["Y"])

# 查看性别特异的基因型
all_gts = species.get_all_genotypes()
print("All genotypes:")
for gt in all_gts:
    print(f"  {gt} (sex: {gt.sex_type})")

# 初始种群
init_pop = {
    "female": {
        "w+|w+": [0, 0, 50, 40, 30, 0, 0, 0],  # XX 红眼雌性
        "w+|w": [0, 0, 20, 15, 10, 0, 0, 0],   # XX 杂合雌性（红眼）
    },
    "male": {
        "w+|Y": [0, 0, 30, 20, 0, 0, 0, 0],    # XY 红眼雄性
        "w|Y": [0, 0, 10, 5, 0, 0, 0, 0],      # XY 白眼雄性
    }
}

pop = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    offspring_per_female=5.0,
    seed=42,
)

# 设置白眼雄性适应度降低
w_Y = species.get_genotype_from_str("w|Y")
pop.set_viability(w_Y, 0.7)  # 白眼雄性存活率降低 30%

pop.run(50)
```

### Monte Carlo 批量模拟

```python
"""使用纯函数核心进行批量 Monte Carlo 模拟"""

import numpy as np
from utils.genetic_structures import Species
from utils.nonWF_population import AgeStructuredPopulation
from utils.simulation_kernels import export_state, batch_ticks

# 设置物种和初始种群
species = Species("MC_Test")
chr1 = species.add("Chr1")
chr1.add("A").add_alleles(["w", "d"])

init_pop = {
    "female": {"w|w": [0, 0, 100, 80, 60, 40, 20, 0]},
    "male": {"w|w": [0, 0, 50, 40, 0, 0, 0, 0]},
}

# 创建模板种群
template = AgeStructuredPopulation(
    species=species,
    initial_population_distribution=init_pop,
    n_ages=8,
    offspring_per_female=5.0,
)

# 导出状态和配置
state, config, _ = export_state(template)

# 运行 1000 次 Monte Carlo 模拟，每次 100 步
rng = np.random.default_rng(seed=12345)
n_particles = 1000
n_steps = 100

results = batch_ticks(
    state, config,
    n_particles=n_particles,
    n_steps_per_particle=n_steps,
    rng=rng,
    record_history=False,  # 只保留最终状态
)

# 分析结果
final_populations = []
for particle_state in results:
    total = particle_state.individual_count.sum()
    final_populations.append(total)

final_populations = np.array(final_populations)
print(f"Mean final population: {final_populations.mean():.1f}")
print(f"Std: {final_populations.std():.1f}")
print(f"Extinction probability: {(final_populations == 0).mean():.3f}")

# 绘制分布
import matplotlib.pyplot as plt
plt.hist(final_populations, bins=50, edgecolor='black')
plt.xlabel('Final Population Size')
plt.ylabel('Frequency')
plt.title(f'Distribution of Final Population Size (n={n_particles})')
plt.savefig('monte_carlo_results.png')
```

---

## API 完整参考

### utils.genetic_structures

#### Species

```python
class Species(GeneticStructure):
    def __init__(self, name: str): ...
    
    # 染色体管理
    def add(self, name: str, sex_type: Optional[str] = None) -> Chromosome: ...
    @property
    def chromosomes(self) -> List[Chromosome]: ...
    @property
    def autosomes(self) -> List[Chromosome]: ...
    @property
    def sex_chromosomes(self) -> List[Chromosome]: ...
    
    # 基因型获取
    def get_all_genotypes(self) -> List[Genotype]: ...
    def get_genotype_from_str(self, s: str) -> Genotype: ...
    def get_gene(self, name: str) -> Gene: ...
    def get_haplotype_from_str(self, s: str) -> Haplotype: ...
    
    # 重组
    def get_recombined_haplotype(self, mat: Haplotype, pat: Haplotype, rng) -> Haplotype: ...
```

#### Chromosome

```python
class Chromosome(GeneticStructure):
    def __init__(
        self, 
        name: str, 
        sex_type: Optional[str | SexChromosomeType] = None,
        recombination_rates: Optional[List[float]] = None
    ): ...
    
    # 位点管理
    def add(self, name: str, position: Optional[float] = None) -> Locus: ...
    @property
    def loci(self) -> List[Locus]: ...
    
    # 性染色体属性
    @property
    def sex_type(self) -> SexChromosomeType: ...
    @property
    def is_sex_chromosome(self) -> bool: ...
    @property
    def sex_system(self) -> Optional[str]: ...  # "XY" 或 "ZW"
    
    # 单倍型
    def get_all_haplotypes(self) -> List[Haplotype]: ...
    
    # 重组
    @property
    def recombination_map(self) -> RecombinationMap: ...
```

#### Locus

```python
class Locus(GeneticStructure):
    def __init__(self, name: str, position: Optional[float] = None): ...
    
    def add_alleles(self, names: List[str]) -> None: ...
    
    @property
    def alleles(self) -> List[Gene]: ...
    
    @classmethod
    def with_alleles(cls, name: str, allele_names: List[str]) -> Locus: ...
```

### utils.genetic_entities

#### Gene

```python
class Gene(GeneticEntity):
    @property
    def name(self) -> str: ...
    @property
    def locus(self) -> Locus: ...
```

#### Haplotype

```python
class Haplotype(GeneticEntity):
    @property
    def genes(self) -> Tuple[Gene, ...]: ...
    @property
    def chromosome(self) -> Chromosome: ...
    
    def __getitem__(self, locus: Locus) -> Gene: ...
```

#### HaploidGenome

```python
class HaploidGenome(GeneticEntity):
    @property
    def haplotypes(self) -> Tuple[Haplotype, ...]: ...
    @property
    def species(self) -> Species: ...
    
    def __getitem__(self, chromosome: Chromosome) -> Haplotype: ...
```

#### Genotype

```python
class Genotype(GeneticEntity):
    @property
    def maternal(self) -> HaploidGenome: ...
    @property
    def paternal(self) -> HaploidGenome: ...
    @property
    def species(self) -> Species: ...
    @property
    def sex_type(self) -> Optional[str]: ...  # "female", "male", None
    
    def get_gene_pair(self, locus: Locus) -> Tuple[Gene, Gene]: ...
    def is_homozygous(self, locus: Optional[Locus] = None) -> bool: ...
    def is_heterozygous(self, locus: Optional[Locus] = None) -> bool: ...
```

### utils.nonWF_population

#### AgeStructuredPopulation

```python
class AgeStructuredPopulation(BasePopulation):
    def __init__(
        self,
        species: Species,
        initial_population_distribution: Dict,
        n_ages: int = 8,
        name: str = "AgeStructuredPop",
        female_survival_rates: Optional[...] = None,
        male_survival_rates: Optional[...] = None,
        female_adult_ages: Optional[List[int]] = None,
        male_adult_ages: Optional[List[int]] = None,
        offspring_per_female: float = 2.0,
        sex_ratio: float = 0.5,
        use_sperm_storage: bool = True,
        sperm_displacement_rate: float = 0.05,
        adult_female_mating_rate: float = 1.0,
        juvenile_growth_mode: int = 2,
        recruitment_size: Optional[int] = None,
        effective_population_size: int = 0,
        gamete_modifiers: Optional[List[Tuple]] = None,
        zygote_modifiers: Optional[List[Tuple]] = None,
        hooks: Optional[Dict] = None,
        seed: Optional[int] = None,
    ): ...
    
    # 模拟控制
    def run(self, n_steps: int, record_every: int = 1, finish: bool = False) -> None: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def finish_simulation(self) -> None: ...
    
    # 状态查询
    def get_total_count(self) -> int: ...
    def get_female_count(self) -> int: ...
    def get_male_count(self) -> int: ...
    def get_adult_count(self, sex: Optional[str] = None) -> int: ...
    def create_snapshot(self) -> Tuple[int, PopulationState]: ...
    
    # 适应度设置
    def set_viability(self, genotype, value: float, sex: Optional[str] = None) -> None: ...
    def set_viability_batch(self, mapping: Dict, sex: Optional[str] = None) -> None: ...
    def set_fecundity(self, genotype, value: float, sex: Optional[str] = None) -> None: ...
    def set_fecundity_batch(self, mapping: Dict, sex: Optional[str] = None) -> None: ...
    def set_sexual_selection(self, female_gt, male_gt, preference: float) -> None: ...
    def set_sexual_selection_batch(self, mapping: Dict) -> None: ...
    
    # Hooks 管理
    def set_hook(self, event: str, func: Callable, hook_id: int = 0, hook_name: str = "") -> None: ...
    def get_hooks(self, event: str) -> List[Tuple]: ...
    def remove_hook(self, event: str, hook_id: int) -> None: ...
    def trigger_event(self, event: str) -> None: ...
    
    # 修饰器管理
    def set_gamete_modifier(self, func: Callable, hook_id: int = 0, hook_name: str = "") -> None: ...
    def set_zygote_modifier(self, func: Callable, hook_id: int = 0, hook_name: str = "") -> None: ...
```

### utils.population_state

#### PopulationState

```python
@dataclass
class PopulationState:
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_genotypes, n_hg * n_glabs)
    female_occupancy: NDArray[np.float64]  # (n_ages, n_genotypes)
    
    def get_count(self, ind: IndType) -> float: ...
    def set_count(self, ind: IndType, value: float) -> None: ...
    def add_count(self, ind: IndType, delta: float) -> None: ...
    
    def copy(self) -> PopulationState: ...
    def to_dict(self) -> Dict: ...
    
    @classmethod
    def from_dict(cls, d: Dict) -> PopulationState: ...
    
    @classmethod
    def zeros(cls, n_sexes: int, n_ages: int, n_genotypes: int, 
              n_haplogenotypes: int, n_gamete_labels: int) -> PopulationState: ...
```

### utils.type_def

#### Sex 枚举

```python
class Sex(IntEnum):
    FEMALE = 0
    MALE = 1
```

#### IndType

```python
# IndType 是一个命名元组，用于标识特定的个体类型
IndType = namedtuple('IndType', ['sex', 'age', 'genotype_index'])

def make_indtype(sex: Sex | int, age: int, genotype_index: int) -> IndType: ...
```

### utils.index_core

#### IndexCore

```python
class IndexCore:
    def __init__(self, species: Species): ...
    
    # 注册实体
    def register_genotype(self, genotype: Genotype) -> int: ...
    def register_haplogenotype(self, haplotype: Haplotype) -> int: ...
    def register_gamete_label(self, label: str) -> int: ...
    
    # 查询索引
    def genotype_index(self, genotype: Genotype) -> int: ...
    def haplo_index(self, haplotype: Haplotype) -> int: ...
    def gamete_label_index(self, label: str) -> int: ...
    
    # 反向查询
    def genotype_from_index(self, idx: int) -> Genotype: ...
    def haplo_from_index(self, idx: int) -> Haplotype: ...
    
    # 压缩索引
    def compress_hg_glab(self, hg_idx: int, glab_idx: int, n_glabs: int) -> int: ...
    def decompress_hg_glab(self, compressed: int, n_glabs: int) -> Tuple[int, int]: ...
    
    # 属性
    @property
    def n_genotypes(self) -> int: ...
    @property
    def n_haplogenotypes(self) -> int: ...
    @property
    def n_gamete_labels(self) -> int: ...
```

### utils.simulation_kernels

```python
def export_state(population: BasePopulation) -> Tuple[PopulationState, SimConfig, List]: ...

def import_state(population: BasePopulation, state: PopulationState, history: List) -> None: ...

def run_tick(
    state: PopulationState,
    config: SimConfig,
    rng: np.random.Generator,
    use_numba: bool = True,
) -> PopulationState: ...

def run_ticks(
    state: PopulationState,
    config: SimConfig,
    n_steps: int,
    rng: np.random.Generator,
    record_history: bool = True,
    record_every: int = 1,
) -> Tuple[PopulationState, List]: ...

def batch_ticks(
    initial_state: PopulationState,
    config: SimConfig,
    n_particles: int,
    n_steps_per_particle: int,
    rng: np.random.Generator,
    record_history: bool = False,
) -> List[PopulationState]: ...
```

---

## 变更日志

### v0.1.0 (开发中)

**新功能:**
- 初始版本
- 年龄结构种群模型 (`AgeStructuredPopulation`)
- 灵活的遗传架构定义 (`Species`, `Chromosome`, `Locus`)
- 精子存储机制
- 配子/合子修饰器系统
- Hook 事件系统
- Numba 加速支持
- 纯函数模拟核心

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<p align="center">
  <i>PopGen-Sim - 让遗传学模拟更简单</i>
</p>