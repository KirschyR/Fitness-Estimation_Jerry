# 🗺️ 框架开发路线图

## 杂项

- [x] 基因命名采用白名单制，只允许使用字母、数字和下划线。

## 按优先级分类

### 🔴 P0: 核心功能修复（发布前必须）

| 问题 | 描述 | 工作量 |
|------|------|--------|
| **重复的 `run_tick` 逻辑** | 两套逻辑增加维护成本，容易出错 | 2-3天 |
| **连锁交换验证** | 需要单元测试确认 crossover 正确性 | 1-2天 |
| **性染色体验证** | 需要测试 X/Y 遗传模式 | 1-2天 |
| **参数混乱** | 统一命名规范，整理配置结构 | 2-3天 |

### 🟠 P1: 核心功能增强（v0.2）

| 功能 | 描述 | 工作量 |
|------|------|--------|
| **动态 Ne** | `effective_population_size` 随实际种群大小变化 | 1天 |
| **Logistic 增长** | 添加 `low_density_growth_rate` + 幼虫竞争强度 | 2-3天 |
| **灵活的遗传漂变** | 可配置的漂变模型 | 2天 |
| **Wright-Fisher 模型** | 简单的非年龄结构模型 | 3-5天 |

### 🟡 P2: 扩展功能（v0.3）

| 功能 | 描述 | 工作量 |
|------|------|--------|
| **空间模型** | 多种群 + 迁移矩阵 | 1-2周 |
| **Somatic label** | 与 gamete label 对应的体细胞标记 | 3-5天 |
| **Numba 灵活调节** | 可配置的 JIT 选项 | 2-3天 |

### 🟢 P3: 用户体验（v0.4+）

| 功能 | 描述 | 工作量 |
|------|------|--------|
| **Web UI** | 简单的可视化界面 | 1-2周 |
| **改进 Hook 系统** | 更适合矩阵运算的回调机制 | 1周 |

---

## 📋 详细技术方案

### 1. 合并重复的 `run_tick` 逻辑

```python
# 建议：抽象出通用的 tick 流程
class TickExecutor:
    """统一的 tick 执行器"""
    
    def __init__(self, population: BasePopulation):
        self.pop = population
        self.stages: list[TickStage] = []
    
    def register_stage(self, stage: TickStage, order: int):
        """注册生命周期阶段"""
        self.stages.append((order, stage))
        self.stages.sort(key=lambda x: x[0])
    
    def run(self):
        """执行一个 tick"""
        for _, stage in self.stages:
            stage.execute(self.pop)

# 不同模型只需注册不同的 stages
class AgeStructuredExecutor(TickExecutor):
    def __init__(self, pop):
        super().__init__(pop)
        self.register_stage(AgingStage(), 10)
        self.register_stage(MortalityStage(), 20)
        self.register_stage(ReproductionStage(), 30)
        self.register_stage(RecruitmentStage(), 40)
```

---

### 2. 动态 Effective Population Size

```python
# 当前（推测）
self.Ne = config.effective_population_size  # 固定值

# 改进方案
@property
def effective_population_size(self) -> float:
    """动态计算 Ne"""
    N = self.census_size
    
    # 方案1: 简单比例
    # return N * self.config.ne_ratio
    
    # 方案2: 基于性别比例的经典公式
    # Ne = 4 * Nm * Nf / (Nm + Nf)
    n_males = self.count_by_sex(Sex.MALE)
    n_females = self.count_by_sex(Sex.FEMALE)
    if n_males + n_females == 0:
        return 0.0
    return 4 * n_males * n_females / (n_males + n_females)
    
    # 方案3: 考虑年龄结构
    # Ne = N / (1 + Var(k) / mean(k))  # k = 个体繁殖贡献
```

---

### 3. Logistic 增长模型

```python
@dataclass
class LogisticGrowthConfig:
    """Logistic 增长参数"""
    carrying_capacity: int              # K - 环境容纳量
    low_density_growth_rate: float      # r - 内禀增长率
    larval_competition_intensity: float # α - 幼虫竞争强度 (0-1)
    density_dependence: str = "ceiling" # "ceiling" | "logistic" | "contest"

class LogisticRecruitment:
    """Logistic 增长的招募模型"""
    
    def __init__(self, config: LogisticGrowthConfig):
        self.config = config
    
    def calculate_recruitment(self, 
                              current_size: int, 
                              potential_offspring: int) -> int:
        """计算实际招募数量"""
        K = self.config.carrying_capacity
        r = self.config.low_density_growth_rate
        α = self.config.larval_competition_intensity
        N = current_size
        
        if self.config.density_dependence == "ceiling":
            # 简单天花板模型
            return min(potential_offspring, K - N)
        
        elif self.config.density_dependence == "logistic":
            # 经典 Logistic: dN/dt = rN(1 - N/K)
            growth_rate = r * (1 - N / K)
            expected = int(N * growth_rate)
            # 幼虫竞争降低实际招募
            survival_rate = 1 / (1 + α * potential_offspring / K)
            return int(potential_offspring * survival_rate)
        
        elif self.config.density_dependence == "contest":
            # 竞争模型：固定数量的"槽位"
            available_slots = max(0, K - N)
            if potential_offspring <= available_slots:
                return potential_offspring
            else:
                # 竞争存活
                return available_slots
```

---

### 4. Wright-Fisher 模型（简单非年龄结构）

```python
class WrightFisherPopulation(BasePopulation):
    """经典 Wright-Fisher 模型
    
    - 离散世代（非重叠）
    - 固定种群大小
    - 随机配对
    """
    
    def __init__(self, 
                 species: Species,
                 population_size: int,
                 sex_ratio: float = 0.5):
        super().__init__(species)
        self.N = population_size
        self.sex_ratio = sex_ratio
        self._initialize_population()
    
    def run_generation(self):
        """运行一个世代"""
        # 1. 所有个体产生配子
        gamete_pool = self._generate_gamete_pool()
        
        # 2. 随机抽样形成下一代
        new_individuals = self._sample_offspring(gamete_pool, self.N)
        
        # 3. 完全替换（离散世代）
        self.replace_all(new_individuals)
        
        self.generation += 1
    
    def _sample_offspring(self, gamete_pool, n: int):
        """从配子池随机抽样产生后代"""
        # Wright-Fisher: 每个后代独立从亲本池抽样
        # 等价于多项式抽样
        ...
```

---

### 5. 空间模型架构

```python
class MetaPopulation:
    """元种群：多个空间上分离的种群 + 迁移"""
    
    def __init__(self, 
                 species: Species,
                 n_demes: int,
                 migration_matrix: np.ndarray):
        """
        migration_matrix[i,j] = 从 deme i 迁移到 deme j 的概率
        """
        self.demes: list[BasePopulation] = []
        self.migration_matrix = migration_matrix
        
    def run_tick(self):
        """运行一个时间步"""
        # 1. 各 deme 独立演化
        for deme in self.demes:
            deme.run_tick()
        
        # 2. 迁移
        self._apply_migration()
    
    def _apply_migration(self):
        """应用迁移矩阵"""
        migrants = []  # [(from_deme, to_deme, individual), ...]
        
        for i, deme in enumerate(self.demes):
            for ind in deme.individuals:
                dest = np.random.choice(
                    len(self.demes), 
                    p=self.migration_matrix[i]
                )
                if dest != i:
                    migrants.append((i, dest, ind))
        
        # 执行迁移
        for from_d, to_d, ind in migrants:
            self.demes[from_d].remove(ind)
            self.demes[to_d].add(ind)


# 常用迁移模型
def island_model(n_demes: int, m: float) -> np.ndarray:
    """岛屿模型：等概率迁移到任意其他 deme"""
    M = np.full((n_demes, n_demes), m / (n_demes - 1))
    np.fill_diagonal(M, 1 - m)
    return M

def stepping_stone_1d(n_demes: int, m: float) -> np.ndarray:
    """一维踏脚石模型：只能迁移到相邻 deme"""
    M = np.zeros((n_demes, n_demes))
    for i in range(n_demes):
        M[i, i] = 1 - m
        if i > 0:
            M[i, i-1] = m / 2
        if i < n_demes - 1:
            M[i, i+1] = m / 2
    # 边界处理
    M[0, 0] = 1 - m/2
    M[-1, -1] = 1 - m/2
    return M
```

---

### 6. 改进 Hook 系统（适配矩阵运算）

```python
# 当前问题：hook 在 Numba 编译的函数内部难以调用

# 解决方案：事件收集 + 批处理

class EventCollector:
    """收集事件，在 Numba 外部批处理"""
    
    def __init__(self):
        self.events: dict[str, list] = defaultdict(list)
    
    def record(self, event_type: str, data: np.ndarray):
        """记录事件数据（在 Numba 函数返回后调用）"""
        self.events[event_type].append(data)
    
    def flush(self, hooks: dict[str, Callable]):
        """执行所有 hook"""
        for event_type, data_list in self.events.items():
            if event_type in hooks:
                # 合并同类事件，批量处理
                combined = np.concatenate(data_list) if data_list else np.array([])
                hooks[event_type](combined)
        self.events.clear()


# 使用示例
class Population:
    def run_tick(self):
        collector = EventCollector()
        
        # Numba 函数返回事件数据而非调用 hook
        death_indices = _numba_mortality(self.state)
        collector.record("death", death_indices)
        
        birth_data = _numba_reproduction(self.state)
        collector.record("birth", birth_data)
        
        # 在 Python 层面执行 hook
        collector.flush(self.hooks)
```

---

### 7. 简单 Web UI（Gradio 方案）

```python
# 使用 Gradio 快速搭建
import gradio as gr

def create_simulation_ui(species: Species):
    """创建模拟 Web 界面"""
    
    def run_simulation(pop_size, generations, mutation_rate):
        pop = AgeStructuredPopulation(species, pop_size)
        pop.set_mutation_rate(mutation_rate)
        
        history = []
        for gen in range(generations):
            pop.run_tick()
            history.append({
                'generation': gen,
                'size': pop.census_size,
                'heterozygosity': pop.mean_heterozygosity()
            })
        
        # 返回图表
        import pandas as pd
        df = pd.DataFrame(history)
        return df.plot(x='generation', y=['size', 'heterozygosity'])
    
    interface = gr.Interface(
        fn=run_simulation,
        inputs=[
            gr.Slider(10, 10000, value=1000, label="Population Size"),
            gr.Slider(10, 1000, value=100, label="Generations"),
            gr.Slider(0, 0.1, value=0.001, label="Mutation Rate"),
        ],
        outputs=gr.Plot(),
        title="PopGen Simulator"
    )
    
    return interface

# 启动
# ui = create_simulation_ui(my_species)
# ui.launch()
```

---

## 📅 建议开发计划

```
Phase 1 (v0.1.0) - 2周
├── 修复重复 run_tick 逻辑
├── 验证连锁交换 + 性染色体
├── 整理参数命名
└── 发布 alpha

Phase 2 (v0.2.0) - 3周
├── 动态 Ne
├── Logistic 增长
├── Wright-Fisher 模型
└── 改进遗传漂变

Phase 3 (v0.3.0) - 4周
├── 空间模型
├── Somatic label
├── 改进 Hook 系统
└── Numba 配置优化

Phase 4 (v0.4.0) - 2周
├── Web UI
└── 文档 + 教程
```

---

## 💡 建议

1. **先发布 v0.1.0-alpha**：当前代码足够运行，先让用户用起来收集反馈
2. **P0 问题优先**：重复逻辑和验证测试最重要
3. **Logistic 增长**：这是群体遗传学常用模型，建议 P1 优先实现
4. **Web UI 可以延后**：命令行 + Jupyter 够用，UI 是锦上添花

您想先从哪个方向开始？我可以帮您详细设计某个具体功能的实现方案。