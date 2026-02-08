"""纯函数化模拟核心——在 Population 外部运行、支持 Numba 加速。

核心设计：
1. export_state(pop) → (state_arrays, config, history)      # 导出（含历史）
2. run_tick(state_arrays, config, seed, counter) → state   # 纯函数 tick
3. run_ticks(state_arrays, config, n, seed) → (state, history) # 循环运行
4. batch_ticks(...) → batch_states  # 批量（MC采样）
5. import_state(pop, state_arrays, history)                 # 导入（含历史）

所有操作都可被 Numba JIT 编译，适合大规模并行采样。

随机数生成使用 seed + counter 模式，确保 Numba 兼容性：
- seed: 基础随机数种子（整数）
- counter: 调用计数器，每次调用递增，确保不同调用产生不同随机序列
- 内部使用 np.random.seed(seed + counter) 生成随机数

使用例子：
    >>> pop = AgeStructuredPopulation(...)
    >>> state, config, history = export_state(pop)  # 导出包含历史
    >>> 
    >>> # 单个模拟：100 steps，记录历史
    >>> state, history = run(state, config, 100, seed=42, 
    ...                       record_history=True)
    >>> import_state(pop, state, history)  # 导入并恢复历史
    
    >>> # 批量（MC）：100 粒子 × 50步
    >>> particles, _ = batch_ticks(state, config, n_particles=100, 
    ...                             n_steps_per_particle=50, seed=42)
    >>> means = [p[STATE_INDIVIDUAL_COUNT].sum() for p in particles]
    >>> print(f"平均: {np.mean(means)}")
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional, List, Callable, Union, TYPE_CHECKING
import utils.algorithms as alg
from utils.type_def import Sex
from utils.numba_utils import numba_switchable

if TYPE_CHECKING:
    from utils.nonWF_population import AgeStructuredPopulation


# ---------------------------------------------------------------------------
# 索引常数（用于将 dict 状态扁平化为 tuple/array，方便 numba 使用）
# ---------------------------------------------------------------------------
# 使用这些常数可以在不使用 Python 字典的情况下通过整数索引访问状态元素，
# 更容易转换为 numba.typed containers 或直接传入 @numba_switchable 的函数。
STATE_INDIVIDUAL_COUNT = 0
STATE_SPERM_STORAGE = 1
STATE_FEMALE_OCC = 2
STATE_TICK = 3

# ---------------------------------------------------------------------------
# 配置索引常量（对应 export_state 返回的 config tuple 顺序）
# ---------------------------------------------------------------------------
# 注意：这些常量用于通过整数索引访问 config tuple 中的元素，
# 使内部内核能在不依赖 Python dict 的情况下运行，从而便于 Numba 加速。
CFG_N_AGES = 0
CFG_N_GENOTYPES = 1
CFG_N_HAPLOTYPES = 2
CFG_N_GLABS = 3
CFG_ADULT_AGES = 4
CFG_ADULT_FEMALE_MATING_RATE = 5
CFG_SPERM_DISPLACEMENT_RATE = 6
CFG_OFFSPRING_PER_FEMALE = 7
CFG_CARRYING_CAPACITY = 8
CFG_SEX_RATIO = 9
CFG_EFFECTIVE_POP_SIZE = 10
CFG_FEMALE_SURVIVAL = 11
CFG_MALE_SURVIVAL = 12
CFG_SEXUAL_SELECTION = 13
CFG_MALE_FECUNDITY = 14
CFG_FEMALE_FECUNDITY = 15
CFG_GENOTYPE_TO_GAMETES_FEMALE = 16
CFG_GENOTYPE_TO_GAMETES_MALE = 17
CFG_GAMETES_TO_ZYGOTE = 18
CFG_FEMALE_VIABILITY = 19
CFG_MALE_VIABILITY = 20

# 兼容别名
CFG_MEIOSIS_F = CFG_GENOTYPE_TO_GAMETES_FEMALE
CFG_MEIOSIS_M = CFG_GENOTYPE_TO_GAMETES_MALE
CFG_HAPLO_TO_GENOTYPE = CFG_GAMETES_TO_ZYGOTE


def state_dict_to_tuple(state: Dict[str, NDArray]):
    """将 state dict 转换为固定顺序的 tuple: (individual_count, sperm_storage, female_occupancy, tick).

    该函数仅用于在 Python 层将现有 dict 转为按索引访问的 tuple，便于构造 numba-friendly
    结构（例如 numba.typed.List/Dict 或原始 tuple/ndarray 输入）。
    """
    return (
        state['individual_count'],
        state['sperm_storage'],
        state['female_occupancy'],
        state['tick'],
    )


def state_tuple_to_dict(tup):
    """将固定顺序的 state tuple 恢复为与现有代码兼容的 dict。"""
    return {
        'individual_count': tup[STATE_INDIVIDUAL_COUNT],
        'sperm_storage': tup[STATE_SPERM_STORAGE],
        'female_occupancy': tup[STATE_FEMALE_OCC],
        'tick': tup[STATE_TICK],
    }


def config_dict_to_tuple(cfg: Dict[str, Any]):
    """将配置 dict 转为固定顺序的 tuple（与 export_state 产生的顺序对应）。

    期望的 dict 键名为：
      'n_ages','n_genotypes','n_haplotypes','n_glabs','adult_ages',
      'adult_female_mating_rate','sperm_displacement_rate','offspring_per_female',
      'carrying_capacity','sex_ratio','effective_population_size',
      'female_survival_rates','male_survival_rates','sexual_selection',
      'male_fecundity','female_fecundity',
      'genotype_to_gametes_map_female','genotype_to_gametes_map_male','gametes_to_zygote_map',
      'female_viability','male_viability'

    如果传入的 dict 使用了不同命名，调用方需要先将其映射为上述键。
    """
    return (
        cfg['n_ages'],
        cfg['n_genotypes'],
        cfg['n_haplotypes'],
        cfg.get('n_glabs', 1),
        tuple(cfg.get('adult_ages', ())),
        cfg.get('adult_female_mating_rate'),
        cfg.get('sperm_displacement_rate'),
        cfg.get('offspring_per_female'),
        cfg.get('carrying_capacity', cfg.get('recruitment_size')),
        cfg.get('sex_ratio'),
        cfg.get('effective_population_size'),
        cfg.get('female_survival_rates'),
        cfg.get('male_survival_rates'),
        cfg.get('sexual_selection'),
        cfg.get('male_fecundity'),
        cfg.get('female_fecundity'),
        cfg.get('genotype_to_gametes_map_female') or cfg.get('genotype_to_gametes_map')[Sex.FEMALE.value],
        cfg.get('genotype_to_gametes_map_male') or cfg.get('genotype_to_gametes_map')[Sex.MALE.value],
        cfg.get('gametes_to_zygote_map'),
        cfg.get('female_viability'),
        cfg.get('male_viability'),
    )


def config_tuple_to_dict(tup: tuple) -> Dict[str, Any]:
    """将按索引的 config tuple 恢复为字典形式（便于向后兼容）。"""
    return {
        'n_ages': tup[CFG_N_AGES],
        'n_genotypes': tup[CFG_N_GENOTYPES],
        'n_haplotypes': tup[CFG_N_HAPLOTYPES],
        'n_glabs': tup[CFG_N_GLABS],
        'adult_ages': tup[CFG_ADULT_AGES],
        'adult_female_mating_rate': tup[CFG_ADULT_FEMALE_MATING_RATE],
        'sperm_displacement_rate': tup[CFG_SPERM_DISPLACEMENT_RATE],
        'offspring_per_female': tup[CFG_OFFSPRING_PER_FEMALE],
        'carrying_capacity': tup[CFG_CARRYING_CAPACITY],
        'sex_ratio': tup[CFG_SEX_RATIO],
        'effective_population_size': tup[CFG_EFFECTIVE_POP_SIZE],
        'female_survival_rates': tup[CFG_FEMALE_SURVIVAL],
        'male_survival_rates': tup[CFG_MALE_SURVIVAL],
        'sexual_selection': tup[CFG_SEXUAL_SELECTION],
        'male_fecundity': tup[CFG_MALE_FECUNDITY],
        'female_fecundity': tup[CFG_FEMALE_FECUNDITY],
        'genotype_to_gametes_map_female': tup[CFG_GENOTYPE_TO_GAMETES_FEMALE],
        'genotype_to_gametes_map_male': tup[CFG_GENOTYPE_TO_GAMETES_MALE],
        'gametes_to_zygote_map': tup[CFG_GAMETES_TO_ZYGOTE],
        'female_viability': tup[CFG_FEMALE_VIABILITY],
        'male_viability': tup[CFG_MALE_VIABILITY],
    }


# ============================================================================
# 导出/导入（连接 Population ↔ 纯函数域）
# ============================================================================

def export_state(pop: 'AgeStructuredPopulation') -> Tuple[tuple, tuple, Optional[List]]:
    """导出 AgeStructuredPopulation → tuple/tuple（纯函数友好）。
    
    Args:
        pop: AgeStructuredPopulation 实例
    
    Returns:
        (state_arrays, config, history) 其中
        - state_arrays: dict with keys 'individual_count', 'sperm_storage', 'female_occupancy', 'tick'
        - config: dict with all parameters and static data
        - history: list of (tick, snapshot) tuples from pop._history, or None if empty
    """
    # 返回按固定顺序的 state tuple，便于在内核中使用索引访问
    state_arrays = (
        pop._population_state.individual_count.copy(),
        pop._population_state.sperm_storage.copy(),
        pop._population_state.female_occupancy.copy(),
        pop._tick,
    )
    
    # 返回扁平化的 config tuple，便于 numba-friendly 内核
    config = (
        pop._n_ages,
        len(pop._index_core.index_to_genotype),
        len(pop._index_core.index_to_haplo),
        len(pop._index_core.index_to_glab),
        tuple(pop._male_adult_ages),
        pop._adult_female_mating_rate,
        pop._sperm_displacement_rate,
        pop._offspring_per_female,
        pop._recruitment_size,
        pop._sex_ratio,
        pop._effective_population_size,
        pop._female_survival_rates.copy(),
        pop._male_survival_rates.copy(),
        pop._static_data.sexual_selection_fitness.copy(),
        pop._static_data.male_fecundity_fitness.copy(),
        pop._static_data.female_fecundity_fitness.copy(),
        pop._static_data.genotype_to_gametes_map[Sex.FEMALE.value].copy(),
        pop._static_data.genotype_to_gametes_map[Sex.MALE.value].copy(),
        pop._static_data.gametes_to_zygote_map.copy(),
        pop._static_data.female_viability_fitness.copy(),
        pop._static_data.male_viability_fitness.copy(),
    )
    
    # 导出历史（如果存在），history 已经与 BasePopulation 格式兼容（tick, snapshot_tuple）
    history = pop._history if pop._history else None

    return state_arrays, config, history


def import_state(
    pop: 'AgeStructuredPopulation',
    state_arrays: Dict[str, NDArray],
    history: Optional[List] = None
) -> None:
    """导入 numpy/dict → AgeStructuredPopulation。
    
    Args:
        pop: AgeStructuredPopulation 实例
        state_arrays: 来自 run_ticks/batch_ticks 的状态
        history: 可选的历史记录列表（如果提供，将导入到 pop._history）
    """
    # Accept either dict or tuple for state_arrays
    if isinstance(state_arrays, dict):
        ind = state_arrays['individual_count']
        sperm = state_arrays['sperm_storage']
        occ = state_arrays['female_occupancy']
        tick = state_arrays['tick']
    else:
        ind = state_arrays[STATE_INDIVIDUAL_COUNT]
        sperm = state_arrays[STATE_SPERM_STORAGE]
        occ = state_arrays[STATE_FEMALE_OCC]
        tick = state_arrays[STATE_TICK]

    pop._population_state.individual_count[:] = ind
    pop._population_state.sperm_storage[:] = sperm
    pop._population_state.female_occupancy[:] = occ
    pop._tick = tick
    
    # 导入历史（如果提供）
    if history is not None:
        pop._history = history


# ============================================================================
# 核心：单个 tick（纯函数，可选 Numba JIT）
# ============================================================================

# TODO: 不再接受 dict（为保证 Numba 兼容性）

@numba_switchable
def run_tick(
    state: Dict[str, NDArray],
    config: Dict[str, Any],
    seed: int,
    counter: int
) -> tuple:
    """执行一个 tick：繁殖 → 生存 → 衰老（纯函数）。
    
    Args:
        state: tuple 形式 (individual_count, sperm_storage, female_occupancy, tick)
        config: tuple 形式（由 config_dict_to_tuple 生成）
        seed: 基础随机数种子（整数）。与 counter 组合生成实际种子。
        counter: 调用计数器。用于确保每次调用产生不同的随机序列。
    
    Returns:
        更新后的 state tuple: (individual_count, sperm_storage, female_occupancy, tick)
    
    Note:
        使用 seed + counter 模式保证 Numba 兼容性。
        内部调用 apply_genetic_drift 时传递 seed 和 counter。
    """
    # Accept only tuple for state/config to ensure Numba compatibility

    # 工作副本（按 tuple 索引访问）
    ind_count = state[STATE_INDIVIDUAL_COUNT].copy()
    sperm_store = state[STATE_SPERM_STORAGE].copy()
    female_occ = state[STATE_FEMALE_OCC].copy()

    n_ages = config[CFG_N_AGES]
    n_gen = config[CFG_N_GENOTYPES]
    adult_ages = config[CFG_ADULT_AGES]
    
    # ===== 繁殖 =====
    # 1. 提取成年雄性计数
    male_counts = np.zeros(n_gen)
    for age in adult_ages:
        if age < n_ages:
            male_counts += ind_count[1, age, :]
    
    # if male_counts.sum() == 0:
    #     return
    
    # 2. 计算交配概率矩阵和新精子池
    mating_prob = alg.compute_mating_probability_matrix(
        config[CFG_SEXUAL_SELECTION], 
        male_counts, 
        n_gen
    )

    # genotype_to_gametes_map 已经是压缩形式 (n_sexes, n_genotypes, n_hg*n_glabs)
    s_new = alg.compute_new_sperm_pool(
        mating_prob,
        config[CFG_MALE_FECUNDITY],
        config[CFG_MEIOSIS_M], # genotype_to_gametes_map for male
        n_gen,
        config[CFG_N_HAPLOTYPES],
        config[CFG_N_GLABS],
    )
    
    # 3. 更新精子存储状态
    sperm_store, female_occ = alg.update_sperm_and_occupancy(
        sperm_store,
        female_occ,
        s_new,
        config[CFG_ADULT_FEMALE_MATING_RATE],
        config[CFG_SPERM_DISPLACEMENT_RATE],
        adult_ages[0] if adult_ages else 0,
        n_ages,
        n_gen,
        config[CFG_N_HAPLOTYPES],
        config[CFG_N_GLABS],
    )

    # 4. 提取雌性计数和使用存储的精子生成后代
    female_counts = ind_count[0, :, :]

    n_0_female, n_0_male = alg.generate_offspring_distribution(
        female_counts,
        sperm_store,
        config[CFG_FEMALE_FECUNDITY],
        config[CFG_MEIOSIS_F],
        config[CFG_HAPLO_TO_GENOTYPE],
        config[CFG_OFFSPRING_PER_FEMALE],
        adult_ages[0] if adult_ages else 0,
        n_ages,
        n_gen,
        config[CFG_N_HAPLOTYPES],
        config[CFG_N_GLABS],
        config[CFG_SEX_RATIO],
    )
    
    ind_count[0, 0, :] = n_0_female
    ind_count[1, 0, :] = n_0_male
    
    # ===== 生存 =====

    # 1. 应用年龄特异性生存率
    ind_count[:, :, :] = alg.apply_age_based_survival(
        (ind_count[0], ind_count[1]),
        config[CFG_FEMALE_SURVIVAL],
        config[CFG_MALE_SURVIVAL],
        n_gen,
        n_ages,
    )

    ind_count[:, :, :] = alg.apply_viability(
        (ind_count[0], ind_count[1]),
        config[CFG_FEMALE_VIABILITY],
        config[CFG_MALE_VIABILITY],
        n_gen,
        n_ages,
        target_age=1,
    )

    # 2. 应用遗传漂变
    if config[CFG_EFFECTIVE_POP_SIZE] > 0:
        age_0 = (ind_count[0, 0, :], ind_count[1, 0, :])
        drifted = alg.apply_genetic_drift(
            age_0, config[CFG_EFFECTIVE_POP_SIZE], n_gen, seed=seed, counter=counter)
        ind_count[0, 0, :] = drifted[0]
        ind_count[1, 0, :] = drifted[1]
    
    # 3. 应用环境容量限制
    ind_count[0, 0, :], ind_count[1, 0, :] = alg.recruit_juveniles(
        (ind_count[0, 0, :], ind_count[1, 0, :]),
        config[CFG_CARRYING_CAPACITY],
        n_gen
    )

    # ===== 衰老 =====
    for age in range(n_ages - 1, 0, -1):
        ind_count[:, age, :] = ind_count[:, age - 1, :]
        sperm_store[age, :, :] = sperm_store[age - 1, :, :]
        female_occ[age, :] = female_occ[age - 1, :]
    
    ind_count[:, 0, :] = 0  # 新生代已在繁殖阶段设置
    sperm_store[0, :, :] = 0  # 新生代精子库清空（不影响）
    female_occ[0, :] = 0  # 新生代雌性占位率清空（不影响）
    
    # 构造新的 state tuple
    new_state_t = (
        ind_count,
        sperm_store,
        female_occ,
        state[STATE_TICK] + 1,
    )

    # 返回 tuple（按索引固定顺序），便于上层批量/循环内复用并适配 Numba-friendly 接口
    return new_state_t


# ============================================================================
# 便利函数：循环运行和批量执行
# ============================================================================

def run(
    state_arrays: 'Union[Tuple[NDArray, NDArray, NDArray, int], Dict[str, NDArray]]',
    config: 'Union[Tuple[Any, ...], Dict[str, Any]]',
    n_ticks: int,
    seed: int = 42,
    callback: Optional[Callable] = None,
    record_history: bool = False
) -> Tuple[Dict[str, NDArray], Optional[List[Tuple[int, Tuple]]]]:
    """连续运行 n 个 tick，可选记录历史。
    
    Args:
        state_arrays: 初始状态（推荐为 tuple: (ind_count, sperm_storage, female_occupancy, tick)，
                      也接受旧的 dict 形式以兼容）
        config: 配置（推荐为按索引的 tuple，由 `export_state` 返回；也接受 dict 以兼容）
        n_ticks: 运行多少个 tick
        seed: 基础随机数种子。默认 42。
            内部使用 seed + counter 模式，counter 从 0 递增。
        callback: 可选，每 tick 后调用 callback(tick, state_arrays)
        record_history: 是否记录每个 tick 的状态快照（格式与 BasePopulation._history 兼容）
    
    Returns:
        (final_state, history) 其中
        - final_state: 最终状态
        - history: 若 record_history=True，返回 [(tick, (ind_count, sperm, fem_occ)), ...] 列表；否则返回 None
               快照格式与 BasePopulation.create_snapshot() 一致
    
    Example:
        >>> state, config, _ = export_state(pop)
        >>> final_state, history = run(state, config, 100, seed=42, record_history=True)
        >>> for tick, (ind_count, sperm, fem_occ) in history:
        ...     print(f"tick {tick}: {ind_count.sum():.0f}")
    """
    history = [] if record_history else None

    # normalize inputs to tuples
    state_t = state_dict_to_tuple(state_arrays) if isinstance(state_arrays, dict) else state_arrays
    config_t = config_dict_to_tuple(config) if isinstance(config, dict) else config

    # record initial state
    if record_history:
        snapshot = (
            state_t[STATE_INDIVIDUAL_COUNT].copy(),
            state_t[STATE_SPERM_STORAGE].copy(),
            state_t[STATE_FEMALE_OCC].copy(),
        )
        history.append((state_t[STATE_TICK], snapshot))

    # 使用 counter 确保每个 tick 产生不同的随机序列
    for tick_i in range(n_ticks):
        state_t = run_tick(state_t, config_t, seed=seed, counter=tick_i)

        if record_history:
            snapshot = (
                state_t[STATE_INDIVIDUAL_COUNT].copy(),
                state_t[STATE_SPERM_STORAGE].copy(),
                state_t[STATE_FEMALE_OCC].copy(),
            )
            history.append((state_t[STATE_TICK], snapshot))

        if callback is not None:
            callback(state_t[STATE_TICK], state_t)

    return state_t, history


def batch_ticks(
    initial_state: 'Union[Tuple[NDArray, NDArray, NDArray, int], Dict[str, NDArray]]',
    config: 'Union[Tuple[Any, ...], Dict[str, Any]]',
    n_particles: int,
    n_steps_per_particle: int,
    seed: int = 42,
    record_history: bool = False
) -> Tuple[List[Dict[str, NDArray]], Optional[List[List[Tuple[int, Tuple]]]]]:
    """批量执行多条独立的模拟（Monte-Carlo），可选记录历史。
    
    每条模拟从相同的初始状态开始，但用不同的随机种子。
    粒子 i 在步骤 step 使用的 counter = i * n_steps_per_particle + step，
    确保不同粒子和不同步骤产生不同的随机序列。
    
    Args:
        initial_state: 所有粒子的初始状态（共享）
        config: 配置（共享）
        n_particles: 粒子数量
        n_steps_per_particle: 每条粒子运行多少步
        seed: 基础随机数种子。默认 42。
            内部使用 seed + counter 模式，counter 根据粒子索引和步骤计算。
        record_history: 是否记录每条粒子的历史（格式与 BasePopulation._history 兼容）
    
    Returns:
        (particles, particle_histories) 其中
        - particles: 最终状态列表，长度 = n_particles
        - particle_histories: 若 record_history=True，返回列表的列表；
                           particle_histories[i] = [(tick, (ind_count, sperm, fem_occ)), ...] 
                           格式与 BasePopulation._history 一致；否则返回 None
    
    Example:
        >>> state, config, _ = export_state(pop)
        >>> 
        >>> # 蒙特卡洛采样：100条粒子，各运行50步，记录历史
        >>> particles, histories = batch_ticks(state, config, n_particles=100, 
        ...                                    n_steps_per_particle=50, seed=42,
        ...                                    record_history=True)
        >>> 
        >>> # 分析第一条粒子的历史
        >>> for tick, (ind_count, sperm, fem_occ) in histories[0]:
        ...     print(f"粒子0 tick {tick}: {ind_count.sum():.0f}")
    """
    particles = []
    particle_histories = [] if record_history else None

    # normalize config
    config_t = config_dict_to_tuple(config) if isinstance(config, dict) else config

    for particle_i in range(n_particles):
        # 每条粒子从初始状态的副本开始（使用 tuple 形式）
        state_t = state_dict_to_tuple(initial_state) if isinstance(initial_state, dict) else initial_state
        state_t = (
            state_t[STATE_INDIVIDUAL_COUNT].copy(),
            state_t[STATE_SPERM_STORAGE].copy(),
            state_t[STATE_FEMALE_OCC].copy(),
            state_t[STATE_TICK],
        )

        particle_history = [] if record_history else None

        # 运行指定步数
        # 每个粒子使用不同的 counter 范围确保独立随机序列
        for step in range(n_steps_per_particle):
            # counter = particle_i * n_steps_per_particle + step
            # 确保不同粒子和不同步骤产生不同的随机序列
            counter = particle_i * n_steps_per_particle + step
            state_t = run_tick(state_t, config_t, seed=seed, counter=counter)

            if record_history:
                snapshot = (
                    state_t[STATE_INDIVIDUAL_COUNT].copy(),
                    state_t[STATE_SPERM_STORAGE].copy(),
                    state_t[STATE_FEMALE_OCC].copy(),
                )
                particle_history.append((state_t[STATE_TICK], snapshot))

        particles.append(state_t)
        if record_history:
            particle_histories.append(particle_history)

    return particles, particle_histories


def batch_ticks_with_checkpoint(
    initial_state: 'Union[Tuple[NDArray, NDArray, NDArray, int], Dict[str, NDArray]]',
    config: 'Union[Tuple[Any, ...], Dict[str, Any]]',
    n_particles: int,
    n_steps_per_particle: int,
    checkpoint_interval: int,
    seed: int = 42
) -> Tuple[List[Dict[str, NDArray]], List[List[Dict[str, NDArray]]]]:
    """批量执行并保存中间检查点。
    
    Args:
        initial_state, config, n_particles, n_steps_per_particle: 同 batch_ticks
        checkpoint_interval: 每隔多少步保存一次检查点
        seed: 基础随机数种子。默认 42。内部使用 seed + counter 模式。
    
    Returns:
        (final_particles, checkpoints) 其中
        - final_particles: 最终状态列表
        - checkpoints: 列表的列表，checkpoints[particle_i][checkpoint_idx] = state at checkpoint
    """
    particles = []
    checkpoints = []

    # normalize config
    config_t = config_dict_to_tuple(config) if isinstance(config, dict) else config

    for particle_i in range(n_particles):
        state_t = state_dict_to_tuple(initial_state) if isinstance(initial_state, dict) else initial_state
        state_t = (
            state_t[STATE_INDIVIDUAL_COUNT].copy(),
            state_t[STATE_SPERM_STORAGE].copy(),
            state_t[STATE_FEMALE_OCC].copy(),
            state_t[STATE_TICK],
        )

        particle_checkpoints = []

        for step in range(n_steps_per_particle):
            # 计算 counter 确保不同粒子和步骤产生不同随机序列
            counter = particle_i * n_steps_per_particle + step
            state_t = run_tick(state_t, config_t, seed=seed, counter=counter)

            if (step + 1) % checkpoint_interval == 0:
                particle_checkpoints.append(
                    (
                        state_t[STATE_INDIVIDUAL_COUNT].copy(),
                        state_t[STATE_SPERM_STORAGE].copy(),
                        state_t[STATE_FEMALE_OCC].copy(),
                        state_t[STATE_TICK],
                    )
                )

        particles.append(state_t)
        checkpoints.append(particle_checkpoints)

    return particles, checkpoints


# ============================================================================
# 工具函数
# ============================================================================

def _multinomial_sample(
    expectations: NDArray,
    rng: np.random.Generator
) -> NDArray:
    """期望值 → 整数计数（采样）。"""
    total = float(expectations.sum())
    if total <= 0:
        return np.zeros_like(expectations, dtype=np.float64)
    
    probs = expectations / total
    n = int(np.round(total))
    sampled = rng.multinomial(n, probs)
    
    return sampled.astype(np.float64)


def extract_metric(
    particles: List[Dict[str, NDArray]],
    metric_fn: Callable
) -> List[float]:
    """从粒子批次中提取度量（e.g., 总个体数）。
    
    Args:
        particles: batch_ticks 返回的粒子列表
        metric_fn: 函数，接受单个粒子，返回标量（e.g., lambda p: p['individual_count'].sum()）
    
    Returns:
        度量值列表，长度 = len(particles)
    
    Example:
        >>> particles = batch_ticks(...)
        >>> total_counts = extract_metric(particles, lambda p: p['individual_count'].sum())
        >>> print(f"Mean: {np.mean(total_counts)}, Std: {np.std(total_counts)}")
    """
    return [metric_fn(p) for p in particles]


# =============================================================================
# 别名（兼容性）
# =============================================================================
# 保留 run_ticks 作为 run 的别名，以兼容旧代码
run_ticks = run