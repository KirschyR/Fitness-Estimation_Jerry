"""Non-Wright-Fisher (age-structured) population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.

Primary class:
    ``AgeStructuredPopulation``: An age-structured population model built on
    ``BasePopulation`` and ``PopulationState``.
"""

from typing import Dict, List, Optional, Union, Tuple, Set, Callable
import numpy as np
from utils.base_population import BasePopulation, Species, Genotype, Sex, HaploidGenome
from utils.population_state import PopulationState

# =============================================================================
# 新架构年龄结构种群模型（基于 BasePopulation）
# =============================================================================

class AgeStructuredPopulation(BasePopulation):
    """Age-structured population model (overlapping generations).

    This class implements an age-structured population built on top of
    ``BasePopulation`` and ``PopulationState``. It supports age-dependent
    survival, age-specific fecundity, juvenile recruitment modes, sperm
    storage, and hook-based extensibility.

    Example:
        >>> species = Species('Simple')
        >>> pop = AgeStructuredPopulation(
        ...     species,
        ...     initial_genotypes,
        ...     n_ages=8,
        ...     survival_rates=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
        ... )
        >>> pop.run(100)

        # Numba-accelerated access
        >>> counts = pop.state.get_counts_array()  # shape (n_types, n_ages)
    """
    
    def __init__(
        self,
        species: Species,
        name: str = "AgeStructuredPop",
        n_ages: int = 8,
        initial_population_distribution: Optional[Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]]] = None,
        female_survival_rates: Optional[List[float]] = None,
        male_survival_rates: Optional[List[float]] = None,
        female_adult_ages: Optional[List[int]] = None,
        male_adult_ages: Optional[List[int]] = None,
        offspring_per_female: float = 50.0,
        recruitment_size: Optional[int] = None,
        old_juvenile_carrying_capacity: Optional[int] = None,
        expected_num_adult_females: Optional[int] = None,
        juvenile_growth_mode: int = 2,
        low_density_growth_rate: float = 6.0,
        relative_competition_factor: float = 5.0,
        use_sperm_storage: bool = True,
        sperm_displacement_rate: float = 0.05,
        gamete_labels: Optional[List[str]] = None,
        adult_female_mating_rate: float = 1.0,
        effective_population_size: int = 0,
        sex_ratio: float = 0.5,
        n_glabs: int = 1,
        seed: Optional[int] = None,
        hooks: Optional[Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]] = None,
        gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None,
        zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
    ):
        """Initialize an age-structured population instance.

        Args:
            species: Species object describing genetic architecture.
            initial_population_distribution: Initial population mapping in the
                canonical format ``Dict[sex, Dict[Genotype|str, Union[List[int], Dict[int,int]]]]``.
                Sex keys must be "male" or "female". Genotypes may be provided
                as ``Genotype`` objects or strings parsable by
                ``species.get_genotype_from_str``. Counts may be dense lists by
                age or sparse dicts mapping age -> count.
            n_ages: Number of age classes.
            female_survival_rates: Per-age female survival specification (see
                ``_resolve_survival_rates`` for supported formats).
            male_survival_rates: Per-age male survival specification.
            female_adult_ages: List of ages considered adult for females.
            male_adult_ages: List of ages considered adult for males.
            offspring_per_female: Baseline number of eggs per adult female.
            recruitment_size: Fixed recruitment size.
            old_juvenile_carrying_capacity: Carrying capacity for old juveniles.
            expected_num_adult_females: Expected number of adult females at equilibrium.
            juvenile_growth_mode: Juvenile recruitment/growth mode (0/1/2).
            low_density_growth_rate: Parameter for low-density growth (mode 1).
            relative_competition_factor: Competition parameter for mode 1.
            use_sperm_storage: If True enables sperm storage mechanics.
            sperm_displacement_rate: Fractional displacement of stored sperm on remating.
            gamete_labels: Optional explicit list of gamete label strings.
            adult_female_mating_rate: Fraction of adult females that mate.
            effective_population_size: Effective population size for drift.
            sex_ratio: Proportion of females to males in the population.
            n_glabs: Number of gamete labels (legacy; overridden by ``gamete_labels`` if provided).
            rng: Optional NumPy random Generator.
            hooks: Optional hook registrations passed to ``BasePopulation``.
            gamete_modifiers: Optional list of gamete modifiers to register.
            zygote_modifiers: Optional list of zygote modifiers to register.
        """
        # 参数验证
        if n_ages <= 0:
            raise ValueError(f"n_ages must be positive, got {n_ages}")
        if offspring_per_female < 0:
            raise ValueError(f"offspring_per_female must be non-negative, got {offspring_per_female}")
        if sperm_displacement_rate < 0 or sperm_displacement_rate > 1:
            raise ValueError(f"sperm_displacement_rate must be in [0, 1], got {sperm_displacement_rate}")
        if effective_population_size < 0:
            raise ValueError(f"effective_population_size must be non-negative, got {effective_population_size}")
        
        # 解析新的 initial_population_distribution 格式
        if not initial_population_distribution:
            raise ValueError("initial_population_distribution must not be empty.")
        
        parsed_dist = self._parse_population_distribution(species, initial_population_distribution, n_ages)
        # 使用 Species 提供的枚举方法获取所有可能的基因型（而不是仅从初始分布提取）
        genotypes = species.get_all_genotypes()
        haploid_genotypes = self._get_all_possible_haploid_genotypes_from_genotypes(genotypes)
        
        super().__init__(species, name, hooks=hooks)

        # 初始化 IndexCore 并注册所有基因型
        from utils.index_core import IndexCore
        self._index_core = IndexCore()
        self._registry = self._index_core  # 保持向后兼容
        
        for genotype in genotypes:
            self._index_core.register_genotype(genotype)
        
        for hg in haploid_genotypes:
            self._index_core.register_haplogenotype(hg)
        if not initial_population_distribution:
            raise ValueError("initial_population_distribution must not be empty.")
        
        self._n_ages = n_ages

        # 默认参数
        self._female_adult_ages = female_adult_ages or [2, 3, 4, 5, 6, 7]
        self._male_adult_ages = male_adult_ages or [2, 3, 4]

        self._starting_adult_age = min(self._female_adult_ages)

        # 初始化 PopulationState 和 PopulationStaticData
        n_genotypes = len(genotypes)
        n_haplogenotypes = len(haploid_genotypes)
        # 处理 gamete labels：优先使用显式传入的字符串列表，否则使用 legacy n_glabs
        if gamete_labels is not None:
            if not isinstance(gamete_labels, (list, tuple)):
                raise TypeError("gamete_labels must be a list or tuple of strings")
            # register provided labels (strings)
            self.register_gamete_labels(gamete_labels)
            n_glabs = len(gamete_labels)
        else:
            # 兼容旧参数 n_glabs（如果存在），默认 1
            if not hasattr(self, '_n_glabs'):
                n_glabs = 1
            else:
                n_glabs = int(self._n_glabs)
            # ensure at least n_glabs labels exist
            self.register_gamete_labels(n_glabs)
        
                # 创建 PopulationStaticData
        from utils.population_static import PopulationStaticData
        self._static_data = PopulationStaticData.make_empty(
            n_genotypes=n_genotypes,
            n_hg=n_haplogenotypes,
            n_glabs=n_glabs
        )
        # 注册并排序传入的 high-level modifiers（可选传入），在构造 wrapper 之前设置
        # 允许传入的 modifiers 为 List[Tuple[hook_id, name, callable]]
        # 将其存入 BasePopulation 管理的列表以便 wrapper 能读取
        if gamete_modifiers:
            # overwrite any existing list
            self._gamete_modifiers = list(gamete_modifiers)
            self._gamete_modifiers.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
        if zygote_modifiers:
            self._zygote_modifiers = list(zygote_modifiers)
            self._zygote_modifiers.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))

        # 初始化配子表和合子表（应用 modifiers）
        # 使用 BasePopulation 提供的统一包装器将高层 modifiers 转为 tensor modifiers
        # print(f"Applying {len(gamete_modifiers)} gamete modifiers ...")
        gamete_tensor_mods, zygote_tensor_mods = self._build_modifier_wrappers(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=genotypes,
            n_glabs=n_glabs
        )

        self._static_data.initialize_gamete_map(
            diploid_genotypes=genotypes,
            haploid_genotypes=haploid_genotypes,
            gamete_modifiers=gamete_tensor_mods
        )
        self._static_data.initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=genotypes,
            zygote_modifiers=zygote_tensor_mods
        )
        
        # 验证成年年龄
        for ages, sex_name in [(self._female_adult_ages, "female"), (self._male_adult_ages, "male")]:
            if not all(0 <= age < n_ages for age in ages):
                raise ValueError(f"Invalid {sex_name} adult ages: {ages}")
        
        # 解析并缓存生存率（支持多种输入格式，详见 _resolve_survival_rates）
        _default_female = [1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0.0]
        _default_male =   [1.0, 1.0, 2/3, 1/2, 0.0, 0.0, 0.0, 0.0]

        self._female_survival_rates = self._resolve_survival_rates(
            female_survival_rates, n_ages, _default_female
        )
        self._male_survival_rates = self._resolve_survival_rates(
            male_survival_rates, n_ages, _default_male
        )
        
        # Fitness 参数（现在存储在 _static_data 中）
        # 初始化为默认值 1.0
        self._static_data.female_viability_fitness.fill(1.0)
        self._static_data.male_viability_fitness.fill(1.0)
        self._static_data.female_fecundity_fitness.fill(1.0)
        self._static_data.male_fecundity_fitness.fill(1.0)
        self._static_data.sexual_selection_fitness.fill(1.0)
        
        # 创建 PopulationState（明确指定 n_sexes=2，只使用雌性和雄性）
        self._population_state = PopulationState.make_empty(
            n_genotypes=n_genotypes,
            n_hg=n_haplogenotypes,
            n_ages=n_ages,
            n_sexes=2,  # 只使用 FEMALE 和 MALE
            n_glabs=n_glabs
        )

        # 初始化：将个体分布到年龄组（新统一格式）
        self._distribute_initial_population(parsed_dist)

        # 保存初始状态快照以便之后 reset() 能够恢复到初始分布
        self._initial_population_snapshot = (
            self._population_state.individual_count.copy(),
            self._population_state.sperm_storage.copy() if self._population_state.sperm_storage is not None else None,
            self._population_state.female_occupancy.copy() if self._population_state.female_occupancy is not None else None,
        )

        self._offspring_per_female = offspring_per_female
        self._juvenile_growth_mode = juvenile_growth_mode

        # 幼虫生长模式（old_juvenile_carrying_capacity 覆盖 recruitment_size）
        if old_juvenile_carrying_capacity is not None:
            self._recruitment_size = old_juvenile_carrying_capacity
        elif recruitment_size is not None:
            self._recruitment_size = recruitment_size
        else:
            # if not provided, set recruitment_size = 2 * age_2 total females
            self._recruitment_size = 2 * self._population_state.individual_count[0, self._starting_adult_age, :].sum()

        self._old_juvenile_carrying_capacity = self._recruitment_size

        # 预期成年雌性数量（默认 total females）
        if expected_num_adult_females is not None:
            self._expected_num_adult_females = expected_num_adult_females
        else:
            self._expected_num_adult_females = self._population_state.individual_count[0, self._starting_adult_age:self._n_ages, :].sum()
        
        # TODO: 现在 adult_ages List 其实没有实际作用，改为只传入一个最小成年年龄就好

        self._low_density_growth_rate = low_density_growth_rate
        self._relative_competition_factor = relative_competition_factor
        self._use_sperm_storage = use_sperm_storage
        self._sperm_displacement_rate = sperm_displacement_rate
        self._adult_female_mating_rate = adult_female_mating_rate
        self._effective_population_size = effective_population_size
        self._sex_ratio = sex_ratio
        self._seed = seed if seed is not None else np.random.randint(0, 2**32)
        # self._rng = rng if rng is not None else np.random.default_rng()
        
        # 触发初始化完成的 hook
        self.trigger_event("initialization")

    def _parse_population_distribution(
        self,
        species: Species,
        dist: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]],
        n_ages: int
    ) -> Dict[Sex, Dict[Genotype, Dict[int, int]]]:
        """Validate and parse the initial population distribution.

        Supported genotype keys:
            - ``Genotype`` instances
            - ``str`` values parsable by ``species.get_genotype_from_str``

        Returns:
            Dict[Sex, Dict[Genotype, Dict[int,int]]]: Parsed sparse mapping of
            counts by age for each genotype and sex.
        """
        parsed_dist = {}
        
        for sex_str, genotype_dist in dist.items():
            if sex_str not in ("male", "female"):
                raise ValueError(f"Sex must be 'male' or 'female', got '{sex_str}'")
            
            sex = Sex.MALE if sex_str == "male" else Sex.FEMALE
            parsed_dist[sex] = {}
            
            for genotype_key, age_data in genotype_dist.items():
                # 支持字符串和 Genotype 对象
                if isinstance(genotype_key, str):
                    genotype = species.get_genotype_from_str(genotype_key)
                elif isinstance(genotype_key, Genotype):
                    genotype = genotype_key
                else:
                    raise TypeError(f"Genotype key must be Genotype object or str, got {type(genotype_key)}")
                
                if genotype.species is not species:
                    raise ValueError("Genotype must belong to this species")
                
                # 转换为 Dict[int, int]（稀疏格式）
                if isinstance(age_data, list):
                    # List 格式：[count_age0, count_age1, ...]
                    age_dict = {age: count for age, count in enumerate(age_data) if count > 0}
                elif isinstance(age_data, dict):
                    # Dict 格式：{age: count, ...}
                    age_dict = {}
                    for age, count in age_data.items():
                        if not isinstance(age, int):
                            raise TypeError(f"Age must be int, got {type(age)}")
                        if age < 0 or age >= n_ages:
                            raise ValueError(f"Age {age} out of range [0, {n_ages})")
                        if count < 0:
                            raise ValueError(f"Count must be non-negative, got {count}")
                        if count > 0:
                            age_dict[age] = count
                else:
                    raise TypeError(f"Age data must be List or Dict, got {type(age_data)}")
                
                parsed_dist[sex][genotype] = age_dict
        
        return parsed_dist
    
    def _extract_genotypes_from_distribution(
        self,
        parsed_dist: Dict[Sex, Dict[Genotype, Dict[int, int]]]
    ) -> List[Genotype]:
        """Extract unique genotypes from a parsed distribution mapping.

        Returns:
            List[Genotype]: Sorted list of unique genotypes appearing in the
            provided distribution.
        """
        genotypes_set = set()
        for sex_dict in parsed_dist.values():
            for genotype in sex_dict.keys():
                genotypes_set.add(genotype)
        return sorted(list(genotypes_set), key=lambda gt: str(gt))

    def _distribute_initial_population(
        self,
        parsed_dist: Dict[Sex, Dict[Genotype, Dict[int, int]]]
    ) -> None:
        """Populate the internal ``PopulationState`` from a parsed distribution.

        Args:
            parsed_dist: Parsed mapping returned by ``_parse_population_distribution``.
        """
        for sex, genotype_dict in parsed_dist.items():
            sex_idx = int(sex.value)
            for genotype, age_dict in genotype_dict.items():
                genotype_idx = self._index_core.genotype_to_index[genotype]
                
                # 直接写入 PopulationState
                for age, count in age_dict.items():
                    if 0 <= age < self._n_ages:
                        self._population_state.individual_count[sex_idx, age, genotype_idx] += float(count)
    
    def _get_all_possible_haploid_genotypes_from_genotypes(self, genotypes: List[Genotype]) -> List[HaploidGenome]:
        """Return all unique haploid genomes found in a list of diploid genotypes."""
        haplotypes = set()
        for genotype in genotypes:
            haplotypes.add(genotype.maternal)
            haplotypes.add(genotype.paternal)
        return sorted(haplotypes, key=lambda h: str(h)) # TODO: 可能需要支持两性不同的配子
    
    def _compute_initial_age2_total(self) -> int:
        """Compute the total number of individuals at age 2 in the initial state."""
        if self._n_ages <= 2:
            return 0
        return self._population_state.individual_count[:, 2, :].sum()

    def _resolve_survival_rates(self, rates, n_ages: int, default: List[float]) -> np.ndarray:
        """Parse a flexible survival-rate specification into a NumPy array.

        Supported input formats (in order):
            - sequence (list/tuple/ndarray) of floats: truncated or padded as needed;
              a sentinel ``None`` at the end indicates fill-with-last-non-None.
            - dict mapping age (int) -> float: unspecified ages default to 1.0.
            - callable(age) -> float: invoked per age index.
            - scalar float: same value for all ages.

        Args:
            rates: User-provided specification (one of the supported formats).
            n_ages: Target length of the returned array.
            default: Default fallback list used when ``rates`` is None.

        Returns:
            np.ndarray: Array of length ``n_ages`` containing parsed survival rates.
        """
        # 默认
        if rates is None:
            return np.array(default[:n_ages], dtype=float)

        # 常数情况
        if isinstance(rates, (int, float)) and not isinstance(rates, bool):
            val = float(rates)
            if val < 0:
                raise ValueError("Survival rates must be non-negative")
            return np.full(n_ages, val, dtype=float)

        # 序列情况（list/tuple/ndarray）
        if isinstance(rates, (list, tuple, np.ndarray)):
            arr = np.array(rates, dtype=object)
            # 支持以 None 结尾表示用最后一个非 None 值填充
            if arr.size > 0 and arr[-1] is None:
                # 找到最后一个非 None
                non_none = None
                for v in arr[::-1]:
                    if v is not None:
                        non_none = float(v)
                        break
                if non_none is None:
                    # 全 None -> 使用默认
                    return np.array(default[:n_ages], dtype=float)
                vals = []
                for v in arr[:-1]:
                    if v is None:
                        raise TypeError("None only allowed as final sentinel in survival list")
                    vals.append(float(v))
                # pad with last non-none
                if len(vals) >= n_ages:
                    return np.array(vals[:n_ages], dtype=float)
                padded = np.empty(n_ages, dtype=float)
                padded[: len(vals)] = vals
                padded[len(vals) :] = non_none
                return padded

            # 普通序列：不足部分使用 0 填充
            arrf = np.array(arr, dtype=float)
            if arrf.size >= n_ages:
                out = arrf[:n_ages].astype(float)
            else:
                out = np.zeros(n_ages, dtype=float)
                out[: arrf.size] = arrf
            if (out < 0).any():
                raise ValueError("Survival rates must be non-negative")
            return out

        # 字典情况：缺省为 1.0
        if isinstance(rates, dict):
            out = np.ones(n_ages, dtype=float)
            for k, v in rates.items():
                if not isinstance(k, int):
                    raise TypeError("Age keys in survival dict must be int")
                if k < 0 or k >= n_ages:
                    raise ValueError(f"Age {k} out of range [0, {n_ages})")
                val = float(v)
                if val < 0:
                    raise ValueError("Survival rates must be non-negative")
                out[k] = val
            return out

        # 可调用情况：逐年龄调用 callable(age)
        if callable(rates):
            vals = []
            for age in range(n_ages):
                try:
                    v = rates(age)
                    vals.append(float(v))
                except Exception as e:
                    raise ValueError(f"Error calling survival rate function at age {age}: {e}")
            arrf = np.array(vals, dtype=float)
            if (arrf < 0).any():
                raise ValueError("Survival rates must be non-negative")
            return arrf

        raise TypeError("female_survival_rates / male_survival_rates must be None, sequence, dict, callable or numeric constant")

    def set_female_survival_rates(self, rates) -> None:
        """Set or update female per-age survival rates using the same formats as initialization."""
        self._female_survival_rates = self._resolve_survival_rates(rates, self._n_ages, self._female_survival_rates.tolist())

    def set_male_survival_rates(self, rates) -> None:
        """Set or update male per-age survival rates using the same formats as initialization."""
        self._male_survival_rates = self._resolve_survival_rates(rates, self._n_ages, self._male_survival_rates.tolist())
    
    @property
    def state(self) -> PopulationState:
        """Population state data container."""
        return self._population_state
    
    @property
    def n_ages(self) -> int:
        """Number of age classes in this population."""
        return self._n_ages
    
    @property
    def female_adult_ages(self) -> List[int]:
        """List of age indices considered adult for females."""
        return self._female_adult_ages.copy()  # 返回副本防止外部修改
    
    @property
    def male_adult_ages(self) -> List[int]:
        """List of age indices considered adult for males."""
        return self._male_adult_ages.copy()  # 返回副本防止外部修改
    
    def get_total_count(self) -> int:
        """Return the total number of individuals in the population."""
        return self._population_state.individual_count.sum()
    
    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        return self._population_state.individual_count[Sex.FEMALE.value, :, :].sum()
    
    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        return self._population_state.individual_count[Sex.MALE.value, :, :].sum()
    
    def get_adult_count(self, sex: str = 'both') -> int:
        """Return the number of adult individuals for the given sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'`` (aliases accepted).

        Returns:
            int: Total number of adults for the requested sex(es).
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")
        
        total = 0
        
        if sex in ('female', 'F', 'both'):
            for age in self._female_adult_ages:
                if age < self._n_ages:
                    total += self._population_state.individual_count[Sex.FEMALE.value, age, :].sum()
        
        if sex in ('male', 'M', 'both'):
            for age in self._male_adult_ages:
                if age < self._n_ages:
                    total += self._population_state.individual_count[Sex.MALE.value, age, :].sum()
        
        return total
    
    # ========================================================================
    # Fitness 管理（简化接口）
    # ========================================================================
    
    def set_viability(self, genotype: Genotype, value: float, sex: Optional[str] = None) -> None:
        """Set viability fitness for a genotype.

        Args:
            genotype: Target genotype instance.
            value: Non-negative viability multiplier.
            sex: Optional sex qualifier (``'female'`` or ``'male'``). If omitted
                the value is applied to both sexes.
        """
        if value < 0:
            raise ValueError(f"Viability must be non-negative, got {value}")
        
        if isinstance(genotype, str):
            genotype = self.species.get_genotype_from_str(genotype)

        genotype_idx = self._index_core.genotype_to_index[genotype]
        if sex is None:
            # 当 sex 未指定时，同时设置两个性别
            self._static_data.female_viability_fitness[genotype_idx] = value
            self._static_data.male_viability_fitness[genotype_idx] = value
        elif sex in ('female', 'F'):
            self._static_data.female_viability_fitness[genotype_idx] = value
        elif sex in ('male', 'M'):
            self._static_data.male_viability_fitness[genotype_idx] = value
        else:
            raise ValueError(f"sex must be 'female', 'male', or None, got '{sex}'")
    
    def set_viability_batch(self, values: Dict[Genotype, float], sex: Optional[str] = None) -> None:
        """Batch set viability values for multiple genotypes."""
        for genotype, value in values.items():
            self.set_viability(genotype, value, sex=sex)
    
    def set_fecundity(self, genotype: Genotype, value: float, sex: Optional[str] = None) -> None:
        """Set fecundity fitness for a genotype.

        Args and behavior mirror ``set_viability``.
        """
        if value < 0:
            raise ValueError(f"Fecundity must be non-negative, got {value}") 

        if isinstance(genotype, str):
            genotype = self.species.get_genotype_from_str(genotype)

        genotype_idx = self._index_core.genotype_to_index[genotype]
        if sex is None:
            # 当 sex 未指定时，同时设置两个性别
            self._static_data.female_fecundity_fitness[genotype_idx] = value
            self._static_data.male_fecundity_fitness[genotype_idx] = value
        elif sex in ('female', 'F'):
            self._static_data.female_fecundity_fitness[genotype_idx] = value
        elif sex in ('male', 'M'):
            self._static_data.male_fecundity_fitness[genotype_idx] = value
        else:
            raise ValueError(f"sex must be 'female', 'male', or None, got '{sex}'")
    
    def set_fecundity_batch(self, values: Dict[Genotype, float]) -> None:
        """Batch set fecundity values for multiple genotypes."""
        for genotype, value in values.items():
            self.set_fecundity(genotype, value)
    
    def set_sexual_selection(self, female_genotype: Genotype, male_genotype: Genotype, preference: float) -> None:
        """Set sexual selection preference (female genotype -> male genotype)."""
        if preference < 0:
            raise ValueError(f"Preference must be non-negative, got {preference}")
        
        if isinstance(female_genotype, str):
            female_genotype = self.species.get_genotype_from_str(female_genotype)
        if isinstance(male_genotype, str):
            male_genotype = self.species.get_genotype_from_str(male_genotype)

        f_idx = self._index_core.genotype_to_index[female_genotype]
        m_idx = self._index_core.genotype_to_index[male_genotype]
        self._static_data.sexual_selection_fitness[f_idx, m_idx] = preference
    
    def set_sexual_selection_batch(self, preferences: Dict[Tuple[Genotype, Genotype], float]) -> None:
        """Batch set sexual selection preferences for multiple genotype pairs."""
        for (f_gt, m_gt), pref in preferences.items():
            self.set_sexual_selection(f_gt, m_gt, pref)
    
    def _get_viability(self, genotype: Genotype, sex: Sex) -> float:
        """Internal helper: return viability for a genotype and sex."""
        genotype_idx = self._index_core.genotype_to_index[genotype]
        if sex == Sex.FEMALE:
            return self._static_data.female_viability_fitness[genotype_idx]
        elif sex == Sex.MALE:
            return self._static_data.male_viability_fitness[genotype_idx]
        else:
            raise ValueError(f"sex must be 'female' or 'male', got {sex}")
    
    def _get_fecundity(self, genotype: Genotype, sex: Sex) -> float:
        """Internal helper: return fecundity for a genotype and sex."""
        genotype_idx = self._index_core.genotype_to_index[genotype]
        if sex == Sex.FEMALE:
            return self._static_data.female_fecundity_fitness[genotype_idx]
        elif sex == Sex.MALE:
            return self._static_data.male_fecundity_fitness[genotype_idx]
        else:
            raise ValueError(f"sex must be 'female' or 'male', got {sex}")
    
    def _get_sexual_preference(self, female_genotype: Genotype, male_genotype: Genotype) -> float:
        """Internal helper: return sexual preference value for a genotype pair."""
        f_idx = self._index_core.genotype_to_index[female_genotype]
        m_idx = self._index_core.genotype_to_index[male_genotype]
        return self._static_data.sexual_selection_fitness[f_idx, m_idx]
    
    # ========================================================================
    # Hooks 系统
    # ========================================================================

    # 允许的 Hook 事件
    #
    #     Before simulation:  [initialization]
    #                                |
    #                                v
    #     For tick in T:    |-------------------------------------------------------------------------|
    #                       |     [first] -> [reproduction] --> [early] --> [survival] --> [late]     |
    #                       |        |<-------------------------------------------------------|       |
    #                       |-------------------------------------------------------------------------| 
    #                                |
    #                                v
    #     After simulation:      [finish]                       
    #
    
    # ========================================================================
    # 演化逻辑
    # ========================================================================
    
    def _step_reproduction(self) -> None:
        """Reproduction step: compute newborns and add them to age 0.

        Adult individuals produce offspring during this step.
        """
        self._reproduce()
    
    def _step_survival(self) -> None:
        """Survival step: apply survival rates and aging-related updates."""
        self._apply_survival()
    
    def _step_aging(self) -> None:
        """Aging step placeholder (aging handled during survival in this model)."""
        # 统一在每个 tick 末尾推进年龄与精子存储
        # 年龄推进：所有个体年龄 +1，年龄0 清空（等待下次繁殖填充）
        counts = self._population_state.individual_count
        aged = np.zeros_like(counts)
        aged[:, 1:, :] = counts[:, :-1, :]
        self._population_state.individual_count[:] = aged

        # 精子存储与占位率也在此推进（如果启用）
        if self._use_sperm_storage and self._n_ages > 1:
            S = self._population_state.sperm_storage
            aged_s = np.zeros_like(S)
            aged_s[1:, :, :] = S[:-1, :, :]
            self._population_state.sperm_storage[:] = aged_s

            Q = self._population_state.female_occupancy
            aged_q = np.zeros_like(Q)
            aged_q[1:, :] = Q[:-1, :]
            self._population_state.female_occupancy[:] = aged_q
    
    def _reproduce(self) -> None:
        """Perform reproduction: compute and add newborn individuals to age 0.

        By default this uses the sperm-storage mechanics when enabled.
        """
        from utils import algorithms
        
        n_genotypes = len(self._index_core.index_to_genotype)
        n_haplogenotypes = len(self._index_core.index_to_haplo)
        n_glabs = len(self._index_core.index_to_glab)
        
        # 1. 提取成年雄性计数
        male_counts = np.zeros(n_genotypes)
        for age in self._male_adult_ages:
            if age < self._n_ages:
                male_counts += self._population_state.individual_count[Sex.MALE.value, age, :]
        
        if male_counts.sum() == 0:
            return  # 没有成年雄性
        
        # 2. 计算交配概率矩阵和新精子池
        P = algorithms.compute_mating_probability_matrix(
            sexual_selection_matrix=self._static_data.sexual_selection_fitness,
            male_counts=male_counts,
            n_genotypes=n_genotypes
        )
        
        # genotype_to_gametes_map 已经是压缩形式 (n_sexes, n_genotypes, n_hg*n_glabs)
        male_gamete_matrix = self._static_data.genotype_to_gametes_map[Sex.MALE.value, :, :]
        S_new = algorithms.compute_new_sperm_pool(
            mating_probability_matrix=P,
            male_fecundity_fitness=self._static_data.male_fecundity_fitness,
            male_gamete_production_matrix=male_gamete_matrix,
            n_genotypes=n_genotypes,
            n_haplogenotypes=n_haplogenotypes,
            n_glabs=n_glabs
        )
        
        # 3. 更新精子存储状态（使用 sperm storage）
        if self._use_sperm_storage:
            # 直接使用压缩形式，不需要 reshape
            self._population_state.sperm_storage, self._population_state.female_occupancy = \
                algorithms.update_sperm_and_occupancy(
                    sperm_storage=self._population_state.sperm_storage,
                    female_occupancy=self._population_state.female_occupancy,
                    new_sperm_pool=S_new,  # 已经是压缩形式 (g, hl)
                    adult_female_mating_rate=self._adult_female_mating_rate,  # 成年雌性交配率
                    sperm_displacement_rate=self._sperm_displacement_rate,  # 精子置换率
                    adult_start_idx=min(self._female_adult_ages) if self._female_adult_ages else 0,
                    n_ages=self._n_ages,
                    n_genotypes=n_genotypes,
                    n_haplogenotypes=n_haplogenotypes,
                    n_glabs=n_glabs
                )
        else:
            # 不使用存储，直接替换（sperm_displacement_rate=1.0）
            # S_new 是压缩形式 (g, hl)，sperm_storage 是 (A, g, hl)
            for age in self._female_adult_ages:
                if age < self._n_ages:
                    self._population_state.sperm_storage[age, :, :] = S_new
                    self._population_state.female_occupancy[age, :] = 1.0
        
        # 4. 提取雌性计数和使用存储的精子生成后代
        female_counts = np.zeros((self._n_ages, n_genotypes))
        for age in range(self._n_ages):
            female_counts[age, :] = self._population_state.individual_count[Sex.FEMALE.value, age, :]
        
        adult_female_total = sum(female_counts[age, :].sum() for age in self._female_adult_ages if age < self._n_ages)
        if adult_female_total == 0:
            return  # 没有成年雌性
        
        # genotype_to_gametes_map 已经是压缩形式 (n_sexes, n_genotypes, n_hg*n_glabs)
        female_gamete_matrix = self._static_data.genotype_to_gametes_map[Sex.FEMALE.value, :, :]
        offspring = algorithms.generate_offspring_distribution(
            population_females=female_counts,
            sperm_storage=self._population_state.sperm_storage,
            fertility_f=self._static_data.female_fecundity_fitness,
            meiosis_f=female_gamete_matrix,
            haplo_to_genotype_map=self._static_data.gametes_to_zygote_map,
            average_eggs_per_wt_female=self._offspring_per_female,
            adult_start_idx=min(self._female_adult_ages) if self._female_adult_ages else 0,
            n_ages=self._n_ages,
            n_genotypes=n_genotypes,
            n_haplogenotypes=n_haplogenotypes,
            n_glabs=n_glabs,
            sex_ratio=self._sex_ratio,
        )

        # print(f"Total offspring generated: {offspring}")
        
        # 5. 将后代添加到年龄0
        n_0_female, n_0_male = offspring
        self._population_state.individual_count[Sex.FEMALE.value, 0, :] += n_0_female
        self._population_state.individual_count[Sex.MALE.value, 0, :] += n_0_male
    
    # ========================================================================
    # 旧方法已删除（_reproduce_with_sexual_selection, _get_modified_gametes,
    # _clear_gamete_cache, _compute_adult_gamete_frequencies）
    # 所有繁殖逻辑统一在 _reproduce() 中实现
    # ========================================================================
    
    def _apply_survival(self) -> None:
        """Apply survival, viability, drift, and juvenile growth.

        The pipeline performs:
            1. Age-specific survival and viability application.
            2. Genetic drift on newborns (if enabled).
            3. Juvenile growth / recruitment mode application.
            4. Update of internal population state (aging is applied later).
        """
        from utils import algorithms
        
        n_genotypes = len(self._index_core.index_to_genotype)
        
        # 1. 应用年龄特异性生存率（含 viability）
        population = (
            self._population_state.individual_count[Sex.FEMALE.value, :, :],
            self._population_state.individual_count[Sex.MALE.value, :, :]
        )
        # print(f"Total individuals before survival: {self._population_state.individual_count.sum()}")
        # print(f"Age 0 counts before survival: {self._population_state.individual_count[:, 0, :].sum()}")
        
        survived = algorithms.apply_age_based_survival(
            population=population,
            female_survival_rates=self._female_survival_rates,
            male_survival_rates=self._male_survival_rates,
            n_genotypes=n_genotypes,
            n_ages=self._n_ages
        )

        survived = algorithms.apply_viability(
            population=survived,
            female_viability_rates=self._static_data.female_viability_fitness,
            male_viability_rates=self._static_data.male_viability_fitness,
            n_genotypes=n_genotypes,
            n_ages=self._n_ages,
            target_age=1
        )
        # print(f"Total individuals survived: {survived.sum()}")
        # print(f"Age 0 counts after survival: {survived[:, 0, :].sum()}")

        # 2. 应用遗传漂变（如果启用） — 在新的流程中，漂变作用于 age0（新生儿）
        if self._effective_population_size > 0 and self._n_ages > 1:
            age0_counts = (survived[Sex.FEMALE.value, 0, :], survived[Sex.MALE.value, 0, :])
            drifted = algorithms.apply_genetic_drift(
                age_1_juvenile_counts=age0_counts,
                effective_population_size=self._effective_population_size,
                n_genotypes=n_genotypes,
                seed=self._seed,
                counter=self._tick
            )
            survived[Sex.FEMALE.value, 0, :] = drifted[Sex.FEMALE.value, :]
            survived[Sex.MALE.value, 0, :] = drifted[Sex.MALE.value, :]

        # 3. 应用幼虫生长模式（如果启用） — 操作 age0
        if self._juvenile_growth_mode > 0 and self._n_ages > 0:
            survived = self._apply_juvenile_growth(survived)

        # 4. 更新种群状态（不在此处进行老化；老化在 tick 末尾统一处理）
        self._population_state.individual_count[:] = survived
    
    def _apply_age_survival(self, counts: np.ndarray) -> np.ndarray:
        """Apply per-age survival rates and viability to counts.

        Args:
            counts: Array of shape ``(n_types, n_ages)`` representing current counts
                before aging.

        Returns:
            np.ndarray: Counts after applying survival and viability (still in
            the current age binning).
        """
        survived_counts = counts.copy()
        
        for individual_type in self._registry:
            idx = individual_type.index
            survival_rates = (self._female_survival_rates if individual_type.is_female 
                             else self._male_survival_rates)
            
            # 获取该基因型的 viability
            viability = self._get_viability(individual_type.genotype, individual_type.sex)
            
            for age in range(self._n_ages):
                if age < len(survival_rates):
                    base_survival = survival_rates[age]
                else:
                    base_survival = 0.0
                combined_survival = base_survival * viability
                survived_counts[idx, age] = counts[idx, age] * combined_survival
        
        return survived_counts
    
    def _age_individuals(self, counts: np.ndarray) -> np.ndarray:
        """Advance ages by one year and clear age-0 counts.

        This should be called after survival, drift, and juvenile growth have
        been applied.

        Args:
            counts: Array of shape ``(n_types, n_ages)`` representing counts
                before aging.

        Returns:
            np.ndarray: Aged counts with age-0 cleared.
        """
        aged_counts = np.zeros_like(counts)
        aged_counts[:, 1:] = counts[:, :-1]  # 年龄 0→1, 1→2, ..., n-2→n-1
        # aged_counts[:, 0] 自动为 0（新生儿已被移到年龄 1）
        return aged_counts
    
    def _apply_genetic_drift(self, aged_counts: np.ndarray) -> np.ndarray:
        """Apply genetic drift sampling to newborns (age 0) using multinomial draws.

        This routine should be invoked after survival but before aging. It
        samples female and male newborns separately using half of ``N_e`` each
        (where ``N_e`` is the configured effective population size).

        Args:
            aged_counts: Array of shape ``(n_types, n_ages)`` representing counts
                prior to drift.

        Returns:
            np.ndarray: Counts after applying genetic drift to age-0 individuals.
        """
        new_counts = aged_counts.copy()
        sex_indices = self._registry.get_sex_indices()
        n_registered = len(sex_indices)
        
        if n_registered == 0:
            return new_counts
        
        # 提取年龄 0 的计数（新生儿）进行漂变
        age0_counts = aged_counts[:n_registered, 0]
        
        # 分别处理雌性和雄性
        female_mask = sex_indices == 0
        male_mask = sex_indices == 1
        
        half_ne = self._effective_population_size // 2
        
        # 雌性多项式抽样
        if half_ne > 0:
            female_counts = age0_counts[female_mask]
            total_females = np.sum(female_counts)
            
            if total_females > 0:
                try:
                    probs_females = female_counts / total_females
                    sampled_females = self._rng.multinomial(half_ne, probs_females)
                    # 缩放回原总数
                    new_female_counts = (sampled_females / half_ne) * total_females
                    new_counts[female_mask, 0] = new_female_counts
                except ValueError as e:
                    # 抽样失败时保持原值（可选：添加警告日志）
                    print(f"Warning: Genetic drift sampling failed for females: {e}")
        
        # 雄性多项式抽样
        if half_ne > 0:
            male_counts = age0_counts[male_mask]
            total_males = np.sum(male_counts)
            
            if total_males > 0:
                try:
                    probs_males = male_counts / total_males
                    sampled_males = self._rng.multinomial(half_ne, probs_males)
                    # 缩放回原总数
                    new_male_counts = (sampled_males / half_ne) * total_males
                    new_counts[male_mask, 0] = new_male_counts
                except ValueError as e:
                    print(f"Warning: Genetic drift sampling failed for males: {e}")
        
        return new_counts
    
    def _apply_juvenile_growth(self, aged_counts: np.ndarray) -> np.ndarray:
        """Apply juvenile growth / recruitment to age-0 individuals.

        Modes supported:
            - Mode 0: No additional recruitment handling.
            - Mode 1: Density-dependent competition (Beverton-Holt style).
            - Mode 2: Fixed-ratio recruitment.

        Args:
            aged_counts: Array with shape ``(n_sexes, n_ages, n_genotypes)``.

        Returns:
            np.ndarray: Counts after juvenile recruitment.
        """
        # print(f"Applying juvenile growth mode {self._juvenile_growth_mode}")
        if self._juvenile_growth_mode == 0:
            return aged_counts
        
        from utils import algorithms
        n_genotypes = len(self._index_core.index_to_genotype)
        
        # 使用算法库中的幼虫招募函数
        """
        def recruit_juveniles(
            age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
            carrying_capacity: float,
            n_genotypes: int
        ) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
        """
        # print(f"Applying juvenile growth mode {self._juvenile_growth_mode} with carrying capacity {self._initial_age2_wildtype_total}")
        # print(f"Age 0 juvenile counts female before recruitment: {aged_counts[0, 0, :].sum()}")

        if self._juvenile_growth_mode == 2:
            aged_counts[0, 0, :], aged_counts[1, 0, :] = algorithms.recruit_juveniles(
                age_0_juvenile_counts=(aged_counts[0, 0, :], aged_counts[1, 0, :]),
                carrying_capacity=self._old_juvenile_carrying_capacity,
                n_genotypes=n_genotypes,
            )
        elif self._juvenile_growth_mode == 1:
            aged_counts[0, 0, :], aged_counts[1, 0, :] = algorithms.recruit_juveniles_logistic(
                age_0_juvenile_counts=(aged_counts[0, 0, :], aged_counts[1, 0, :]),
                age_1_juvenile_counts=(aged_counts[0, 1, :], aged_counts[1, 1, :]),
                age_1_carrying_capacity=self._old_juvenile_carrying_capacity,
                expected_num_adult_females=self._expected_num_adult_females,
                eggs_per_wt_female=self._offspring_per_female,
                relative_competition_ability=self._relative_competition_factor,
                low_density_growth_rate=self._low_density_growth_rate,
                n_genotypes=n_genotypes,
            )

        return aged_counts
    
    # 旧方法已删除：_apply_density_dependent_competition, _apply_fixed_ratio_recruitment
    # 现在统一使用 algorithms.recruit_juveniles
    
    def compute_allele_frequencies(self) -> Dict[str, float]:
        """Compute allele frequencies across the entire population.

        Returns:
            Dict[str, float]: Mapping from allele name to frequency in the
            population (based on allele counts aggregated over sexes and ages).
        """
        # 初始化等位基因频率
        allele_freqs = {}
        allele_counts = {}
        
        # 收集所有等位基因
        for chrom in self.species.chromosomes:
            for locus in chrom.loci:
                for gene in locus.alleles:
                    allele_freqs[gene.name] = 0.0
                    allele_counts[gene.name] = 0
        
        total_alleles = 0
        
        # 统计每个基因型的个体数（使用索引访问）
        for genotype_idx, genotype in enumerate(self._index_core.index_to_genotype):
            # 所有性别和年龄的总计数
            total_count = self._population_state.individual_count[:, :, genotype_idx].sum()
            
            if total_count == 0:
                continue
            
            # 统计等位基因
            for chrom in self.species.chromosomes:
                for locus in chrom.loci:
                    mat_gene, pat_gene = genotype.get_alleles_at_locus(locus)
                    
                    for gene in [mat_gene, pat_gene]:
                        if gene is not None:
                            allele_counts[gene.name] = allele_counts.get(gene.name, 0) + total_count
                            total_alleles += total_count
        
        # 计算频率
        if total_alleles > 0:
            for allele_name in allele_freqs.keys():
                allele_freqs[allele_name] = allele_counts.get(allele_name, 0) / total_alleles
        
        return allele_freqs
    
    def get_age_distribution(self, sex: str = 'both') -> np.ndarray:
        """Return the age distribution for the requested sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'``.

        Returns:
            np.ndarray: Age distribution array with shape ``(n_ages,)``.
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")
        
        # 直接从 PopulationState 访问
        if sex in ('female', 'F'):
            return self._population_state.individual_count[Sex.FEMALE.value, :, :].sum(axis=1)
        elif sex in ('male', 'M'):
            return self._population_state.individual_count[Sex.MALE.value, :, :].sum(axis=1)
        else:
            return self._population_state.individual_count.sum(axis=(0, 2))
    
    def get_genotype_count(self, genotype: Genotype) -> Tuple[int, int]:
        """Return total counts for a genotype as (female_count, male_count).

        Args:
            genotype: Target genotype instance.

        Returns:
            Tuple[int,int]: ``(female_count, male_count)`` across all ages.
        """
        genotype_idx = self._index_core.genotype_to_index[genotype]
        female_count = self._population_state.individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        male_count = self._population_state.individual_count[Sex.MALE.value, :, genotype_idx].sum()
        return (female_count, male_count)
    
    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Return the set of genotypes currently present in the population."""
        present = set()
        for genotype_idx, genotype in enumerate(self._index_core.index_to_genotype):
            total_count = self._population_state.individual_count[:, :, genotype_idx].sum()
            if total_count > 0:
                present.add(genotype)
        return present
    
    def __repr__(self) -> str:
        """Return a compact string representation of the population."""
        return (f"AgeStructuredPopulation(name='{self.name}', n_ages={self.n_ages}, "
                f"total_count={self.get_total_count()}, "
                f"adult_females={self.get_adult_count('female')}, "
                f"adult_males={self.get_adult_count('male')})")