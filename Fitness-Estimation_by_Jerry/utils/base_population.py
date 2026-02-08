"""Base population model helpers and abstractions.

This module provides the abstract base class and utilities for population
models (Wright-Fisher, age-structured non-Wright-Fisher, and related
architectures). The base class defines common interfaces, hook management,
modifier registration, and helpers that are implemented by concrete
population classes.

Design goals:
- Provide a user-friendly high-level API using Python objects (e.g. ``IndividualType``, ``Genotype``).
- Store internal state in NumPy arrays for compatibility with Numba acceleration.
- Separate logical indexing from storage via an index mapping layer.

Docstring style: Google style (Args, Returns, Raises, Example).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Callable, Any, FrozenSet, Union, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from utils.genetic_structures import *
from utils.genetic_entities import *
from utils.index_core import IndexCore
from utils.type_def import *
from utils.population_state import PopulationState
from utils.population_static import PopulationStaticData
from utils.modifiers import GameteModifier, ZygoteModifier

class BasePopulation(ABC):
    """Abstract base class for population models.

    The base class unifies common behavior for different population model
    implementations (for example, Wright-Fisher and age-structured
    non-Wright-Fisher models). It manages the species/genetic architecture,
    indexing, hook registration, and modifier pipelines.

    Core components:
        - ``species``: Genetic architecture descriptor.
        - ``registry``: ``IndexCore`` instance for managing genotype/haplotype indices.
        - ``state``: Abstract property implemented by subclasses (``PopulationState`` or
          age-structured variants).
        - ``_hooks``: Event hook registry mapping event names to ordered hook lists.
    """
    
    # 允许的 Hook 事件（子类可扩展此列表）
    ALLOWED_EVENTS = [
        "initialization",
        "first",
        "reproduction",
        "early",
        "survival",
        "late",
        "finish",
    ]
    
    def __init__(
        self,
        species: Species,
        name: str = "Population",
        hooks: Optional[Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]] = None
    ):
        """Initialize the base population.

        Args:
            species: Genetic architecture specifying chromosomes, loci, and alleles.
            name: Optional population name (default: "Population").
            hooks: Optional mapping of event names to hook registrations. Each
                entry should be a sequence of tuples in the form ``(func,)``,
                ``(func, hook_name)``, or ``(func, hook_name, hook_id)``. Hooks
                provided here will be registered during initialization.

        Note:
            Subclasses should initialize concrete ``state`` containers after
            calling ``super().__init__()``.
        """
        if not isinstance(species, Species):
            raise TypeError("species must be a Species instance.")
        
        self._species = species
        self._name = name
        self._tick = 0
        
        # 用 IndexCore 替代原来的 IndividualTypeRegistry
        self._index_core = IndexCore()
        # 保留 `_registry` 作为向后兼容别名，后续会改为直接使用 `index_core`
        self._registry = self._index_core
        
        # 演化历史：(tick, state_copy) 对的列表
        self._history: List[Tuple[int, 'PopulationState']] = []
        
        # Hooks 系统：事件名 -> [(hook_id, hook_name, hook_func), ...]
        self._hooks: Dict[str, List[Tuple[int, Optional[str], Callable]]] = {
            event: [] for event in self.ALLOWED_EVENTS
        }

        # 统一的配子修饰器列表
        self._gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]] = []

        # 统一的合子修饰器列表
        self._zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]] = []

        # 静态数据容器
        self._static_data: Optional[PopulationStaticData] = None

        # PopulationState 容器
        self._population_state: Optional[PopulationState] = None

        # 演化状态：是否已完成（finish）
        self._finished = False
        
        # 防止递归调用的标志
        self._running = False
        
        # 注册 hooks（在子类初始化之前）
        if hooks:
            for event_name, hooks_list in hooks.items():
                for hook_info in hooks_list:
                    if len(hook_info) == 1:
                        func = hook_info[0]
                        hook_name = None
                        hook_id = None
                    elif len(hook_info) == 2:
                        func, hook_name = hook_info
                        hook_id = None
                    else:
                        func, hook_name, hook_id = hook_info
                    
                    self.set_hook(event_name, func, hook_id=hook_id, hook_name=hook_name)
    
    # ========================================================================
    # 基础属性
    # ========================================================================
    
    @property
    def species(self) -> Species:
        """The species/genetic architecture for this population."""
        return self._species
    
    @property
    def name(self) -> str:
        """The human-readable name of the population."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = value
    
    @property
    def tick(self) -> int:
        """The current simulation tick or generation index."""
        return self._tick
    
    @tick.setter
    def tick(self, value: int) -> None:
        self._tick = value
    
    @property
    def registry(self) -> IndexCore:
        """IndexCore instance managing genotype, haplotype, and label indices."""
        return self._registry
    
    @property
    def state(self) -> PopulationState:
        """Return the current population state container.

        Returns:
            PopulationState: The current state object used by the population.
        """
        return self._population_state
    
    @property
    def history(self) -> List[Tuple[int, 'PopulationState']]:
        """A list of recorded historical states as ``(tick, state_copy)`` tuples."""
        return list(self._history)
    
    # ========================================================================
    # Modifier 管理
    # ========================================================================
    
    def add_gamete_modifier(
        self, 
        modifier: GameteModifier, 
        name: Optional[str] = None, 
        hook_id: Optional[int] = None
    ) -> None:
        """Register a gamete-level modifier.

        Args:
            modifier: A ``GameteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            hook_id: Optional numeric priority used for ordering.
        """
        self._gamete_modifiers.append((hook_id, name, modifier))
    
    def add_zygote_modifier(
        self, 
        modifier: ZygoteModifier, 
        name: Optional[str] = None, 
        hook_id: Optional[int] = None
    ) -> None:
        """Register a zygote-level modifier.

        Args:
            modifier: A ``ZygoteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            hook_id: Optional numeric priority used for ordering.
        """
        self._zygote_modifiers.append((hook_id, name, modifier))

    # 确保 set_zygote_modifier 方法与 ZygoteModifier 定义一致
    def set_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None
    ) -> None:
        """Register a zygote modifier with an optional priority.

        Args:
            modifier: A ``ZygoteModifier`` instance or callable.
            hook_id: Numeric priority (lower values execute earlier). If omitted
                an id will be auto-assigned.
            hook_name: Optional name for debugging.
        """
        if not callable(modifier):
            raise TypeError("Zygote modifier must be callable")
        
        # 自动分配 hook_id
        if hook_id is None:
            if self._zygote_modifiers:
                hook_id = max(hid for hid, _, _ in self._zygote_modifiers) + 1
            else:
                hook_id = 0
        
        # 添加并排序
        self._zygote_modifiers.append((hook_id, hook_name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])

    def set_gamete_modifier(
        self,
        modifier: GameteModifier,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None
    ) -> None:
        """Register a gamete modifier with optional priority and name."""
        if not callable(modifier):
            raise TypeError("Gamete modifier must be callable")
        
        # 自动分配hook_id
        if hook_id is None:
            hook_id = max((hid for hid, _, _ in self._gamete_modifiers), default=0) + 1
        
        # 添加并排序
        self._gamete_modifiers.append((hook_id, hook_name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])

    def initialize_static_data(self) -> None:
        """Initialize static lookup tensors used by the population model.

        This prepares precomputed maps such as ``gametes_to_zygote_map`` and
        ``genotype_to_gametes_map`` and wraps high-level modifiers so they can
        be applied at tensor-level during simulation steps.
        """
        from utils.population_static import PopulationStaticData
        from utils.genetic_entities import HaploidGenotype, Genotype
        from utils.type_def import Sex
        
        # 获取所有可能的单倍型和二倍型
        haploid_genotypes = self._get_all_possible_haploid_genotypes()
        diploid_genotypes = self._get_all_possible_diploid_genotypes()
        
        n_hg = len(haploid_genotypes)
        n_genotypes = len(diploid_genotypes)
        n_glabs = 1  # 根据实际情况设置
        
        # 创建静态数据容器
        self._static_data = PopulationStaticData.make_empty(
            n_genotypes=n_genotypes,
            n_hg=n_hg,
            n_glabs=n_glabs
        )
        
        # 使用统一的 wrapper 生成器将高层 modifier 转换为 tensor-level modifier
        gamete_modifier_funcs, zygote_modifier_funcs = self._build_modifier_wrappers(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs
        )

        # 初始化 gametes_to_zygote_map 与 genotype_to_gametes_map
        self._static_data.initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            zygote_modifiers=zygote_modifier_funcs
        )

        self._static_data.initialize_gamete_map(
            diploid_genotypes=diploid_genotypes,
            haploid_genotypes=haploid_genotypes,
            gamete_modifiers=gamete_modifier_funcs
        )

    def register_gamete_labels(self, labels: 'Optional[Sequence[str]] | int') -> None:
        """
        Register gamete labels in the IndexCore.

        Args:
            labels: Either an integer (number of labels, legacy behavior) or a
                sequence of string labels. If an integer `k` is provided, labels
                named `'glab0'..'glab{k-1}'` will be registered for any missing
                labels. If a sequence is provided, each element will be
                registered as a gamete label (strings preferred).
        """
        if not hasattr(self, "_index_core") or self._index_core is None:
            raise RuntimeError("IndexCore not initialized; cannot register gamete labels")

        existing = self._index_core.num_gamete_labels()

        # Legacy integer API: register that many labels if needed
        if isinstance(labels, int):
            n_glabs = int(labels)
            for i in range(existing, n_glabs):
                self._index_core.register_gamete_label(f"glab{i}")
            return

        # Sequence of label strings
        try:
            seq = list(labels) if labels is not None else []
        except Exception:
            raise TypeError("labels must be int or sequence of strings")

        # Register each string label if not already present
        for lab in seq:
            if lab not in self._index_core.glab_to_index:
                self._index_core.register_gamete_label(str(lab))

    # ------------------------------------------------------------------
    # Helper routines to simplify modifier key/value parsing. These were
    # extracted from the inline closures in _build_modifier_wrappers to
    # reduce cognitive complexity and improve testability.
    # ------------------------------------------------------------------
    def _resolve_hg_glab(self, haploid_genotypes: List[HaploidGenotype], part: Any, n_glabs: int, strict: bool = True) -> Tuple[int, int]:
        """Resolve a flexible haploid/genotype+glab part into numeric indices.

        Args:
            haploid_genotypes: list of HaploidGenotype objects.
            part: flexible selector (HaploidGenotype, int, str, or tuple).
            n_glabs: number of gamete labels.
            strict: pass-through to IndexCore resolver.

        Returns:
            (hg_idx, glab_idx)
        """
        return self._index_core.resolve_hg_glab_part(haploid_genotypes, part, n_glabs, strict=strict)

    def _parse_zygote_key(self, key: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int) -> Tuple[int, int]:
        """Parse modifier key for zygote wrappers into compressed coords (c1,c2).

        Accepts either already-compressed tuple of ints, or a pair of flexible
        'part' selectors which will be resolved and compressed.
        """
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(x, int) for x in key):
            return key[0], key[1]
        # otherwise expect a pair of parts that can be resolved
        part1, part2 = key
        idx_hg1, glab1 = self._resolve_hg_glab(haploid_genotypes, part1, n_glabs, strict=True)
        idx_hg2, glab2 = self._resolve_hg_glab(haploid_genotypes, part2, n_glabs, strict=True)
        c1 = self._index_core.compress_hg_glab(idx_hg1, glab1, n_glabs)
        c2 = self._index_core.compress_hg_glab(idx_hg2, glab2, n_glabs)
        return c1, c2

    def _normalize_zygote_val(self, val: Any, diploid_genotypes: List[Genotype]) -> Dict[int, float]:
        """Normalize zygote replacement `val` into a mapping idx->prob.

        Supported val forms:
        - int or Genotype-like -> {idx:1.0}
        - (idx_or_genotype, prob) -> {idx:prob}
        - dict mapping idx_or_genotype -> prob -> normalized dict
        """
        mapping: Dict[int, float] = {}
        # single tuple (idx_or_genotype, prob)
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], (int, float)):
            idx_candidate, prob = val
            if isinstance(idx_candidate, int):
                idx = int(idx_candidate)
            else:
                idx = self._index_core.resolve_genotype_index(diploid_genotypes, idx_candidate, strict=True)
            mapping[int(idx)] = float(prob)
            return mapping

        # distribution dict
        if isinstance(val, dict):
            for idx_candidate, prob in val.items():
                if not isinstance(idx_candidate, int):
                    idx_candidate = self._index_core.resolve_genotype_index(diploid_genotypes, idx_candidate, strict=True)
                mapping[int(idx_candidate)] = float(prob)
            return mapping

        # single genotype replacement
        idx = self._index_core.resolve_genotype_index(diploid_genotypes, val, strict=True)
        mapping[int(idx)] = 1.0
        return mapping

    def _write_zygote_mapping(self, modified: np.ndarray, c1: int, c2: int, mapping: Dict[int, float]) -> None:
        """Apply mapping (idx->prob) to the compressed zygote slice.

        This zeros the existing slice and writes the provided probabilities.
        """
        modified[c1, c2, :] = 0.0
        for idx_mod, prob in mapping.items():
            modified[c1, c2, int(idx_mod)] = float(prob)

    def _resolve_sex_name(self, key: str) -> Optional[int]:
        """Normalize string sex names to sex index (0=female,1=male).

        Returns None for unknown keys.
        """
        if not isinstance(key, str):
            return None
        k = key.lower()
        from utils.type_def import Sex
        if k in ("male", "m"):
            return int(Sex.MALE)
        if k in ("female", "f"):
            return int(Sex.FEMALE)
        return None

    def _apply_comp_map(self, modified: np.ndarray, sex_idx: int, gidx: int, comp_map: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int, n_hg_glabs: int) -> None:
        """Apply a comp_map (comp_key->freq) into the provided modified tensor slice.

        This helper resolves flexible comp_key forms via IndexCore.resolve_comp_idx
        and writes the frequency after bounds-checking.
        """
        # zero the slice first
        modified[sex_idx, gidx, :] = 0.0
        if not isinstance(comp_map, dict):
            return
        for comp_key, freq in comp_map.items():
            comp_idx = self._index_core.resolve_comp_idx(haploid_genotypes, n_glabs, comp_key, strict=False)
            if comp_idx is None:
                continue
            if not (0 <= comp_idx < n_hg_glabs):
                continue
            modified[sex_idx, gidx, comp_idx] = float(freq)

    def _build_modifier_wrappers(
        self,
        haploid_genotypes: List[HaploidGenotype],
        diploid_genotypes: List[Genotype],
        n_glabs: int = 1
    ) -> Tuple[List[Callable], List[Callable]]:
        """Wrap high-level gamete/zygote modifiers into tensor-level callables.

        High-level modifiers provided by users typically operate on domain
        objects (e.g., ``HaploidGenotype``). This helper builds wrappers that
        accept and return NumPy tensors so they can be applied efficiently
        at the tensor level inside the simulation engine.

        Returns:
            Tuple containing two lists: ``(gamete_modifier_funcs, zygote_modifier_funcs)``.
        """
        gamete_modifier_funcs = []
        zygote_modifier_funcs = []

        # 生成 zygote wrapper（严格 bulk 接口）
        # 要求 mod() -> Dict[(c1,c2) -> (idx_modified|Genotype|{idx:prob,...})]
        for _, _, mod in self._zygote_modifiers:
            def make_z(mod):
                def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
                    # create a writable copy so the original tensor isn't mutated
                    modified = tensor.copy()
                    # n_hg is the number of haploid genotype objects available
                    n_hg = len(haploid_genotypes)

                    # Support both call signatures: mod() or mod(self)
                    import inspect
                    sig = inspect.signature(mod)
                    if len(sig.parameters) == 0:
                        # user-provided modifier expects no arguments
                        bulk = mod()
                    else:
                        # user-provided modifier expects the population instance
                        bulk = mod(self)

                    # The modifier must return a dict mapping keys -> replacements
                    if not isinstance(bulk, dict):
                        raise TypeError("Zygote modifier must return a dict mapping keys to replacements")
                    # Iterate user-provided replacement rules, but delegate parsing
                    # and normalization to private helpers to keep this body concise.
                    for key, val in bulk.items():
                        # parse key into compressed coordinates
                        c1, c2 = self._parse_zygote_key(key, haploid_genotypes, n_glabs)
                        # normalize value into idx->prob mapping
                        mapping = self._normalize_zygote_val(val, diploid_genotypes)
                        # write mapping into the tensor
                        self._write_zygote_mapping(modified, c1, c2, mapping)

                    return modified
                return tensor_modifier
            zygote_modifier_funcs.append(make_z(mod))

        # 生成 gamete wrapper（严格 bulk 接口）
        # 要求 mod() -> Dict[(sex_idx, genotype_idx) -> {compressed_hg_glab_idx: freq, ...}]
        for _, _, mod in self._gamete_modifiers:
            def make_g(mod):
                def tensor_modifier(tensor: np.ndarray) -> np.ndarray:
                    # make a writable copy of the provided tensor so we don't mutate the input
                    modified = tensor.copy()
                    # tensor shape is (n_sexes, n_genotypes, n_hg_glabs)
                    n_sexes, n_genotypes, n_hg_glabs = modified.shape

                    # Support both mod() and mod(self) call signatures.
                    import inspect
                    sig = inspect.signature(mod)
                    if len(sig.parameters) == 0:
                        # user-defined modifier takes no args
                        bulk = mod()
                    else:
                        # user-defined modifier expects the population object
                        bulk = mod(self)

                    # Validate return type
                    if not isinstance(bulk, dict):
                        raise TypeError("Gamete modifier must return a dict mapping keys to compressed-index->freq dicts")

                    # Iterate over top-level mappings provided by the modifier
                    for key, val in bulk.items():
                        # Case A: top-level sex-name ('male'/'female')
                        sex_idx = self._resolve_sex_name(key) if isinstance(key, str) else None
                        if sex_idx is not None and isinstance(val, dict):
                            for gk, comp_map in val.items():
                                try:
                                    gidx = gk if isinstance(gk, int) else self._index_core.resolve_genotype_index(diploid_genotypes, gk, strict=True)
                                except KeyError:
                                    continue
                                if not (0 <= sex_idx < n_sexes and 0 <= gidx < n_genotypes):
                                    continue
                                self._apply_comp_map(modified, sex_idx, gidx, comp_map, haploid_genotypes, n_glabs, n_hg_glabs)
                            continue

                        # Case B: explicit (sex_idx, genotype_key) tuple
                        if isinstance(key, tuple) and len(key) == 2:
                            sex_idx, gk = key
                            gidx = gk if isinstance(gk, int) else self._index_core.resolve_genotype_index(diploid_genotypes, gk, strict=True)
                            if not (0 <= sex_idx < n_sexes and 0 <= gidx < n_genotypes):
                                continue
                            self._apply_comp_map(modified, sex_idx, gidx, val, haploid_genotypes, n_glabs, n_hg_glabs)
                            continue

                        # Case C: key is genotype_key applied to all sexes
                        try:
                            gidx = key if isinstance(key, int) else self._index_core.resolve_genotype_index(diploid_genotypes, key, strict=True)
                        except KeyError:
                            continue
                        if not isinstance(val, dict):
                            continue
                        for sex_idx in range(n_sexes):
                            self._apply_comp_map(modified, sex_idx, gidx, val, haploid_genotypes, n_glabs, n_hg_glabs)

                    # return the modified tensor ready to be written back
                    return modified
                return tensor_modifier
            gamete_modifier_funcs.append(make_g(mod))

        return gamete_modifier_funcs, zygote_modifier_funcs

    def _get_all_possible_haploid_genotypes(self) -> List[HaploidGenotype]:
        """
        获取所有可能的单倍型列表
        这是一个示例实现，需要根据实际情况扩展
        """
        # 简化实现，假设已经有方法可以获取所有可能的单倍型
        # 实际应用中，这可能需要通过枚举所有可能的等位基因组合来生成
        pass

    def _get_all_possible_diploid_genotypes(self) -> List[Genotype]:
        """
        获取所有可能的二倍型列表
        这是一个示例实现，需要根据实际情况扩展
        """
        # 简化实现，假设已经有方法可以获取所有可能的二倍型
        # 实际应用中，这可能需要通过枚举所有可能的单倍型组合来生成
        pass

    # ========================================================================
    # 核心方法
    # ========================================================================
    
    def step(self) -> 'BasePopulation':
        """
        执行一个演化步骤。
        
        标准流程：
        1. 检查是否已 finish
        2. 设置 _running 标志防止递归
        3. 触发 'first' hook
        4. 调用 _step_reproduction()
        5. 触发 'early' hook
        6. 调用 _step_survival()
        7. 触发 'late' hook
        8. 更新 tick
        9. 清除 _running 标志
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish 或正在运行中
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot step() after finish=True."
            )
        
        if self._running:
            raise RuntimeError(
                f"Population '{self.name}' is already running. "
                "Cannot call step()/run_tick()/run() recursively (e.g., from within a hook)."
            )
        
        try:
            self._running = True
            
            # first hook
            self.trigger_event("first")
            
            # 繁殖阶段
            self._step_reproduction()
            self.trigger_event("reproduction")
            
            # early hook
            self.trigger_event("early")
            
            # 生存阶段
            self._step_survival()
            self.trigger_event("survival")
            
            # late hook
            self.trigger_event("late")

            # update age
            self._step_aging()
            
            # 更新 tick
            self._tick += 1
            
        finally:
            self._running = False
        
        return self
        
    @abstractmethod
    def _step_reproduction(self) -> None:
        """
        繁殖阶段的内部实现。
        
        子类必须实现此方法来定义具体的繁殖逻辑。
        注意：此方法不应更新 tick。
        """
        pass
    
    @abstractmethod
    def _step_survival(self) -> None:
        """
        生存阶段的内部实现。
        
        子类必须实现此方法来定义具体的生存/选择逻辑。
        注意：此方法不应更新 tick。
        """
        pass

    @abstractmethod
    def _step_aging(self) -> None:
        """
        老化阶段的内部实现。
        
        子类必须实现此方法来定义具体的年龄逻辑。
        注意：此方法不应更新 tick。
        """
        pass
    
    def run_tick(self) -> 'BasePopulation':
        """
        run_tick 是 step() 的完全别名。
        
        两个方法严格等价，都执行相同的逻辑。
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish 或正在运行中
        
        Example:
            >>> pop.run_tick()  # 与 pop.step() 完全等价
        """
        return self.step()
    
    @abstractmethod
    def get_total_count(self) -> int:
        """返回种群总个体数"""
        pass
    
    @abstractmethod
    def get_female_count(self) -> int:
        """返回雌性总个体数"""
        pass
    
    @abstractmethod
    def get_male_count(self) -> int:
        """返回雄性总个体数"""
        pass
    
    # ========================================================================
    # 通用方法（可被子类继承或覆写）
    # ========================================================================
    
    @property
    def total_population_size(self) -> int:
        """种群总大小（get_total_count 的别名）"""
        return self.get_total_count()
    
    @property
    def total_females(self) -> int:
        """雌性总数（get_female_count 的别名）"""
        return self.get_female_count()
    
    @property
    def total_males(self) -> int:
        """雄性总数（get_male_count 的别名）"""
        return self.get_male_count()
    
    @property
    def sex_ratio(self) -> float:
        """性比（雌/雄），雄性为0时返回 np.inf"""
        males = self.get_male_count()
        return self.get_female_count() / males if males > 0 else np.inf
    
    @property
    def is_finished(self) -> bool:
        """检查种群是否已完成（finish=True）"""
        return self._finished
    
    def finish_simulation(self) -> None:
        """
        结束模拟，触发 'finish' 事件并锁定种群。
        
        此方法可以被 hooks 调用以提前结束模拟。
        调用后，种群将无法再运行 step()/run_tick()/run()。
        
        Raises:
            RuntimeError: 如果种群已经 finished
        
        Example:
            >>> def check_extinction(pop):
            ...     if pop.get_total_count() == 0:
            ...         print("Population extinct, finishing simulation.")
            ...         pop.finish_simulation()
            >>> pop.set_hook('late', check_extinction)
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has already finished."
            )
        
        self._finished = True
        self.trigger_event("finish")
    
    def run(
        self, 
        n_steps: int, 
        record_every: int = 1,
        finish: bool = False
    ) -> 'BasePopulation':
        """
        运行多步演化。
        
        Args:
            n_steps: 要运行的步数
            record_every: 每隔多少步记录一次快照（0 表示不记录）
            finish: 是否在运行完成后标记为 finished
                如果为 True，运行完成后会触发 'finish' 事件，
                并将种群标记为已完成，之后无法再运行 run_tick()
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish，无法继续运行
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot run() again after finish=True."
            )
        
        # Create a snapshot at the beginning if tick is 0
        if self.tick == 0:
            self.create_snapshot()
        
        for i in range(n_steps):
            self.step()
            if record_every > 0 and self.tick % record_every == 0:
                self.create_snapshot()
        
        if finish:
            self.finish_simulation()
        
        return self
    
    def create_snapshot(self) -> None:
        """
        创建当前种群状态的历史记录。
        
        将当前 tick 和 state 的副本保存到历史列表。
        """
        state_copy = (self.state.individual_count.copy(), 
                      self.state.sperm_storage.copy() if self.state.sperm_storage is not None else None,
                      self.state.female_occupancy.copy() if self.state.female_occupancy is not None else None)
        self._history.append((self.tick, state_copy))

    def reset(self) -> None:
        """Reset the population to its initial state.

        Behavior:
        - Reset `self._tick` to 0.
        - Clear the history list.
        - Clear the `finished` flag so the population may be run again.
        - If the instance provides an `_initial_population_snapshot` (tuple
          of arrays created by subclasses), restore it. Otherwise reallocate
          an empty `PopulationState` with the same array shapes.
        """
        # reset tick and flags
        self._tick = 0
        self._history = []
        self._finished = False

        # restore initial snapshot if subclass saved one
        if hasattr(self, '_initial_population_snapshot') and self._initial_population_snapshot is not None:
            ind_copy, sperm_copy, occ_copy = self._initial_population_snapshot
            if self._population_state is None:
                # allocate fresh state with inferred sizes
                from utils.population_state import PopulationState
                n_genotypes = len(self._index_core.index_to_genotype)
                n_hg = len(self._index_core.index_to_haplo)
                n_glabs = len(self._index_core.index_to_glab)
                # infer ages/sexes from saved arrays
                n_ages = None
                n_sexes = None
                if ind_copy is not None:
                    if ind_copy.ndim == 3:
                        n_sexes, n_ages, _ = ind_copy.shape
                    else:
                        n_sexes, _ = ind_copy.shape
                self._population_state = PopulationState.make_empty(
                    n_genotypes=n_genotypes,
                    n_hg=n_hg,
                    n_ages=n_ages,
                    n_sexes=n_sexes,
                    n_glabs=n_glabs
                )

            # copy arrays back into the active state
            if ind_copy is not None:
                self._population_state.individual_count[:] = ind_copy.copy()
            if hasattr(self._population_state, 'sperm_storage') and self._population_state.sperm_storage is not None:
                if sperm_copy is not None:
                    self._population_state.sperm_storage[:] = sperm_copy.copy()
                else:
                    self._population_state.sperm_storage[:] = 0
            if hasattr(self._population_state, 'female_occupancy') and self._population_state.female_occupancy is not None:
                if occ_copy is not None:
                    self._population_state.female_occupancy[:] = occ_copy.copy()
                else:
                    self._population_state.female_occupancy[:] = 0
            return

        # Fallback: reallocate an empty PopulationState preserving shapes
        if self._population_state is not None:
            from utils.population_state import PopulationState
            n_genotypes = len(self._index_core.index_to_genotype)
            n_hg = len(self._index_core.index_to_haplo)
            n_glabs = len(self._index_core.index_to_glab)

            # infer whether age-structured
            ind = self._population_state.individual_count
            if ind.ndim == 3:
                n_sexes, n_ages, _ = ind.shape
            else:
                n_sexes = ind.shape[0]
                n_ages = None

            self._population_state = PopulationState.make_empty(
                n_genotypes=n_genotypes,
                n_hg=n_hg,
                n_ages=n_ages,
                n_sexes=n_sexes,
                n_glabs=n_glabs
            )
    
    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        计算种群中所有等位基因的频率。
        
        默认实现，子类可覆写以优化性能。
        
        Returns:
            Dict[allele_name, frequency]
        """
        # 初始化所有等位基因频率为 0
        allele_frequencies = {}
        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                for gene in locus.alleles:
                    allele_frequencies[gene.name] = 0.0
        
        # 具体实现依赖子类的数据结构
        # 这里提供一个空壳，子类应覆写
        return allele_frequencies
    
    # ========================================================================
    # Hooks 系统
    # ========================================================================
    
    def set_hook(
        self,
        event_name: str,
        func: Callable,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None
    ) -> None:
        """
        注册事件 Hook。
        
        Args:
            event_name: 事件名称（必须在 ALLOWED_EVENTS 中）
            func: 回调函数，签名应为 func(population)
                  Hook 通过 population 对象访问所有必要数据
            hook_id: Hook 的数值优先级（可选，自动分配）
                     较小的 ID 先执行
            hook_name: Hook 的可读名称（可选，用于调试）
        
        Raises:
            ValueError: 如果事件不存在或 hook_id 已被使用
        
        Example:
            >>> pop.set_hook('first', lambda p: print(f'Step {p.tick}'), hook_name='print_step')
            >>> pop.set_hook('reproduction', lambda p: p.create_snapshot(), hook_id=10)
        """
        if event_name not in self.ALLOWED_EVENTS:
            raise ValueError(f"Event '{event_name}' not in {self.ALLOWED_EVENTS}")
        
        current_ids = [hid for hid, _, _ in self._hooks[event_name]]
        
        if hook_id is None:
            hook_id = (max(current_ids) + 1) if current_ids else 0
        
        if hook_id in current_ids:
            raise ValueError(f"hook_id {hook_id} already exists in event '{event_name}'")
        
        self._hooks[event_name].append((hook_id, hook_name, func))
        # 按 ID 排序保证执行顺序
        self._hooks[event_name].sort(key=lambda x: x[0])
    
    def trigger_event(self, event_name: str) -> None:
        """
        触发事件，执行所有已注册的 hooks（彼此独立）。
        
        Args:
            event_name: 要触发的事件名称
        
        Note:
            - 所有 hooks 只接收 population 对象
            - Hooks 通过 population 对象访问所有必要数据
            - Hooks 按注册时的 hook_id 排序执行（可控制顺序）
            - Hooks 返回值被忽略（应通过副作用修改 population）
            - 标准事件驱动模式：hooks 彼此独立，无隐式依赖
        
        Example:
            >>> pop.set_hook('first', lambda p: print(f'Step {p.tick}'))
            >>> pop.set_hook('reproduction', lambda p: p.create_snapshot())
            >>> pop.trigger_event('first')  # 执行所有注册的 'first' hooks
        """
        for _, _, hook in self._hooks.get(event_name, []):
            hook(self)
    
    def get_hooks(self, event_name: str) -> List[Tuple[int, Optional[str], Callable]]:
        """
        获取特定事件的所有已注册 hooks。
        
        Args:
            event_name: 事件名称
        
        Returns:
            [(hook_id, hook_name, hook_func), ...] 列表
        """
        return list(self._hooks.get(event_name, []))
    
    def remove_hook(self, event_name: str, hook_id: int) -> bool:
        """
        删除指定事件的指定 hook。
        
        Args:
            event_name: 事件名称
            hook_id: Hook 的 ID
        
        Returns:
            删除成功返回 True，否则返回 False
        """
        if event_name not in self._hooks:
            return False
        
        original_len = len(self._hooks[event_name])
        self._hooks[event_name] = [(hid, name, func) for hid, name, func in self._hooks[event_name]
                                    if hid != hook_id]
        return len(self._hooks[event_name]) < original_len
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
