from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Annotated, Callable, List
from numpy.typing import NDArray

from utils.type_def import *
from utils.genetic_entities import Genotype, HaploidGenotype
from utils.index_core import IndexCore

@dataclass
class PopulationStaticData:
    """Static, reusable tensors used by the population model.

    This container holds precomputed lookup tensors and fitness arrays used
    across simulation steps. To support non-trivial zygote formation rules the
    ``gametes_to_zygote_map`` is not required to be one-hot; it may encode
    probabilistic or modified mappings. Index-mapping helpers are intentionally
    kept separate.

    Attributes (shapes documented):
        sexual_selection_fitness: ``(n_genotypes, n_genotypes)`` float64
        female_fecundity_fitness / male_fecundity_fitness: ``(n_genotypes,)`` float64
        female_viability_fitness / male_viability_fitness: ``(n_genotypes,)`` float64
        genotype_to_gametes_map: ``(n_sexes, n_genotypes, n_hg*n_glabs)`` float64
            (compressed haplotype/glab dimension)
        gametes_to_zygote_map: ``(n_hg*n_glabs, n_hg*n_glabs, n_genotypes)`` float64
            (compressed representation mapping gamete pairs to offspring genotypes)
    """

    sexual_selection_fitness: Annotated[NDArray[np.float64], "n_genotypes, n_genotypes"] = None
    female_fecundity_fitness: Annotated[NDArray[np.float64], "n_genotypes"] = None
    male_fecundity_fitness: Annotated[NDArray[np.float64], "n_genotypes"] = None
    female_viability_fitness: Annotated[NDArray[np.float64], "n_genotypes"] = None
    male_viability_fitness: Annotated[NDArray[np.float64], "n_genotypes"] = None
    genotype_to_gametes_map: Annotated[NDArray[np.float64], "n_sexes, n_genotypes, n_hg*n_glabs"] = None  # 压缩形式
    gametes_to_zygote_map: Annotated[NDArray[np.float64], "n_hg*n_glabs, n_hg*n_glabs, n_genotypes"] = None  # 压缩形式
    
    @classmethod
    def make_empty(
        cls,
        n_genotypes: int,
        n_hg: int,
        n_sexes: int = None,
        n_glabs: int = 1,
        dtype_float: np.dtype = np.float64
    ) -> "PopulationStaticData":
        """Create an empty PopulationStaticData with zeroed tensors.

        Args:
            n_genotypes: Number of diploid genotypes.
            n_hg: Number of haploid genotypes.
            n_sexes: Number of sexes; if ``None`` it is inferred from ``Sex`` enum.

        Returns:
            PopulationStaticData: Instance with allocated zero arrays.
        """
        if n_sexes is None:
            n_sexes = max(int(s.value) for s in Sex) + 1

        sexual_selection_fitness = np.zeros((n_genotypes, n_genotypes), dtype=dtype_float)
        female_fecundity_fitness = np.zeros((n_genotypes,), dtype=dtype_float)
        male_fecundity_fitness = np.zeros((n_genotypes,), dtype=dtype_float)
        female_viability_fitness = np.zeros((n_genotypes,), dtype=dtype_float)
        male_viability_fitness = np.zeros((n_genotypes,), dtype=dtype_float)
        # 使用压缩形式: n_hg * n_glabs
        n_hg_glabs = n_hg * n_glabs
        genotype_to_gametes_map = np.zeros((n_sexes, n_genotypes, n_hg_glabs), dtype=dtype_float)
        gametes_to_zygote_map = np.zeros((n_hg_glabs, n_hg_glabs, n_genotypes), dtype=dtype_float)

        return cls(
            sexual_selection_fitness=sexual_selection_fitness,
            female_fecundity_fitness=female_fecundity_fitness,
            male_fecundity_fitness=male_fecundity_fitness,
            female_viability_fitness=female_viability_fitness,
            male_viability_fitness=male_viability_fitness,
            genotype_to_gametes_map=genotype_to_gametes_map,
            gametes_to_zygote_map=gametes_to_zygote_map,
        )
    
    def initialize_zygote_map(
        self,
        haploid_genotypes: List[HaploidGenotype],
        diploid_genotypes: List[Genotype],
        zygote_modifiers: Optional[List[Callable]] = None
    ) -> None:
        """Initialize the ``gametes_to_zygote_map`` tensor.

        The method first populates a baseline mapping following Mendelian
        inheritance for all haplotype pairs and gamete-label combinations, and
        then applies optional zygote modifiers to transform the tensor.

        Args:
            haploid_genotypes: List of all haploid genotype objects.
            diploid_genotypes: List of all diploid genotype objects.
            zygote_modifiers: Optional sequence of callables that accept and
                return a modified ``gametes_to_zygote_map`` tensor.
        """
        n_hg = len(haploid_genotypes)
        n_genotypes = len(diploid_genotypes)
        n_hg_glabs = self.gametes_to_zygote_map.shape[0]  # 压缩维度
        # derive n_glabs from shape and provided n_hg
        if n_hg <= 0:
            raise ValueError("haploid_genotypes must be non-empty")
        if n_hg_glabs % n_hg != 0:
            raise ValueError("inconsistent shapes: n_hg_glabs not divisible by n_hg")
        n_glabs = n_hg_glabs // n_hg
        
        # 1. 按默认遗传规律生成one-hot张量
        # 初始化所有组合为零
        self.gametes_to_zygote_map.fill(0.0)
        
        # 为每个单倍型组合创建对应的二倍型
        for idx_hg1, hg1 in enumerate(haploid_genotypes):
            for idx_hg2, hg2 in enumerate(haploid_genotypes):
                # 生成合子基因型
                zygote_gt = Genotype(
                    species=hg1.species,
                    maternal=hg1,
                    paternal=hg2
                )
                
                # 如果这个基因型在我们的列表中
                if zygote_gt in diploid_genotypes:
                    idx_gt = diploid_genotypes.index(zygote_gt)
                    # Baseline: labels are equivalent — populate all (glab1, glab2)
                    for glab1 in range(n_glabs):
                        for glab2 in range(n_glabs):
                            compressed_idx1 = IndexCore().compress_hg_glab(idx_hg1, glab1, n_glabs)
                            compressed_idx2 = IndexCore().compress_hg_glab(idx_hg2, glab2, n_glabs)
                            self.gametes_to_zygote_map[compressed_idx1, compressed_idx2, idx_gt] = 1.0
        
        # 2. 应用合子修饰器进行改造
        if zygote_modifiers:
            for modifier in zygote_modifiers:
                self.apply_zygote_modifier(modifier)
    
    def apply_zygote_modifier(self, modifier_func: Callable[[NDArray], NDArray]) -> None:
        """Apply a zygote modifier callable to the gametes-to-zygote tensor.

        Args:
            modifier_func: Callable that accepts the current
                ``gametes_to_zygote_map`` NDArray and returns a modified NDArray.
        """
        self.gametes_to_zygote_map = modifier_func(self.gametes_to_zygote_map)
    
    def initialize_gamete_map(
        self,
        diploid_genotypes: List[Genotype],
        haploid_genotypes: List[HaploidGenotype],
        gamete_modifiers: Optional[List[Callable]] = None
    ) -> None:
        """Initialize the ``genotype_to_gametes_map`` tensor.

        The baseline mapping is derived from each diploid genotype's gamete
        production (``genotype.produce_gametes()``) and is then transformed by
        optional gamete modifier callables.
        """
        n_genotypes = len(diploid_genotypes)
        n_hg = len(haploid_genotypes)
        n_sexes = self.genotype_to_gametes_map.shape[0]
        n_hg_glabs = self.genotype_to_gametes_map.shape[2]  # 压缩维度
        # derive n_glabs from shape and provided n_hg
        if n_hg <= 0:
            raise ValueError("haploid_genotypes must be non-empty")
        if n_hg_glabs % n_hg != 0:
            raise ValueError("inconsistent shapes: n_hg_glabs not divisible by n_hg")
        n_glabs = n_hg_glabs // n_hg
        
        # 1. 按默认遗传规律生成基本映射
        self.genotype_to_gametes_map.fill(0.0)
        
        # 为每个基因型和性别生成配子
        for idx_genotype, genotype in enumerate(diploid_genotypes):
            for sex_idx in range(n_sexes):
                gametes = genotype.produce_gametes()
                
                # 将配子频率映射到张量（使用压缩索引）
                for gamete, freq in gametes.items():
                    if gamete in haploid_genotypes:
                        idx_hg = haploid_genotypes.index(gamete)
                        # baseline: only map glab index 0 by default
                        compressed_idx = IndexCore().compress_hg_glab(idx_hg, 0, n_glabs)
                        self.genotype_to_gametes_map[sex_idx, idx_genotype, compressed_idx] = freq
        
        # 2. 应用配子修饰器进行改造
        if gamete_modifiers:
            for modifier in gamete_modifiers:
                self.apply_gamete_modifier(modifier)

    def apply_gamete_modifier(self, modifier_func: Callable[[NDArray], NDArray]) -> None:
        """Apply a gamete modifier callable to the genotype-to-gametes tensor.

        Args:
            modifier_func: Callable that accepts the current
                ``genotype_to_gametes_map`` NDArray and returns a modified NDArray.
        """
        self.genotype_to_gametes_map = modifier_func(self.genotype_to_gametes_map)
        
    def set_gamete_to_zygote(self, hg1_idx: int, glab1_idx: int, hg2_idx: int, glab2_idx: int, genotype_idx: int) -> None:
        """Set a deterministic mapping from a gamete pair to a genotype index.

        The function zeros the corresponding slice and writes a one-hot mapping
        for the provided ``genotype_idx``.
        """
        self.gametes_to_zygote_map[hg1_idx, glab1_idx, hg2_idx, glab2_idx, :] = 0
        self.gametes_to_zygote_map[hg1_idx, glab1_idx, hg2_idx, glab2_idx, int(genotype_idx)] = 1

    def get_gamete_to_zygote(self, hg1_idx: int, glab1_idx: int, hg2_idx: int, glab2_idx: int) -> int:
        """Return the genotype index for a gamete pair, or -1 if unset.

        If multiple genotype entries are present the index of the maximum value
        is returned.
        """
        row = self.gametes_to_zygote_map[hg1_idx, glab1_idx, hg2_idx, glab2_idx]
        if not np.any(row):
            return -1
        return int(np.argmax(row))

    def compute_zygote_distribution_from_gametes(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute genotype distribution from gamete distributions p and q.
        p and q are length n_hg arrays (weights or probabilities).
        Returns a length-n_genotypes array of counts/weights.
        """
        if p.shape != q.shape:
            raise ValueError("p and q must have same shape")
        n_hg = p.shape[0]
        
        # 处理n_glabs维度
        # 假设p和q已经是(n_hg, n_glabs)形状，如果不是则扩展维度
        if p.ndim == 1:
            p = p[:, np.newaxis]
            q = q[:, np.newaxis]
        
        n_glabs = p.shape[1]
        
        # 使用张量运算计算合子分布
        # 这是核心优化点，使用预计算的张量进行快速运算
        return np.einsum('hl, hk, lkg -> g', p, q, self.gametes_to_zygote_map, optimize=True)
