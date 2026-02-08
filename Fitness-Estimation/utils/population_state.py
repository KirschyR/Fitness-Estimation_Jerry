"""PopulationState data container.

This module implements a lightweight NumPy-backed ``PopulationState`` used
to store per-sex, per-age (optional), per-genotype individual counts, as
well as optional sperm storage and female occupancy arrays for age-structured
models.

Key points:
- The high-level API uses ``IndividualType`` triples ``(sex, age, genotype_index)``
    to simplify integration with other modules. Use helpers in
    ``utils.individual_type`` to construct and inspect these objects.
- In non-age-structured mode (``n_ages is None``):
    - ``individual_count`` has shape ``(n_sexes, n_genotypes)``.
    - Age-related methods will raise ``RuntimeError``.
- In age-structured mode: 
    - ``individual_count`` has shape ``(n_sexes, n_ages, n_genotypes)``.
    - ``sperm_storage`` and ``female_occupancy`` arrays are created to support
        sperm storage mechanics.

Example:
        from utils.index_core import IndexCore
        from utils.individual_type import make_indtype, Sex

        ic = IndexCore(...)
        st = PopulationState.make_empty(ic, n_ages=4)
        ind = make_indtype(Sex.MALE, 0, 2)
        st.add_count(ind, 5)

Note:
        This module is a data container only; mating, selection and inheritance
        logic belong to higher-level components (for example the population
        implementations and simulation loop).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Annotated, Union, TYPE_CHECKING
from numpy.typing import NDArray

from utils.type_def import Sex
from utils.index_core import IndexCore
from utils.type_def import IndividualType, get_sex, get_age, get_genotype_index, Sex

if TYPE_CHECKING:
    from utils.genetic_entities import Genotype

@dataclass
class PopulationState:
    """Container for population state arrays.

    Attributes (shapes documented):
        - ``individual_count``: ndarray storing counts per sex/age/genotype.
            Shapes: ``(n_sexes, n_ages, n_genotypes)`` or ``(n_sexes, n_genotypes)``.
        - ``sperm_storage``: Optional ndarray for stored sperm with compressed
            haplotype/label dimension: ``(n_ages, n_genotypes, n_hg * n_glabs)``.
        - ``female_occupancy``: Optional ndarray with shape ``(n_ages, n_genotypes)``.
    """

    individual_count: Union[Annotated[NDArray[np.float64], "sex, age, genotype"], Annotated[NDArray[np.float64], "sex, genotype"]] = None
    sperm_storage: Optional[Annotated[NDArray[np.float64], "age, genotype, haplogenotype * gamete_label"]] = None
    female_occupancy: Optional[Annotated[NDArray[np.float64], "age, genotype"]] = None

    @classmethod
    def make_empty(
        cls,
        n_genotypes: int,
        n_hg: int,
        n_ages: Optional[int] = None,
        n_sexes: Optional[int] = None,
        n_glabs: Optional[int] = 1,
        dtype_float: np.dtype = np.float64,
    ) -> "PopulationState":
        """Create an empty ``PopulationState`` with zeroed arrays.

        Args:
            n_genotypes: Number of diploid genotypes (``g``).
            n_hg: Number of haploid genotypes (``h``).
            n_ages: Number of age classes or ``None`` for non-age-structured mode.
            n_sexes: Number of sexes; if ``None`` the number is inferred from
                the ``Sex`` enum (``max(Sex) + 1``).
            n_glabs: Number of gamete labels (default: 1).
            dtype_float: NumPy dtype to use for floating arrays (default: ``np.float64``).

        Returns:
            PopulationState: A newly allocated state object with zero counts.
        """
        if n_sexes is None:
            n_sexes = max(int(s.value) for s in Sex) + 1
        
        if n_ages is None:
            individual_count = np.zeros((n_sexes, n_genotypes), dtype=dtype_float)
            sperm_storage = None
            female_occupancy = None
        
        else:
            individual_count = np.zeros((n_sexes, n_ages, n_genotypes), dtype=dtype_float)
            # 使用压缩形式: 将 n_hg 和 n_glabs 合并为一个维度
            sperm_storage = np.zeros((n_ages, n_genotypes, n_hg * n_glabs), dtype=dtype_float)
            female_occupancy = np.zeros((n_ages, n_genotypes), dtype=dtype_float)
        
        return cls(
            individual_count=individual_count,
            sperm_storage=sperm_storage,
            female_occupancy=female_occupancy,
        )
    
    def is_age_structured(self) -> bool:
        """Return True if this state uses an age dimension (age-structured)."""
        return self.individual_count.ndim == 3
    
    def add_count(self, indtype: IndividualType, count: float) -> None:
        """Add a count for a specific individual type.

        Args:
            indtype: ``IndividualType`` triple ``(sex, age, genotype_index)``.
            count: Number to add (may be float).
        """
        sex = get_sex(indtype)
        genotype_index = get_genotype_index(indtype)
        if self.is_age_structured():
            age = get_age(indtype)
            self.individual_count[sex, age, genotype_index] += count
        else:
            self.individual_count[sex, genotype_index] += count
    
    def get_count(self, indtype: IndividualType) -> float:
        """Return the count for a specific individual type.

        Args:
            indtype: ``IndividualType`` triple ``(sex, age, genotype_index)``.

        Returns:
            float: The stored count for the requested individual type.
        """
        sex = get_sex(indtype)
        genotype_index = get_genotype_index(indtype)
        if self.is_age_structured():
            age = get_age(indtype)
            return self.individual_count[sex, age, genotype_index]
        else:
            return self.individual_count[sex, genotype_index]
    
    def store_sperm(self, age: int, genotype_index: int, haplotype_index: int, gamete_label: int = 0, count: float = 1.0) -> None:
        """Store sperm (haplotype) for female individuals of a given age/genotype.

        Args:
            age: Age index of the female recipients.
            genotype_index: Genotype index of the female recipients.
            haplotype_index: Index of the haplotype to store.
            gamete_label: Gamete label index (default: 0).
            count: Amount to add (float, default: 1.0).

        Raises:
            RuntimeError: If the state is not age-structured.
        """
        if not self.is_age_structured():
            raise RuntimeError("store_sperm 只在年龄结构模式下可用")
        
        self.sperm_storage[age, genotype_index, haplotype_index, gamete_label] += count
    
    def get_stored_sperm(self, age: int, genotype_index: int, haplotype_index: int, gamete_label: int = 0) -> float:
        """Return stored sperm count for a specific age/genotype/haplotype.

        Args:
            age: Age index of the female recipients.
            genotype_index: Genotype index of the female recipients.
            haplotype_index: Haplotype index of the stored sperm.
            gamete_label: Gamete label index (default: 0).

        Returns:
            float: Stored sperm amount.

        Raises:
            RuntimeError: If the state is not age-structured.
        """
        if not self.is_age_structured():
            raise RuntimeError("get_stored_sperm 只在年龄结构模式下可用")
        
        return self.sperm_storage[age, genotype_index, haplotype_index, gamete_label]
    
    def has_stored_sperm(self, age: int, genotype_index: int) -> bool:
        """Return True if any sperm is stored for the given age/genotype.

        Args:
            age: Age index of the female recipients.
            genotype_index: Genotype index of the female recipients.

        Returns:
            bool: True if any stored sperm is present, False otherwise.

        Raises:
            RuntimeError: If the state is not age-structured.
        """
        if not self.is_age_structured():
            raise RuntimeError("has_stored_sperm 只在年龄结构模式下可用")
        
        # 检查该雌性个体的所有单倍型和所有配子标签是否有任何存储的精子
        return np.any(self.sperm_storage[age, genotype_index, :, :] > 0)