from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Optional, Dict, Any, Callable, Union
from typing import TypeVar, cast
import inspect
from utils.type_def import Sex
from utils.genetic_entities import Genotype, HaploidGenotype

# Bulk-only modifier interface expectations (strict form):
# - gamete modifier: callable() -> Dict[(sex_idx:int, genotype_idx:int) -> Dict[compressed_hg_glab_idx:int -> freq:float]]
# - zygote modifier: callable() -> Dict[(c1:int, c2:int) -> replacement]
#
# The modifiers use compressed integer indices as keys so that outputs can be
# written back directly into underlying numeric tensors. This avoids expensive
# object-to-index lookups inside wrappers and prevents passing large object
# graphs at runtime.

class GameteModifier(Protocol):
    """Protocol for a bulk gamete modifier.

    Implementations should provide a callable that accepts either zero or one
    argument (an optional `population` object) and returns a nested mapping of
    gamete frequency updates. The canonical return type is::

        Dict[Tuple[int, int], Dict[int, float]]

    where the outer key is ``(sex_idx, genotype_idx)`` and the inner mapping is
    ``{ compressed_hg_glab_idx: frequency, ... }``. Keys may be flexible types
    in wrappers (for convenience) but should ultimately resolve to integers.

    Notes:
        - ``sex_idx`` is an ``int``.
        - ``genotype_idx`` may be an ``int``, a ``Genotype`` object, or a
          string produced by ``Genotype.to_string()``.

    Example:

        return {(0, 5): {3: 0.2, 4: 0.8}, (1, 5): {3: 1.0}}

    The result writes frequency distributions for compressed indices directly
    back into numeric tensors.
    """
    def __call__(self, *args: Any) -> Dict[Any, Dict[int, float]]: ...

# 合子修饰器接口保持不变
class ZygoteModifier(Protocol):
    """Protocol for a bulk zygote modifier.

    Implementations should provide a callable that accepts zero or one argument
    (an optional `population`) and returns a mapping from a flexible key to a
    replacement. The key identifies the zygote pairing and may take one of
    several forms that wrappers can resolve into compressed coordinate pairs
    ``(c1, c2)``.

    Supported key representations include:
        - compressed index pair ``(c1, c2)``
        - nested tuples ``((hg_obj|hg_str|idx_hg, glab_label?), (hg_obj|hg_str|idx_hg, glab_label?))``
        - other wrapper-resolvable representations

    Replacement values may be one of:
        - an integer index ``idx_modified`` (index into diploid genotype list)
        - a ``Genotype`` instance (wrappers will convert to an index)
        - a dict ``{ idx_modified: probability, ... }`` specifying a distribution

    The protocol returns::

        Dict[Any, Union[int, Genotype, Dict[int, float]]]
    """
    def __call__(self, *args: Any) -> Dict[Any, Union[int, Genotype, Dict[int, float]]]: ...
