"""Simplified observation/filter helpers.

Provides a small, clear API to build observation "rules" (NumPy arrays)
from user-friendly group specifications and to apply those rules to a
`PopulationState.individual_count` array.

Design choices in this simplified version:
- Keep logic minimal and easy to read.
- Accept group specs as list/tuple (unnamed) or dict (named), each spec is
  a dict with optional keys: `genotype`, `age`, `sex`, `unordered`.
- `genotype` selectors may be ints, strings, or Genotype objects (requires
  `diploid_genotypes` to resolve strings/objects to indices).
- Provide a pure function `apply_rule(individual_count, rule)` for projection.
- No numba backend here — keep implementation straightforward.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from utils.index_core import IndexCore
from utils.population_state import PopulationState
from utils.type_def import Sex
from utils.genetic_structures import Species
from utils.base_population import BasePopulation

class ObservationFilter:
    """Build observation rules (NumPy arrays) from simple group specs.

    Example group item (dict):
        {"age": [2,3,4], "genotype": ["WT|WT"], "sex": ["male"]}

    `build_filter` accepts:
      - None (default: one group per genotype)
      - list/tuple of spec-items (auto-named group_0, group_1, ...)
      - dict mapping name -> spec-item.

    Returns:
      - rule: np.ndarray mask
      - labels: List[str]
    """

    def __init__(self, registry: IndexCore):
        self.registry = registry

    def _resolve_genotype_index(self, diploid_genotypes: Sequence[Any], sel: Any) -> Optional[int]:
        try:
            return self.registry.resolve_genotype_index(diploid_genotypes, sel, strict=True)
        except Exception:
            return None

    @staticmethod
    def _normalize_genotype_key(g) -> str:
        """Return a canonical string for a diploid genotype for unordered grouping.

        Prefer `to_string()` where available; split on '|' and sort parts.
        """
        try:
            s = g.to_string()
        except Exception:
            s = str(g)
        if "|" in s:
            a, b = s.split("|", 1)
            parts = sorted([a.strip(), b.strip()])
            return "::".join(parts)
        return s

    @staticmethod
    def _make_age_predicate(age_spec: Optional[Union[Iterable[int], Tuple[int, int], Callable[[int], bool], Iterable[Iterable[int]]]]) -> Callable[[int], bool]:
        """Build an age predicate supporting several shorthand forms.

        Supported forms:
        - None -> all ages
        - callable(a) -> used directly
        - single tuple/list (start, end) -> closed interval [start, end]
        - iterable of ints -> explicit ages
        - iterable of (start,end) pairs -> union of closed intervals
        
        Examples:
            [2,3,4] -> explicit ages
            [ [2,7] ] -> ages 2..7 inclusive
            [ [2,4], [6,7] ] -> ages 2,3,4,6,7
        """
        if age_spec is None:
            return lambda a: True
        if callable(age_spec):
            return age_spec

        # If it's a 2-tuple/list treat as a closed interval
        try:
            # detect simple pair like (2,7)
            if (isinstance(age_spec, (list, tuple)) and len(age_spec) == 2
                    and all(not isinstance(x, (list, tuple)) for x in age_spec)):
                start, end = int(age_spec[0]), int(age_spec[1])
                return lambda a: (a >= start and a <= end)
        except Exception:
            pass

        # If it's an iterable, collect either explicit ages or ranges
        allowed = set()
        for item in age_spec:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    s, e = int(item[0]), int(item[1])
                except Exception:
                    continue
                if e < s:
                    continue
                for aa in range(s, e + 1):
                    allowed.add(aa)
            else:
                try:
                    allowed.add(int(item))
                except Exception:
                    # ignore unparseable entries
                    continue

        return lambda a: int(a) in allowed

    @staticmethod
    def _resolve_sexes(spec_sex: Optional[Union[str, int, Iterable[Any]]], n_sexes: int) -> List[int]:
        if spec_sex is None:
            return list(range(n_sexes))
        if isinstance(spec_sex, (str, int, Sex)):
            if isinstance(spec_sex, str):
                s = spec_sex.lower()
                if s in ("male", "m"):
                    return [int(Sex.MALE)]
                if s in ("female", "f"):
                    return [int(Sex.FEMALE)]
                # try int
                return [int(spec_sex)]
            if isinstance(spec_sex, Sex):
                return [int(spec_sex)]
            return [int(spec_sex)]
        res: List[int] = []
        for x in spec_sex:
            res.extend(ObservationFilter._resolve_sexes(x, n_sexes))
        return sorted(set(res))

    def _build_unordered_map(self, diploid_genotypes: Sequence[Any]) -> Dict[str, List[int]]:
        """Build mapping canonical_key -> list of genotype indices."""
        mp: Dict[str, List[int]] = {}
        for i, g in enumerate(diploid_genotypes):
            key = self._normalize_genotype_key(g)
            mp.setdefault(key, []).append(i)
        return mp

    def _resolve_genotype_list(self, gen_spec: Optional[Iterable[Any]], diploid_genotypes: Optional[Sequence[Any]], unordered: bool) -> List[int]:
        """Resolve genotype selectors into a list of indices.

        Rules:
          - If gen_spec is None => all genotype indices.
          - If elements are ints => used directly.
          - If strings/objects and diploid_genotypes provided => resolved via registry or unordered map.
        """
        if gen_spec is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required to enumerate genotypes")
            return list(range(len(diploid_genotypes)))

        # prepare unordered map lazily
        unordered_map = None
        if unordered and diploid_genotypes is not None:
            unordered_map = self._build_unordered_map(diploid_genotypes)

        out: List[int] = []
        for sel in gen_spec:
            if isinstance(sel, int):
                out.append(int(sel))
                continue
            if diploid_genotypes is None:
                # cannot resolve non-int selectors
                continue
            # try registry resolution (by object or string)
            idx = self._resolve_genotype_index(diploid_genotypes, sel)
            if idx is not None:
                out.append(int(idx))
                # if unordered grouping requested, also include any other
                # indices that share the same canonical key (e.g. "A|a" and "a|A")
                if unordered and isinstance(sel, str) and unordered_map is not None:
                    key = sel
                    if "|" in key:
                        parts = key.split("|", 1)
                        key = "::".join(sorted([parts[0].strip(), parts[1].strip()]))
                    if key in unordered_map:
                        out.extend(unordered_map[key])
                continue
            # if unordered, try normalized key
            if unordered and isinstance(sel, str):
                key = sel
                # allow users to pass either 'A|a' or canonical sorted form
                if "|" in key:
                    parts = key.split("|", 1)
                    key = "::".join(sorted([parts[0].strip(), parts[1].strip()]))
                if unordered_map is not None and key in unordered_map:
                    out.extend(unordered_map[key])
                    continue
            # last resort: try matching to_string() by scanning
            for i, g in enumerate(diploid_genotypes):
                try:
                    if hasattr(g, "to_string") and g.to_string() == str(sel):
                        out.append(i)
                        break
                except Exception:
                    pass
        return sorted(set(out))

    def build_filter(
        self,
        pop_or_state: Union[PopulationState, BasePopulation],
        *,
        diploid_genotypes: Optional[Union[Sequence[Any], Species, BasePopulation]] = None,
        groups: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = None,
        collapse_age: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build a rule (NumPy mask) and labels from `groups`.

        `groups` may be:
          - None: one group per genotype (requires `diploid_genotypes`).
          - list/tuple: sequence of spec-items (each spec-item is a dict or iterable of genotype selectors).
          - dict: mapping name->spec-item.

        Spec-item keys supported: `genotype`, `age`, `sex`, `unordered`.
        """
        # Support passing either a PopulationState or a BasePopulation instance.
        if isinstance(pop_or_state, BasePopulation):
            pop = pop_or_state
            state = pop.state
        elif isinstance(pop_or_state, PopulationState):
            state = pop_or_state
            pop = None
        else:
            raise TypeError("first argument must be a PopulationState or BasePopulation instance")

        arr = state.individual_count
        n_sexes = int(arr.shape[0])
        is_age = state.is_age_structured()
        n_ages = int(arr.shape[1]) if is_age else None
        n_genotypes = int(arr.shape[-1])

        specs: List[Tuple[str, Dict[str, Any]]] = []
        if groups is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required when groups is None")
            specs = [(f"g{g}", {"genotype": [g]}) for g in range(n_genotypes)]
        elif isinstance(groups, (list, tuple)):
            for i, item in enumerate(groups):
                name = f"group_{i}"
                if isinstance(item, dict):
                    specs.append((name, item))
                else:
                    specs.append((name, {"genotype": item}))
        elif isinstance(groups, dict):
            for name, item in groups.items():
                if isinstance(item, dict):
                    specs.append((str(name), item))
                else:
                    specs.append((str(name), {"genotype": item}))
        else:
            raise ValueError("groups must be None, list/tuple, or dict")

        labels = [s[0] for s in specs]
        n_groups = len(labels)

        # Coerce diploid_genotypes if caller passed a Species or BasePopulation
        if isinstance(diploid_genotypes, BasePopulation):
            try:
                diploid_genotypes = diploid_genotypes._get_all_possible_diploid_genotypes()
            except Exception:
                # fallback to species if available
                try:
                    diploid_genotypes = list(diploid_genotypes.species.iter_genotypes())
                except Exception:
                    diploid_genotypes = None
        elif isinstance(diploid_genotypes, Species):
            try:
                diploid_genotypes = list(diploid_genotypes.iter_genotypes())
            except Exception:
                diploid_genotypes = None

        # resolve per-group selectors
        per_genotypes: List[List[int]] = []
        per_sexes: List[List[int]] = []
        per_age_preds: List[Callable[[int], bool]] = []

        for _, spec in specs:
            gen_spec = spec.get("genotype") or spec.get("genotypes")
            unordered = bool(spec.get("unordered", False))
            gen_list = self._resolve_genotype_list(gen_spec, diploid_genotypes, unordered)
            per_genotypes.append(gen_list)

            sex_spec = spec.get("sex")
            per_sexes.append(self._resolve_sexes(sex_spec, n_sexes))

            age_spec = spec.get("age")
            per_age_preds.append(self._make_age_predicate(age_spec))

        # build mask
        if is_age and not collapse_age:
            mask = np.zeros((n_groups, n_sexes, n_ages, n_genotypes), dtype=np.float64)
            for gi in range(n_groups):
                for gidx in per_genotypes[gi]:
                    for s in per_sexes[gi]:
                        for a in range(n_ages):
                            if per_age_preds[gi](a):
                                mask[gi, s, a, gidx] = 1.0
        elif is_age and collapse_age:
            mask = np.zeros((n_groups, n_sexes, n_genotypes), dtype=np.float64)
            for gi in range(n_groups):
                for gidx in per_genotypes[gi]:
                    for s in per_sexes[gi]:
                        # If any ages match the predicate, mark this genotype as
                        # selected for the collapsed mask. The collapsed mask is
                        # expanded uniformly across ages in `apply_rule`, so it
                        # cannot represent per-age inclusion — it only indicates
                        # that the genotype should be included when summing ages.
                        any_selected = False
                        for a in range(n_ages):
                            if per_age_preds[gi](a):
                                any_selected = True
                                break
                        mask[gi, s, gidx] = 1.0 if any_selected else 0.0
        else:
            mask = np.zeros((n_groups, n_sexes, n_genotypes), dtype=np.float64)
            for gi in range(n_groups):
                for gidx in per_genotypes[gi]:
                    for s in per_sexes[gi]:
                        mask[gi, s, gidx] = 1.0

        return mask, labels


def apply_rule(individual_count: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """Pure function: apply `rule` to `individual_count` and sum over genotype axis.

    Supported shapes:
      - individual_count: (n_sexes, n_ages, n_genotypes) or (n_sexes, n_genotypes)
      - rule: (n_groups, n_sexes, n_ages, n_genotypes)
              (n_groups, n_sexes, n_genotypes)   (collapsed ages or non-age)

    Returns:
      - observed: (n_groups, n_sexes, n_ages) or (n_groups, n_sexes)
    """
    arr = individual_count
    mask = rule
    if arr.ndim == 3:
        # age-structured
        if mask.ndim == 4:
            prod = mask * arr[np.newaxis, ...]
            return prod.sum(axis=-1)
        if mask.ndim == 3:
            # collapsed ages: expand then sum ages
            expanded = mask[:, :, None, :]
            prod = expanded * arr[np.newaxis, ...]
            return prod.sum(axis=-1).sum(axis=-1)
        raise ValueError("Unsupported rule ndim for age-structured state")

    if arr.ndim == 2:
        # non-age
        if mask.ndim == 3:
            prod = mask * arr[np.newaxis, ...]
            return prod.sum(axis=-1)
        if mask.ndim == 2:
            prod = mask[:, None, :] * arr[None, ...]
            return prod.sum(axis=-1)
        raise ValueError("Unsupported rule ndim for non-age state")

    raise ValueError("Unsupported individual_count ndim")
