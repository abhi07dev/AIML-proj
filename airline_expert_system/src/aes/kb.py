from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .terms import Predicate


@dataclass(frozen=True)
class Rule:
    """
    A Horn-style rule: premises -> conclusion

    Example (FOL-ish):
      cargo(?C) ∧ flight(?F) ∧ same_destination(?C, ?F) -> can_assign(?C, ?F)
    """

    name: str
    premises: Tuple[Predicate, ...]
    conclusion: Predicate

    def __str__(self) -> str:
        left = " ∧ ".join(str(p) for p in self.premises) if self.premises else "TRUE"
        return f"{self.name}: {left} -> {self.conclusion}"


class KnowledgeBase:
    """
    Stores facts + rules.

    Facts are predicates (ground, or partially ground if you want, but we prefer ground facts).
    """

    def __init__(self) -> None:
        self._facts: Set[Predicate] = set()
        self._rules: List[Rule] = []
        self._provenance: Dict[Predicate, str] = {}  # "how" fact was derived

        # Defaults (non-monotonic-ish simulation):
        # - on_time(F) is assumed for every flight(F) unless delayed(F) is known.
        self.enable_defaults: bool = True

    # -------- Facts --------
    def add_fact(self, fact: Predicate, *, why: str = "") -> None:
        if fact not in self._facts:
            self._facts.add(fact)
            if why:
                self._provenance[fact] = why

    def remove_fact(self, fact: Predicate) -> None:
        self._facts.discard(fact)
        self._provenance.pop(fact, None)

    def has_fact(self, fact: Predicate) -> bool:
        return fact in self._facts

    def facts(self) -> Set[Predicate]:
        return set(self._facts)

    def why(self, fact: Predicate) -> str:
        return self._provenance.get(fact, "")

    # -------- Rules --------
    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def rules(self) -> List[Rule]:
        return list(self._rules)

    # -------- Defaults / exceptions --------
    def apply_defaults(self) -> List[Predicate]:
        """
        Add default facts (like on_time(F)) based on existing facts.
        Returns a list of newly added default facts.
        """

        if not self.enable_defaults:
            return []

        new: List[Predicate] = []
        flights = [f for f in self._facts if f.name == "flight" and len(f.args) == 1]
        delayed = {f.args[0] for f in self._facts if f.name == "delayed" and len(f.args) == 1}

        for fl in flights:
            flight_id = fl.args[0]
            if flight_id in delayed:
                continue
            on_time = Predicate("on_time", (flight_id,))
            if on_time not in self._facts:
                self.add_fact(on_time, why="default: flights are on time unless delayed")
                new.append(on_time)
        return new

    def apply_exception_overrides(self) -> List[str]:
        """
        Enforce a small set of exception policies.
        Returns human-readable notes about overrides performed.

        - delayed(F) overrides on_time(F)
        """

        notes: List[str] = []
        delayed = [f for f in self._facts if f.name == "delayed" and len(f.args) == 1]
        for d in delayed:
            fl = d.args[0]
            ot = Predicate("on_time", (fl,))
            if ot in self._facts:
                self.remove_fact(ot)
                notes.append(f"override: removed {ot} because delayed({fl}) is true")
        return notes

