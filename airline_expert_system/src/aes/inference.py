from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .kb import KnowledgeBase, Rule
from .terms import Const, Predicate, Term, Var
from .unification import Substitution, apply_subst_pred, unify


@dataclass(frozen=True)
class TraceStep:
    kind: str  # "default" | "rule" | "conflict" | "query"
    message: str


def _iter_facts_by_name(facts: Iterable[Predicate], name: str) -> Iterator[Predicate]:
    for f in facts:
        if f.name == name:
            yield f


def _match_premises(
    premises: Sequence[Predicate],
    facts: Set[Predicate],
    subst: Optional[Substitution] = None,
) -> Iterator[Substitution]:
    """
    Find substitutions that make all premises true given current facts.

    This is a simple backtracking join:
      unify(premise_i, some_fact) and propagate bindings.
    """

    subst = dict(subst or {})
    if not premises:
        yield subst
        return

    first, rest = premises[0], premises[1:]
    # Only try facts with same predicate name for speed/readability.
    for fact in _iter_facts_by_name(facts, first.name):
        s2 = unify(apply_subst_pred(first, subst), fact, subst)
        if s2 is None:
            continue
        yield from _match_premises(rest, facts, s2)


def find_conflicts(facts: Set[Predicate]) -> List[Tuple[Predicate, Predicate]]:
    """
    Simplified resolution-style contradiction detection:
      conflict if both P(...) and not_P(...) exist.
    """

    conflicts: List[Tuple[Predicate, Predicate]] = []
    fact_set = set(facts)
    for f in fact_set:
        g = f.negate()
        if g in fact_set:
            # Keep only one direction
            if f.name < g.name:
                conflicts.append((f, g))
    return conflicts


class InferenceEngine:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    # ---------------- Forward chaining ----------------
    def forward_chain(self, *, max_iterations: int = 50, trace: bool = True) -> List[TraceStep]:
        steps: List[TraceStep] = []

        for _ in range(max_iterations):
            changed = False

            # Defaults first (so rules can use them)
            new_defaults = self.kb.apply_defaults()
            if new_defaults:
                changed = True
                if trace:
                    for d in new_defaults:
                        steps.append(TraceStep("default", f"Derived default fact: {d}"))

            # Exceptions override defaults
            overrides = self.kb.apply_exception_overrides()
            if overrides:
                changed = True
                if trace:
                    for n in overrides:
                        steps.append(TraceStep("default", n))

            facts_snapshot = self.kb.facts()

            for rule in self.kb.rules():
                for s in _match_premises(rule.premises, facts_snapshot):
                    concl = apply_subst_pred(rule.conclusion, s)
                    if concl not in self.kb.facts():
                        self.kb.add_fact(concl, why=f"rule {rule.name} with {format_subst(s)}")
                        changed = True
                        if trace:
                            steps.append(
                                TraceStep(
                                    "rule",
                                    f"Applied {rule.name} => derived {concl} using {format_subst(s)}",
                                )
                            )

            # Conflict check (after each iteration)
            confs = find_conflicts(self.kb.facts())
            if confs and trace:
                for a, b in confs:
                    steps.append(TraceStep("conflict", f"Conflict detected: {a} AND {b}"))

            if not changed:
                break

        return steps

    # ---------------- Backward chaining ----------------
    def ask(
        self,
        goal: Predicate,
        *,
        trace: bool = True,
        max_depth: int = 20,
    ) -> Tuple[List[Substitution], List[TraceStep]]:
        steps: List[TraceStep] = []
        answers = list(self._prove(goal, {}, steps, trace=trace, depth=0, max_depth=max_depth))
        return answers, steps

    def _prove(
        self,
        goal: Predicate,
        subst: Substitution,
        steps: List[TraceStep],
        *,
        trace: bool,
        depth: int,
        max_depth: int,
    ) -> Iterator[Substitution]:
        if depth > max_depth:
            return

        goal2 = apply_subst_pred(goal, subst)
        if trace:
            steps.append(TraceStep("query", f"{'  '*depth}Prove: {goal2}"))

        # 1) Try matching against known facts
        for fact in _iter_facts_by_name(self.kb.facts(), goal2.name):
            s2 = unify(goal2, fact, subst)
            if s2 is not None:
                if trace:
                    steps.append(
                        TraceStep("query", f"{'  '*depth}Matched fact {fact} with {format_subst(s2)}")
                    )
                yield s2

        # 2) Try rules that can conclude this predicate
        for rule in self.kb.rules():
            if rule.conclusion.name != goal2.name:
                continue

            # Unify goal with rule conclusion to get initial bindings
            s_head = unify(goal2, rule.conclusion, subst)
            if s_head is None:
                continue

            if trace:
                steps.append(
                    TraceStep(
                        "query",
                        f"{'  '*depth}Try rule {rule.name} (need premises) with {format_subst(s_head)}",
                    )
                )

            yield from self._prove_all(rule.premises, s_head, steps, trace=trace, depth=depth + 1, max_depth=max_depth)

    def _prove_all(
        self,
        goals: Sequence[Predicate],
        subst: Substitution,
        steps: List[TraceStep],
        *,
        trace: bool,
        depth: int,
        max_depth: int,
    ) -> Iterator[Substitution]:
        if not goals:
            yield subst
            return

        first, rest = goals[0], goals[1:]
        for s2 in self._prove(first, subst, steps, trace=trace, depth=depth, max_depth=max_depth):
            yield from self._prove_all(rest, s2, steps, trace=trace, depth=depth, max_depth=max_depth)


def format_subst(subst: Substitution) -> str:
    if not subst:
        return "{}"
    parts = []
    for k, v in subst.items():
        parts.append(f"{k}={v}")
    return "{ " + ", ".join(parts) + " }"

