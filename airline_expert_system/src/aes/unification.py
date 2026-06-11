from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .terms import Const, Predicate, Term, Var


Substitution = Dict[Var, Term]


def apply_subst_term(t: Term, subst: Substitution) -> Term:
    # Follow chains: ?X -> ?Y -> C1
    while isinstance(t, Var) and t in subst:
        t = subst[t]
    return t


def apply_subst_pred(p: Predicate, subst: Substitution) -> Predicate:
    return Predicate(p.name, tuple(apply_subst_term(a, subst) for a in p.args))


def occurs_check(v: Var, t: Term, subst: Substitution) -> bool:
    """
    Occurs check prevents infinite bindings like ?X = f(?X).
    Our terms are just Var/Const (no function symbols), so this is simple.
    """

    t = apply_subst_term(t, subst)
    return v == t


def unify_terms(t1: Term, t2: Term, subst: Substitution) -> Optional[Substitution]:
    t1 = apply_subst_term(t1, subst)
    t2 = apply_subst_term(t2, subst)

    if t1 == t2:
        return subst

    if isinstance(t1, Var):
        if occurs_check(t1, t2, subst):
            return None
        subst[t1] = t2
        return subst

    if isinstance(t2, Var):
        if occurs_check(t2, t1, subst):
            return None
        subst[t2] = t1
        return subst

    # Both constants, different -> cannot unify
    if isinstance(t1, Const) and isinstance(t2, Const):
        return None

    return None


def unify(p1: Predicate, p2: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two predicates (same name, same arity) and return a substitution map if possible.

    Example:
      unify(destination(?C, DEL), destination(C1, DEL)) -> { ?C = C1 }
    """

    if p1.name != p2.name or len(p1.args) != len(p2.args):
        return None

    subst = dict(subst or {})
    for a, b in zip(p1.args, p2.args):
        subst = unify_terms(a, b, subst)
        if subst is None:
            return None
    return subst


@dataclass(frozen=True)
class UnificationResult:
    ok: bool
    subst: Substitution
    reason: str = ""


def unify_explain(p1: Predicate, p2: Predicate) -> UnificationResult:
    s = unify(p1, p2)
    if s is None:
        return UnificationResult(ok=False, subst={}, reason="Predicate mismatch or constants conflict.")
    return UnificationResult(ok=True, subst=s, reason="Unified successfully.")

