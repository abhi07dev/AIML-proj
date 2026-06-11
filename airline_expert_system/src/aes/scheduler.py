from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .inference import InferenceEngine, TraceStep
from .kb import KnowledgeBase
from .models import Cargo, Flight
from .terms import Const, Predicate, const, pred


@dataclass(frozen=True)
class Assignment:
    cargo_id: str
    flight_id: str


def _get_int_fact(kb: KnowledgeBase, name: str, a1: str) -> Optional[int]:
    for f in kb.facts():
        if f.name == name and len(f.args) == 2 and str(f.args[0]) == a1:
            try:
                return int(str(f.args[1]))
            except ValueError:
                return None
    return None


def _set_int_fact(kb: KnowledgeBase, name: str, a1: str, value: int, *, why: str = "") -> None:
    # remove old
    old = []
    for f in kb.facts():
        if f.name == name and len(f.args) == 2 and str(f.args[0]) == a1:
            old.append(f)
    for f in old:
        kb.remove_fact(f)
    kb.add_fact(pred(name, const(a1), const(str(value))), why=why)


def inject_numeric_constraints(kb: KnowledgeBase, flights: Dict[str, Flight], cargo: Dict[str, Cargo]) -> List[TraceStep]:
    """
    Because our toy FOL engine doesn't implement arithmetic, we inject two
    *environment/perception* facts based on numbers:

    - fits_in(C,F): cargo weight <= remaining capacity of F
    - meets_deadline(C,F): flight departs before cargo deadline
    """

    steps: List[TraceStep] = []

    # Clear old injected facts
    for f in list(kb.facts()):
        if f.name in {"fits_in", "meets_deadline"}:
            kb.remove_fact(f)

    for c in cargo.values():
        for fl in flights.values():
            rem = _get_int_fact(kb, "remaining_capacity_kg", fl.fid)
            if rem is None:
                rem = fl.capacity_kg
            if c.weight_kg <= rem:
                kb.add_fact(pred("fits_in", const(c.cid), const(fl.fid)), why="numeric check: weight <= remaining capacity")
            if fl.depart_hour <= c.deadline_hour:
                kb.add_fact(pred("meets_deadline", const(c.cid), const(fl.fid)), why="numeric check: depart_hour <= deadline_hour")

    steps.append(TraceStep("rule", "Injected numeric constraints: fits_in/ meets_deadline (computed from numbers)."))
    return steps


def schedule_cargo(
    kb: KnowledgeBase,
    flights: Dict[str, Flight],
    cargo: Dict[str, Cargo],
    *,
    explanation: bool = True,
) -> Tuple[List[Assignment], List[TraceStep]]:
    """
    Scheduling loop:

    - inject numeric constraints facts (fits_in, meets_deadline)
    - forward chain to derive can_assign
    - greedily assign cargo by priority/deadline
    - update remaining capacities and repeat
    """

    engine = InferenceEngine(kb)
    trace: List[TraceStep] = []

    def priority_key(p: str) -> int:
        return {"urgent": 0, "high": 1, "normal": 2}.get(p, 3)

    pending = sorted(cargo.values(), key=lambda c: (priority_key(c.priority), c.deadline_hour, c.weight_kg))
    assignments: List[Assignment] = []

    # Reset previous assignment facts
    for f in list(kb.facts()):
        if f.name in {"assigned", "can_assign"}:
            kb.remove_fact(f)

    # Main loop: attempt assignment for each cargo item once (greedy)
    for c in pending:
        trace.extend(inject_numeric_constraints(kb, flights, cargo))
        trace.extend(engine.forward_chain(trace=explanation))

        # find candidate flights from derived can_assign(C, F)
        candidates: List[str] = []
        for f in kb.facts():
            if f.name == "can_assign" and len(f.args) == 2 and str(f.args[0]) == c.cid:
                candidates.append(str(f.args[1]))

        if not candidates:
            if explanation:
                trace.append(TraceStep("rule", f"No feasible flight found for cargo {c.cid}."))
            continue

        # Choose earliest departure flight among candidates (simple heuristic)
        candidates.sort(key=lambda fid: flights[fid].depart_hour)
        chosen = candidates[0]

        # Commit assignment as a fact
        kb.add_fact(pred("assigned", const(c.cid), const(chosen)), why="scheduler: greedy selection among can_assign")
        assignments.append(Assignment(cargo_id=c.cid, flight_id=chosen))

        # Update remaining capacity
        rem = _get_int_fact(kb, "remaining_capacity_kg", chosen)
        if rem is None:
            rem = flights[chosen].capacity_kg
        _set_int_fact(
            kb,
            "remaining_capacity_kg",
            chosen,
            rem - c.weight_kg,
            why=f"updated due to assigned({c.cid}, {chosen})",
        )

        if explanation:
            trace.append(
                TraceStep(
                    "rule",
                    f"Assigned {c.cid} -> {chosen}; remaining_capacity_kg({chosen})={rem - c.weight_kg}",
                )
            )

    return assignments, trace

