from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .inference import InferenceEngine, TraceStep, format_subst
from .kb import KnowledgeBase
from .models import Cargo, Flight
from .sample_data import build_sample_world
from .scheduler import schedule_cargo
from .terms import Predicate, const, pred, var
from .unification import unify_explain


def _line() -> str:
    return "-" * 78


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = " | ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def _show_flights(kb: KnowledgeBase, flights: Dict[str, Flight]) -> None:
    rows = []
    for f in flights.values():
        # derived/default facts:
        on_time = kb.has_fact(pred("on_time", const(f.fid)))
        delayed = kb.has_fact(pred("delayed", const(f.fid)))
        rem = _get_remaining_capacity(kb, f.fid, f.capacity_kg)
        rows.append([f.fid, f.origin, f.destination, str(f.depart_hour), str(f.capacity_kg), str(rem), "YES" if on_time else "NO", "YES" if delayed else "NO"])
    _print_table(
        ["Flight", "From", "To", "Dep(h)", "Cap(kg)", "Rem(kg)", "OnTime", "Delayed"],
        rows,
    )


def _show_cargo(cargo: Dict[str, Cargo]) -> None:
    rows = []
    for c in cargo.values():
        rows.append([c.cid, c.destination, str(c.weight_kg), c.priority, str(c.deadline_hour)])
    _print_table(["Cargo", "Dest", "Weight(kg)", "Priority", "Deadline(h)"], rows)


def _get_remaining_capacity(kb: KnowledgeBase, fid: str, fallback: int) -> int:
    for f in kb.facts():
        if f.name == "remaining_capacity_kg" and len(f.args) == 2 and str(f.args[0]) == fid:
            try:
                return int(str(f.args[1]))
            except ValueError:
                return fallback
    return fallback


def _add_cargo(kb: KnowledgeBase, cargo: Dict[str, Cargo]) -> None:
    cid = input("Cargo id (e.g., C4): ").strip()
    if not cid:
        return
    if cid in cargo:
        print("Cargo id already exists.")
        return

    dest = input("Destination airport code (e.g., DEL): ").strip().upper()
    weight = int(input("Weight kg (e.g., 800): ").strip())
    priority = input("Priority (urgent/high/normal): ").strip().lower()
    deadline = int(input("Deadline hour (e.g., 17): ").strip())

    c = Cargo(cid, destination=dest, weight_kg=weight, priority=priority, deadline_hour=deadline)
    cargo[cid] = c

    kb.add_fact(pred("cargo", const(cid)))
    kb.add_fact(pred("instance_of", const(cid), const("Cargo")))
    kb.add_fact(pred("cargo_dest", const(cid), const(dest)))
    kb.add_fact(pred("cargo_weight_kg", const(cid), const(str(weight))))
    kb.add_fact(pred("cargo_deadline_hour", const(cid), const(str(deadline))))
    kb.add_fact(pred("cargo_priority", const(cid), const(priority)))

    print(f"Added cargo {cid}.")


def _set_weather(kb: KnowledgeBase) -> None:
    airport = input("Airport (e.g., DEL): ").strip().upper()
    condition = input("Condition (clear/storm): ").strip().lower()

    # remove previous weather facts for that airport
    for f in list(kb.facts()):
        if f.name == "weather" and len(f.args) == 2 and str(f.args[0]) == airport:
            kb.remove_fact(f)
    kb.add_fact(pred("weather", const(airport), const(condition)), why="user updated weather")
    print(f"Weather updated: weather({airport}, {condition}).")


def _delay_event(kb: KnowledgeBase) -> None:
    fid = input("Flight id to delay (e.g., F102): ").strip().upper()
    reason = input("Reason (e.g., storm/runway): ").strip().lower() or "unknown"
    kb.add_fact(pred("event_delay", const(fid), const(reason)), why="user delay event")
    print(f"Event recorded: event_delay({fid}, {reason}).")


def _run_scheduling(kb: KnowledgeBase, flights: Dict[str, Flight], cargo: Dict[str, Cargo]) -> None:
    explanation = input("Explanation/trace mode? (y/n) ").strip().lower().startswith("y")
    assignments, trace = schedule_cargo(kb, flights, cargo, explanation=explanation)

    if explanation:
        print(_line())
        for s in trace:
            tag = {"default": "[Default]", "rule": "[Rule]", "conflict": "[Conflict]", "query": "[Query]"}.get(s.kind, "[Info]")
            print(f"{tag} {s.message}")
        print(_line())

    if not assignments:
        print("No assignments could be made.")
        return

    print("Final assignments:")
    rows = [[a.cargo_id, a.flight_id] for a in assignments]
    _print_table(["Cargo", "Assigned Flight"], rows)


def _query(kb: KnowledgeBase) -> None:
    """
    Beginner-friendly query interface:
      - Ask: assigned(C1, ?F)
      - Or: can_assign(C1, ?F)
    """

    q = input("Enter query like assigned(C1, ?F): ").strip()
    if not q:
        return

    parsed = _parse_simple_predicate(q)
    if parsed is None:
        print("Could not parse. Use format: predicate(arg1, arg2, ...). Variables start with '?'.")
        return

    engine = InferenceEngine(kb)
    answers, steps = engine.ask(parsed, trace=True)

    print(_line())
    for s in steps:
        tag = {"query": "[Query]", "rule": "[Rule]", "default": "[Default]", "conflict": "[Conflict]"}.get(s.kind, "[Info]")
        print(f"{tag} {s.message}")
    print(_line())

    if not answers:
        print("No proof found.")
        return

    print("Answers (substitutions):")
    for s in answers[:10]:
        print(" ", format_subst(s))


def _unification_demo() -> None:
    """
    Small interactive demo: shows how unification works.
    """

    print("Unification demo.")
    print("Example: unify destination(?X, DEL) with destination(C1, DEL) gives {?X=C1}")
    a = input("Predicate A (e.g., destination(?X, DEL)): ").strip()
    b = input("Predicate B (e.g., destination(C1, DEL)): ").strip()
    p1 = _parse_simple_predicate(a)
    p2 = _parse_simple_predicate(b)
    if p1 is None or p2 is None:
        print("Parse failed.")
        return
    res = unify_explain(p1, p2)
    if not res.ok:
        print("Unification failed:", res.reason)
        return
    print("Unification succeeded:", res.reason)
    print("Substitution:", format_subst(res.subst))


def _parse_simple_predicate(text: str) -> Optional[Predicate]:
    """
    Very small parser for inputs like:
      assigned(C1, ?F)
      on_time(F101)

    This is only for CLI convenience (not a full parser).
    """

    text = text.strip()
    if "(" not in text or not text.endswith(")"):
        return None
    name, rest = text.split("(", 1)
    name = name.strip()
    inside = rest[:-1].strip()
    if not name:
        return None
    if inside == "":
        return pred(name)
    parts = [p.strip() for p in inside.split(",")]
    args = []
    for p in parts:
        if p.startswith("?"):
            args.append(var(p))
        else:
            args.append(const(p))
    return Predicate(name, tuple(args))


def run_cli() -> None:
    kb, flights, cargo = build_sample_world()
    engine = InferenceEngine(kb)

    # prime defaults / simple derived facts
    engine.forward_chain(trace=False)

    while True:
        print("\n" + _line())
        print("Airline Scheduling and Cargo Management Expert System")
        print(_line())
        print("1) View flights")
        print("2) View cargo")
        print("3) Add cargo")
        print("4) Set weather (clear/storm)")
        print("5) Add delay event for flight")
        print("6) Run scheduling inference (forward chaining + greedy assignment)")
        print("7) Query system (backward chaining)")
        print("8) Unification demo")
        print("0) Exit")
        choice = input("Select: ").strip()

        if choice == "1":
            engine.forward_chain(trace=False)
            _show_flights(kb, flights)
        elif choice == "2":
            _show_cargo(cargo)
        elif choice == "3":
            _add_cargo(kb, cargo)
        elif choice == "4":
            _set_weather(kb)
            engine.forward_chain(trace=False)
        elif choice == "5":
            _delay_event(kb)
            engine.forward_chain(trace=False)
        elif choice == "6":
            _run_scheduling(kb, flights, cargo)
        elif choice == "7":
            _query(kb)
        elif choice == "8":
            _unification_demo()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")

