from __future__ import annotations

from typing import Dict, List, Tuple

from .kb import KnowledgeBase, Rule
from .models import Cargo, Flight
from .terms import const, pred, var


def build_sample_world() -> Tuple[KnowledgeBase, Dict[str, Flight], Dict[str, Cargo]]:
    """
    Create a small sample dataset + rules.

    Facts are represented as FOL-like predicates so that:
      - forward chaining can derive schedule-related facts
      - backward chaining can answer queries
    """

    kb = KnowledgeBase()

    # ---------------- Ontology (very small) ----------------
    # Category/object representation
    kb.add_fact(pred("class", const("Aircraft")))
    kb.add_fact(pred("class", const("Cargo")))
    kb.add_fact(pred("class", const("Airport")))
    kb.add_fact(pred("class", const("WeatherCondition")))

    kb.add_fact(pred("subclass_of", const("Flight"), const("Aircraft")))
    kb.add_fact(pred("subclass_of", const("WideBody"), const("Aircraft")))
    kb.add_fact(pred("subclass_of", const("NarrowBody"), const("Aircraft")))

    # ---------------- Flights ----------------
    flights: Dict[str, Flight] = {
        "F101": Flight("F101", origin="PUNE", destination="DEL", capacity_kg=5000, depart_hour=14),
        "F102": Flight("F102", origin="PUNE", destination="BOM", capacity_kg=2500, depart_hour=12),
        "F103": Flight("F103", origin="PUNE", destination="DEL", capacity_kg=1500, depart_hour=18),
    }

    for f in flights.values():
        kb.add_fact(pred("flight", const(f.fid)))
        kb.add_fact(pred("instance_of", const(f.fid), const("Flight")))
        kb.add_fact(pred("origin", const(f.fid), const(f.origin)))
        kb.add_fact(pred("destination", const(f.fid), const(f.destination)))
        kb.add_fact(pred("capacity_kg", const(f.fid), const(str(f.capacity_kg))))
        kb.add_fact(pred("depart_hour", const(f.fid), const(str(f.depart_hour))))
        kb.add_fact(pred("remaining_capacity_kg", const(f.fid), const(str(f.capacity_kg))), why="initial remaining capacity = capacity")

        kb.add_fact(pred("operates_between", const(f.fid), const(f.origin), const(f.destination)))

    # ---------------- Cargo ----------------
    cargo: Dict[str, Cargo] = {
        "C1": Cargo("C1", destination="DEL", weight_kg=1200, priority="urgent", deadline_hour=16),
        "C2": Cargo("C2", destination="BOM", weight_kg=900, priority="normal", deadline_hour=15),
        "C3": Cargo("C3", destination="DEL", weight_kg=2000, priority="high", deadline_hour=19),
    }

    for c in cargo.values():
        kb.add_fact(pred("cargo", const(c.cid)))
        kb.add_fact(pred("instance_of", const(c.cid), const("Cargo")))
        kb.add_fact(pred("cargo_dest", const(c.cid), const(c.destination)))
        kb.add_fact(pred("cargo_weight_kg", const(c.cid), const(str(c.weight_kg))))
        kb.add_fact(pred("cargo_deadline_hour", const(c.cid), const(str(c.deadline_hour))))
        kb.add_fact(pred("cargo_priority", const(c.cid), const(c.priority)))

    # ---------------- Weather / events ----------------
    # Default assumption: flights are on time (added by KB defaults) unless delayed(...)
    kb.add_fact(pred("weather", const("DEL"), const("clear")))
    kb.add_fact(pred("weather", const("BOM"), const("clear")))

    # ---------------- Expert rules (Horn-style) ----------------
    # (1) Same destination relation
    kb.add_rule(
        Rule(
            name="R_same_destination",
            premises=(
                pred("cargo_dest", var("?C"), var("?D")),
                pred("destination", var("?F"), var("?D")),
            ),
            conclusion=pred("same_destination", var("?C"), var("?F")),
        )
    )

    # (2) If destination has storm, it is bad weather.
    kb.add_rule(
        Rule(
            name="R_storm_is_bad_weather",
            premises=(pred("weather", var("?A"), const("storm")),),
            conclusion=pred("bad_weather", var("?A")),
        )
    )

    # (3) If flight goes to bad weather airport, flight has weather risk.
    kb.add_rule(
        Rule(
            name="R_weather_risk_for_flight",
            premises=(
                pred("destination", var("?F"), var("?A")),
                pred("bad_weather", var("?A")),
            ),
            conclusion=pred("weather_risk", var("?F")),
        )
    )

    # (4) An explicit delay event implies delayed flight (events and reasoning).
    kb.add_rule(
        Rule(
            name="R_delay_event_causes_delay",
            premises=(pred("event_delay", var("?F"), var("?Reason")),),
            conclusion=pred("delayed", var("?F")),
        )
    )

    # (5) If flight has weather risk, we assume delay (simple expert heuristic).
    kb.add_rule(
        Rule(
            name="R_weather_risk_causes_delay",
            premises=(pred("weather_risk", var("?F")),),
            conclusion=pred("delayed", var("?F")),
        )
    )

    # (5b) If delayed, then "not_on_time" (negation-style predicate).
    # This supports simplified contradiction checking:
    #   on_time(F) AND not_on_time(F)  => conflict (resolution-style check).
    kb.add_rule(
        Rule(
            name="R_delayed_implies_not_on_time",
            premises=(pred("delayed", var("?F")),),
            conclusion=pred("not_on_time", var("?F")),
        )
    )

    # (6) Deadline feasibility: if depart <= deadline then meets_deadline(C,F)
    # We will inject facts meets_deadline(...) procedurally (because numeric compare).
    #
    # Same for fits_in(...) based on remaining capacity.

    # (7) Main eligibility rule (FOL-style): can_assign(C,F) if all constraints hold.
    kb.add_rule(
        Rule(
            name="R_can_assign_basic",
            premises=(
                pred("cargo", var("?C")),
                pred("flight", var("?F")),
                pred("same_destination", var("?C"), var("?F")),
                pred("fits_in", var("?C"), var("?F")),
                pred("meets_deadline", var("?C"), var("?F")),
                pred("on_time", var("?F")),
            ),
            conclusion=pred("can_assign", var("?C"), var("?F")),
        )
    )

    # (8) If cargo is urgent then treat it as high priority (mental object simulation).
    # We simulate a belief-like predicate: believes(agent, urgent(C)) -> treat_as_high(C)
    kb.add_fact(pred("agent", const("dispatcherAI")))
    kb.add_rule(
        Rule(
            name="R_urgent_is_high_priority",
            premises=(
                pred("cargo_priority", var("?C"), const("urgent")),
                pred("agent", const("dispatcherAI")),
            ),
            conclusion=pred("treat_as_high", var("?C")),
        )
    )

    # (9) Default-like policy (but as an explicit rule): if treat_as_high(C) then prefer(C)
    kb.add_rule(
        Rule(
            name="R_prefer_high",
            premises=(pred("treat_as_high", var("?C")),),
            conclusion=pred("preferred_cargo", var("?C")),
        )
    )

    return kb, flights, cargo

