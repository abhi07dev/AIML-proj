## Short AI theory notes (for viva / report)

### Propositional Logic vs First-Order Logic (FOL)

- **Propositional logic** uses atomic statements with no internal structure, e.g.:
  - `P` = “Flight F101 is delayed”
  - You can combine them with connectives: \(P \land Q \rightarrow R\)
- **FOL** adds **objects** and **relations** using predicates, e.g.:
  - `delayed(F101)`
  - `cargo(C1)` and `destination(C1, DEL)`
  - It supports **variables** like `?X` to express general rules:
    - `cargo(?C) ∧ urgent(?C) -> high_priority(?C)`

In this project, facts look like FOL predicates (e.g., `cargo(C1)`, `destination(C1, DEL)`), and rules use variables (e.g., `?C`, `?F`).

### Unification (very important in FOL)

**Unification** is the process of finding variable substitutions that make two predicates match.

Example:

- Goal: `destination(?C, DEL)`
- Fact: `destination(C1, DEL)`

Unification result: `{ ?C = C1 }`

This lets one rule work for many objects (all cargo items, all flights).

### Forward Chaining vs Backward Chaining

- **Forward chaining** (data-driven):
  - Start with known facts.
  - Apply rules to **derive new facts**.
  - Continue until no new facts can be derived.
  - Good for: “Generate a schedule / compute all consequences.”

- **Backward chaining** (goal-driven):
  - Start with a query/goal like `assigned(C1, ?F)`.
  - Try to prove it using facts and rules (sub-goals).
  - Good for: “Answer a question about the knowledge base.”

This project implements both and prints a trace.

### Resolution-based inference (simplified here)

Full resolution is usually shown with CNF and clause refutation.
For simplicity, this project includes **conflict checking** in a resolution spirit:

- If the system derives both `on_time(F101)` and `delayed(F101)`, that is a contradiction.
- The engine reports the conflict and prefers the **exception** (`delayed`) over the **default** (`on_time`).

### Default reasoning

Human experts often assume defaults:

- “Flights are on time **unless** we know they are delayed.”

This is **non-monotonic** (adding new facts can invalidate old conclusions).
We simulate it by:

- Adding `on_time(?F)` by default for flights
- Removing/overriding it when `delayed(?F)` becomes true (e.g., due to a weather event)

