## Airline Scheduling and Cargo Management Expert System (Rule-Based AI)

This is a **beginner-friendly, college-level (SPPU) AI assignment project** that implements a small **rule-based Expert System** for airline **cargo scheduling**.

It demonstrates core AI concepts:

- **First-Order Logic (FOL)**-style knowledge representation (predicates, variables, constants)
- **Propositional vs FOL** (same engine, different expressiveness)
- **Unification** (variable binding for predicate matching)
- **Forward chaining** (data-driven inference)
- **Backward chaining** (goal-driven inference / query answering)
- **Resolution-style conflict checking** (simplified contradiction detection)
- **Ontology basics** (categories/classes + relations like `is_a`, `operates_between`)
- **Events** (e.g., a delay event changes what can be scheduled)
- **Default reasoning** (e.g., flights are on time unless delayed)
- **Trace / explanation mode** (shows rules used and why)

The system uses **pure Python (no external libraries)**.

### Project structure

- `src/aes/terms.py`: Terms (`Var`, `Const`) and `Predicate` (FOL-like atoms)
- `src/aes/unification.py`: Unification algorithm (for FOL predicate matching)
- `src/aes/kb.py`: Knowledge Base (facts + rules) and default handling
- `src/aes/inference.py`: Forward chaining + backward chaining + conflict checking
- `src/aes/sample_data.py`: Sample flights/cargo/weather + rules (the “expert knowledge”)
- `src/aes/cli.py`: Command-line interface
- `src/main.py`: Entry point
- `docs/AI_CONCEPTS.md`: Short theory notes (forward/backward, FOL vs propositional, unification)
- `docs/PROJECT_EXPLANATION.txt`: Simple full explanation of concepts + working

### How to run

From the `airline_expert_system` folder:

```bash
python -m src.main
```

### Frontend (simple web UI)

Run the local web server:

```bash
python web/server.py
```

Then open the shown URL in your browser (usually `http://127.0.0.1:8000/`).

### Example actions to try

1) View flights and cargo
2) Run **scheduling inference** (forward chaining)
3) Query: “Which flight will carry cargo C1?” (backward chaining)
4) Trigger an event: delay a flight and re-run scheduling

### Example (short) output (will vary)

```
> 3) Run scheduling
[ForwardChain] Derived: on_time(F101) by default
[ForwardChain] Rule applied: R_can_assign_basic
  because cargo(C1) and flight(F101) match and constraints satisfied
[ForwardChain] Derived: can_assign(C1, F101)
[ForwardChain] Rule applied: R_assign_greedy
[ForwardChain] Derived: assigned(C1, F101)

Final assignments:
  C1 -> F101
  C2 -> F102

> 4) Query cargo assignment
Query: assigned(C1, ?F)
Answer:
  ?F = F101
Proof (backward chaining):
  assigned(C1, F101) was found as a fact (derived earlier)
```

### Notes

- This is **not** a full industrial optimizer; it is a **teaching system**: simple rules, clear traces.
- Facts and rules are stored in a form similar to Horn clauses:
  - `premise1 ∧ premise2 ∧ ... -> conclusion`

