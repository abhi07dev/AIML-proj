from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Ensure imports work when running: python web/server.py
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aes.inference import InferenceEngine, format_subst  # noqa: E402
from src.aes.sample_data import build_sample_world  # noqa: E402
from src.aes.scheduler import schedule_cargo  # noqa: E402
from src.aes.terms import Predicate, const, pred, var  # noqa: E402
from src.aes.unification import unify_explain  # noqa: E402


def _parse_simple_predicate(text: str) -> Predicate | None:
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


def _json(handler: BaseHTTPRequestHandler, status: int, obj: object) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _file_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def _content_type(path: str) -> str:
    if path.endswith(".html"):
        return "text/html; charset=utf-8"
    if path.endswith(".css"):
        return "text/css; charset=utf-8"
    if path.endswith(".js"):
        return "text/javascript; charset=utf-8"
    if path.endswith(".svg"):
        return "image/svg+xml"
    return "application/octet-stream"


class AppState:
    def __init__(self) -> None:
        self.kb, self.flights, self.cargo = build_sample_world()
        self.engine = InferenceEngine(self.kb)
        self.engine.forward_chain(trace=False)

    def snapshot(self) -> dict:
        # Only expose the user-friendly fields
        flights = []
        for f in self.flights.values():
            flights.append(
                {
                    "fid": f.fid,
                    "origin": f.origin,
                    "destination": f.destination,
                    "capacity_kg": f.capacity_kg,
                    "depart_hour": f.depart_hour,
                    "on_time": self.kb.has_fact(pred("on_time", const(f.fid))),
                    "delayed": self.kb.has_fact(pred("delayed", const(f.fid))),
                    "remaining_capacity_kg": _get_remaining_capacity(self.kb, f.fid, f.capacity_kg),
                }
            )

        cargo = []
        for c in self.cargo.values():
            cargo.append(
                {
                    "cid": c.cid,
                    "destination": c.destination,
                    "weight_kg": c.weight_kg,
                    "priority": c.priority,
                    "deadline_hour": c.deadline_hour,
                }
            )

        return {"flights": flights, "cargo": cargo}


def _get_remaining_capacity(kb, fid: str, fallback: int) -> int:
    for f in kb.facts():
        if f.name == "remaining_capacity_kg" and len(f.args) == 2 and str(f.args[0]) == fid:
            try:
                return int(str(f.args[1]))
            except ValueError:
                return fallback
    return fallback


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        # Keep console clean for beginners (comment this out if you want logs)
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            STATE.engine.forward_chain(trace=False)
            return _json(self, 200, STATE.snapshot())

        if path == "/":
            path = "/static/index.html"

        if path.startswith("/static/"):
            fs_path = os.path.join(HERE, path.lstrip("/").replace("/", os.sep))
            data = _file_bytes(fs_path)
            if data is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _content_type(fs_path))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = _read_json(self)

        if path == "/api/add_cargo":
            cid = str(payload.get("cid", "")).strip()
            dest = str(payload.get("destination", "")).strip().upper()
            weight = int(payload.get("weight_kg", 0) or 0)
            priority = str(payload.get("priority", "normal")).strip().lower()
            deadline = int(payload.get("deadline_hour", 0) or 0)

            if not cid or cid in STATE.cargo:
                return _json(self, 400, {"ok": False, "error": "Invalid or duplicate cargo id."})
            if not dest or weight <= 0 or deadline <= 0:
                return _json(self, 400, {"ok": False, "error": "Destination/weight/deadline are required."})
            if priority not in {"urgent", "high", "normal"}:
                priority = "normal"

            # Update object model
            from src.aes.models import Cargo  # local import to keep top simple

            c = Cargo(cid, destination=dest, weight_kg=weight, priority=priority, deadline_hour=deadline)
            STATE.cargo[cid] = c

            # Update KB facts
            STATE.kb.add_fact(pred("cargo", const(cid)))
            STATE.kb.add_fact(pred("instance_of", const(cid), const("Cargo")))
            STATE.kb.add_fact(pred("cargo_dest", const(cid), const(dest)))
            STATE.kb.add_fact(pred("cargo_weight_kg", const(cid), const(str(weight))))
            STATE.kb.add_fact(pred("cargo_deadline_hour", const(cid), const(str(deadline))))
            STATE.kb.add_fact(pred("cargo_priority", const(cid), const(priority)))
            STATE.engine.forward_chain(trace=False)

            return _json(self, 200, {"ok": True, "state": STATE.snapshot()})

        if path == "/api/weather":
            airport = str(payload.get("airport", "")).strip().upper()
            condition = str(payload.get("condition", "")).strip().lower()
            if airport == "" or condition not in {"clear", "storm"}:
                return _json(self, 400, {"ok": False, "error": "Use airport + condition(clear/storm)."})

            # remove previous weather facts for that airport
            for f in list(STATE.kb.facts()):
                if f.name == "weather" and len(f.args) == 2 and str(f.args[0]) == airport:
                    STATE.kb.remove_fact(f)
            STATE.kb.add_fact(pred("weather", const(airport), const(condition)), why="user updated weather (web)")
            STATE.engine.forward_chain(trace=False)
            return _json(self, 200, {"ok": True, "state": STATE.snapshot()})

        if path == "/api/delay":
            fid = str(payload.get("flight_id", "")).strip().upper()
            reason = str(payload.get("reason", "unknown")).strip().lower()
            if fid == "" or fid not in STATE.flights:
                return _json(self, 400, {"ok": False, "error": "Unknown flight id."})
            STATE.kb.add_fact(pred("event_delay", const(fid), const(reason)), why="delay event (web)")
            STATE.engine.forward_chain(trace=False)
            return _json(self, 200, {"ok": True, "state": STATE.snapshot()})

        if path == "/api/schedule":
            explanation = bool(payload.get("explanation", True))
            assignments, trace = schedule_cargo(STATE.kb, STATE.flights, STATE.cargo, explanation=explanation)
            STATE.engine.forward_chain(trace=False)

            return _json(
                self,
                200,
                {
                    "ok": True,
                    "assignments": [{"cargo_id": a.cargo_id, "flight_id": a.flight_id} for a in assignments],
                    "trace": [{"kind": t.kind, "message": t.message} for t in trace],
                    "state": STATE.snapshot(),
                },
            )

        if path == "/api/query":
            q = str(payload.get("query", "")).strip()
            p = _parse_simple_predicate(q)
            if p is None:
                return _json(self, 400, {"ok": False, "error": "Bad query format. Example: assigned(C1, ?F)"})

            answers, steps = STATE.engine.ask(p, trace=True)
            return _json(
                self,
                200,
                {
                    "ok": True,
                    "answers": [format_subst(s) for s in answers[:20]],
                    "trace": [{"kind": t.kind, "message": t.message} for t in steps],
                },
            )

        if path == "/api/unify":
            a = _parse_simple_predicate(str(payload.get("a", "")))
            b = _parse_simple_predicate(str(payload.get("b", "")))
            if a is None or b is None:
                return _json(self, 400, {"ok": False, "error": "Bad predicate format."})
            res = unify_explain(a, b)
            return _json(
                self,
                200,
                {"ok": res.ok, "reason": res.reason, "substitution": format_subst(res.subst)},
            )

        return _json(self, 404, {"ok": False, "error": "Unknown endpoint."})


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Web UI running at http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

