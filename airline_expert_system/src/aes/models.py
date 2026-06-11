from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Flight:
    fid: str
    origin: str
    destination: str
    capacity_kg: int
    depart_hour: int


@dataclass
class Cargo:
    cid: str
    destination: str
    weight_kg: int
    priority: str  # "urgent" | "high" | "normal"
    deadline_hour: int  # simple integer time

