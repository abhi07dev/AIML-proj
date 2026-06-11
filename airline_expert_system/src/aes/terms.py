from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Union


@dataclass(frozen=True)
class Var:
    """
    FOL variable.

    We use a leading '?' by convention, e.g. Var("?C"), Var("?F").
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name.startswith("?"):
            raise ValueError("Variable names must start with '?' (example: '?X').")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    """
    Constant symbol (an object identifier), e.g. Const("C1"), Const("F101"), Const("DEL").
    """

    name: str

    def __str__(self) -> str:
        return self.name


Term = Union[Var, Const]


@dataclass(frozen=True)
class Predicate:
    """
    Predicate(atom) in an FOL-like form: name(arg1, arg2, ...)

    Examples:
      - cargo(C1)
      - destination(C1, DEL)
      - can_assign(C1, F101)
    """

    name: str
    args: Tuple[Term, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return f"{self.name}()"
        inside = ", ".join(str(a) for a in self.args)
        return f"{self.name}({inside})"

    @staticmethod
    def of(name: str, *args: Term) -> "Predicate":
        return Predicate(name=name, args=tuple(args))

    def is_negated(self) -> bool:
        # Simplified negation convention: not_predicateName(...)
        return self.name.startswith("not_")

    def negate(self) -> "Predicate":
        if self.is_negated():
            return Predicate(self.name.removeprefix("not_"), self.args)
        return Predicate("not_" + self.name, self.args)


def const(x: str) -> Const:
    return Const(x)


def var(x: str) -> Var:
    return Var(x)


def pred(name: str, *args: Term) -> Predicate:
    return Predicate.of(name, *args)


def terms_to_str(ts: Iterable[Term]) -> str:
    return ", ".join(str(t) for t in ts)

