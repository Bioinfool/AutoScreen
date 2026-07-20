"""ObjectiveSchema: expensive (AL targets) vs static (filters / constraints)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Kind = Literal["expensive", "static"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    kind: Kind = "expensive"
    maximize: bool = True
    source: str | None = None  # e.g. "dock", "qed", "sa", "moo:col"

    def to_maximize(self, raw_value: float) -> float:
        """Map raw executor/oracle units onto the maximize convention used by AL."""
        v = float(raw_value)
        return v if self.maximize else -v

    def from_maximize(self, model_value: float) -> float:
        v = float(model_value)
        return v if self.maximize else -v


@dataclass
class ObjectiveSchema:
    """Defines what the surrogate optimizes vs what is known a priori."""

    expensive: list[ObjectiveSpec] = field(default_factory=list)
    static: list[ObjectiveSpec] = field(default_factory=list)

    @property
    def n_expensive(self) -> int:
        return len(self.expensive)

    @property
    def n_static(self) -> int:
        return len(self.static)

    @property
    def expensive_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.expensive)

    @property
    def static_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.static)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expensive": [o.__dict__ for o in self.expensive],
            "static": [o.__dict__ for o in self.static],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectiveSchema":
        def _as_spec(x: Any, default_kind: Kind) -> ObjectiveSpec:
            if isinstance(x, ObjectiveSpec):
                return x
            if isinstance(x, dict):
                return ObjectiveSpec(
                    name=str(x["name"]),
                    kind=x.get("kind", default_kind),
                    maximize=bool(x.get("maximize", True)),
                    source=x.get("source"),
                )
            raise TypeError(f"Invalid objective entry: {x!r}")

        return cls(
            expensive=[_as_spec(x, "expensive") for x in d.get("expensive", [])],
            static=[_as_spec(x, "static") for x in d.get("static", [])],
        )



def default_schema() -> ObjectiveSchema:
    """Canonical AutoScreen schema: AL on activity; QED/SA are static filters."""
    return ObjectiveSchema(
        expensive=[ObjectiveSpec("activity", kind="expensive", maximize=True, source="dock")],
        static=[
            ObjectiveSpec("qed", kind="static", maximize=True, source="qed"),
            ObjectiveSpec("sa_ease", kind="static", maximize=True, source="sa"),
        ],
    )


def parse_objective_schema(cfg: dict[str, Any] | None) -> ObjectiveSchema:
    """Parse YAML objectives / static_properties blocks.

    Accepted forms:
      objectives: [activity]                         # names → expensive
      objectives: [{name: activity, kind: expensive}]
      static_properties: [qed, sa_ease]
      objectives: [activity, qed, sa_ease]            # legacy: qed/sa → static
    """
    if not cfg:
        return default_schema()

    raw_obj = cfg.get("objectives")
    raw_static = cfg.get("static_properties")

    # Structured dict form: {expensive: [...], static: [...]}
    if isinstance(raw_obj, dict) and ("expensive" in raw_obj or "static" in raw_obj):
        return ObjectiveSchema.from_dict(
            {
                "expensive": list(raw_obj.get("expensive") or []),
                "static": list(raw_obj.get("static") or []),
            }
        )

    expensive: list[ObjectiveSpec] = []
    static: list[ObjectiveSpec] = []

    if raw_obj is None and raw_static is None:
        return default_schema()

    legacy_static_names = {"qed", "sa_ease", "sa", "mw", "logp", "pains"}
    for item in raw_obj or []:
        spec = _coerce_spec(item, default_kind="expensive")
        if spec.kind == "static" or (
            isinstance(item, str) and item.lower() in legacy_static_names
        ):
            static.append(
                ObjectiveSpec(spec.name, kind="static", maximize=spec.maximize, source=spec.source)
            )
        else:
            expensive.append(
                ObjectiveSpec(
                    spec.name, kind="expensive", maximize=spec.maximize, source=spec.source
                )
            )

    for item in raw_static or []:
        static.append(_coerce_spec(item, default_kind="static"))

    if not expensive:
        expensive = [ObjectiveSpec("activity", kind="expensive", maximize=True, source="dock")]
    if not static and raw_static is None and not any(
        s.name in legacy_static_names for s in static
    ):
        # Keep default static filters unless user explicitly set empty list
        if raw_static is None and raw_obj is not None:
            # legacy list that already moved qed/sa into static above
            pass
        if not static:
            static = list(default_schema().static)

    return ObjectiveSchema(expensive=expensive, static=static)


def _coerce_spec(item: Any, default_kind: Kind = "expensive") -> ObjectiveSpec:
    if isinstance(item, str):
        return ObjectiveSpec(name=item, kind=default_kind)
    if isinstance(item, dict):
        return ObjectiveSpec(
            name=str(item["name"]),
            kind=item.get("kind", default_kind),
            maximize=bool(item.get("maximize", True)),
            source=item.get("source"),
        )
    raise TypeError(f"Invalid objective entry: {item!r}")
