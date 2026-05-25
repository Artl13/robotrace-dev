"""Emit a JSON snapshot of the public ``robotrace`` SDK surface.

Walks each module listed in :data:`MODULES`, expands its ``__all__``
(falling back to every non-underscore public name when ``__all__`` is
missing), and records:

  * The kind of each symbol (``function`` / ``class`` / ``constant`` /
    ``module``).
  * For functions and methods: every parameter's name, kind
    (``POSITIONAL_ONLY``, ``POSITIONAL_OR_KEYWORD``, ``KEYWORD_ONLY``,
    ``VAR_POSITIONAL``, ``VAR_KEYWORD``), default-presence,
    annotation-as-string, and the return annotation.
  * For classes: the same per-method recording plus a list of public
    instance / class attributes.
  * For constants: the type's name (e.g. ``str`` / ``int`` / ``Literal``).

The script writes to ``packages/sdk-python/api-surface.json`` next to
``pyproject.toml`` so the freeze guard test (``tests/test_api_surface_freeze.py``)
can diff the live surface against the committed baseline.

Run it from the repo root::

    python packages/sdk-python/scripts/snapshot_api_surface.py

Re-run after any *intentional additive* change to the SDK surface and
commit the updated baseline. Removals or signature narrowings are the
exact thing the diff test catches - don't snapshot through them
without an explicit major bump.

The output format is the documented contract of the freeze guard - if
you change it, update :mod:`tests.test_api_surface_freeze` in the
same commit.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Modules the freeze guards. Adapter modules (``robotrace.adapters.*``)
# are deliberately excluded for now: they are still expected to churn
# during alpha. Add them here once the freeze gate goes green for the
# core surface and you want to extend the contract.
MODULES: tuple[str, ...] = (
    "robotrace",
    "robotrace.types",
    "robotrace.verify",
    "robotrace.evals",
    "robotrace.errors",
    "robotrace.client",
    "robotrace.episode",
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "api-surface.json"


def _annotation_to_str(annotation: Any) -> str:
    """Render an annotation deterministically.

    ``inspect.Parameter.empty`` becomes ``""`` so the JSON snapshot
    stays small and the diff test can treat absence and ``empty`` the
    same way.
    """
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    # ``typing.get_type_hints`` would resolve forward refs but it also
    # raises on missing transitive imports, which we don't want during
    # a snapshot. ``repr()`` of typing constructs is stable across
    # 3.10 / 3.11 / 3.12 / 3.13 - the SDK's supported window.
    try:
        return inspect.formatannotation(annotation)
    except Exception:
        return repr(annotation)


def _snapshot_signature(fn: Any) -> dict[str, Any] | None:
    """Return ``{"parameters": [...], "return": "..."}`` for a callable.

    Returns ``None`` for objects we can't introspect (builtins without
    a Python signature, C extensions, etc.). The diff test treats a
    ``None`` snapshot as "untracked" - missing it doesn't fail the
    guard, but appearing later still passes.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    params: list[dict[str, Any]] = []
    for name, param in sig.parameters.items():
        params.append(
            {
                "name": name,
                "kind": param.kind.name,
                "has_default": param.default is not inspect.Parameter.empty,
                "annotation": _annotation_to_str(param.annotation),
            }
        )
    return {
        "parameters": params,
        "return": _annotation_to_str(sig.return_annotation),
    }


def _classify_attr(obj: Any, mod_name: str) -> str:
    """Classify a public attribute into one of the snapshot kinds."""
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.isroutine(obj):
        return "function"
    return "constant"


def _public_names(mod: Any) -> list[str]:
    """Return the public surface of a module.

    Prefer ``__all__`` (explicit contract) when present. Otherwise
    fall back to every non-underscore attribute - which is what the
    "implicit Python convention" treats as public.
    """
    declared = getattr(mod, "__all__", None)
    if isinstance(declared, (list, tuple)):
        return sorted({str(n) for n in declared})
    return sorted(n for n in vars(mod) if not n.startswith("_"))


def _snapshot_class(cls: type) -> dict[str, Any]:
    methods: dict[str, dict[str, Any] | None] = {}
    attributes: list[str] = []
    for name in sorted(vars(cls)):
        if name.startswith("_") and name not in {"__init__", "__call__"}:
            continue
        member = vars(cls)[name]
        if inspect.isfunction(member) or inspect.ismethoddescriptor(member):
            methods[name] = _snapshot_signature(member)
        elif inspect.isroutine(member):
            methods[name] = _snapshot_signature(member)
        else:
            # Plain class-level attribute (dataclass field, ClassVar,
            # constant). Record its presence so renames are caught.
            attributes.append(name)
    return {
        "bases": [b.__qualname__ for b in cls.__bases__ if b is not object],
        "methods": methods,
        "attributes": attributes,
    }


def snapshot_module(mod_name: str) -> dict[str, Any]:
    mod = importlib.import_module(mod_name)
    out: dict[str, Any] = {"symbols": {}}
    for name in _public_names(mod):
        if not hasattr(mod, name):
            continue
        obj = getattr(mod, name)
        kind = _classify_attr(obj, mod_name)
        record: dict[str, Any] = {"kind": kind}
        if kind == "function":
            record["signature"] = _snapshot_signature(obj)
        elif kind == "class":
            record.update(_snapshot_class(obj))
        elif kind == "module":
            # Don't recurse - the sub-module is snapshotted in its own
            # MODULES entry if it's meant to be tracked.
            record["target"] = obj.__name__
        else:
            record["type"] = type(obj).__name__
        out["symbols"][name] = record
    return out


def snapshot_all(modules: Iterable[str] = MODULES) -> dict[str, Any]:
    return {
        "$comment": (
            "Auto-generated SDK public-surface snapshot. Regenerate via "
            "`python packages/sdk-python/scripts/snapshot_api_surface.py` "
            "after intentional additive changes; never hand-edit."
        ),
        "modules": {name: snapshot_module(name) for name in modules},
    }


def _write_snapshot(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    OUTPUT_PATH.write_text(serialized + "\n", encoding="utf-8")


def main() -> None:
    payload = snapshot_all()
    _write_snapshot(payload)
    n = sum(len(m["symbols"]) for m in payload["modules"].values())
    print(f"Snapshotted {n} symbols across {len(payload['modules'])} modules → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
