"""SDK 0.2.0 readiness — surface freeze guard (mechanical part).

Pairs with the admin freeze clock at
``apps/web/components/admin/SdkSurfaceFreezeCard.tsx``. The clock is
the *promise* ("nothing breaks for 14 days"); this test is the
*proof* that we haven't already broken it on disk.

Compares ``packages/sdk-python/api-surface.json`` (the committed
baseline) against the live SDK surface re-snapshotted at test time.
Allowed changes (silent pass):

  * New modules, new public symbols, new methods, new attributes.
  * New parameters that have a default value.
  * Loosening a required parameter into an optional one (the
    asymmetric loosen rule).

Disallowed changes (loud fail with a per-symbol report):

  * Removing or renaming any baseline module / symbol / parameter.
  * Making a previously-optional parameter required.
  * Reordering positional parameters.
  * Changing a parameter's ``kind`` (e.g. POSITIONAL_OR_KEYWORD →
    KEYWORD_ONLY).
  * Removing class bases or methods.

If you genuinely need one of the disallowed changes, that's a
breaking change and must be paired with:

  1. A new major version bump in ``packages/sdk-python/_version.py``.
  2. A ``### Removed`` / ``### Breaking`` heading in the CHANGELOG.
  3. Reset the freeze clock from ``/admin/clients`` so the 14-day
     window starts over.
  4. Regenerate ``api-surface.json`` via
     ``python packages/sdk-python/scripts/snapshot_api_surface.py``.

Steps 1–3 are policy; step 4 is what makes this test pass again.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the snapshot helper by absolute path so the test never depends
# on whether `packages/sdk-python/scripts/` happens to be on sys.path
# (it usually isn't - pytest only auto-adds the test-roots).
_SDK_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _SDK_ROOT / "scripts" / "snapshot_api_surface.py"
_spec = importlib.util.spec_from_file_location(
    "_snapshot_api_surface", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
_snapshot_mod = importlib.util.module_from_spec(_spec)
sys.modules["_snapshot_api_surface"] = _snapshot_mod
_spec.loader.exec_module(_snapshot_mod)
snapshot_all = _snapshot_mod.snapshot_all

BASELINE_PATH = _SDK_ROOT / "api-surface.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    """Load the committed snapshot. Skip cleanly when it's missing.

    The freeze guard doesn't exist until someone runs the snapshot
    script for the first time. Skip rather than fail so a clean
    checkout that's mid-bootstrap doesn't go red on this one test.
    """
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"No baseline at {BASELINE_PATH}. Run "
            "`python packages/sdk-python/scripts/snapshot_api_surface.py` "
            "to seed the freeze guard."
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live() -> dict:
    return snapshot_all()


def _diff_parameters(
    base_params: list[dict], live_params: list[dict]
) -> list[str]:
    errors: list[str] = []
    live_by_name = {p["name"]: p for p in live_params}

    # 1. Every baseline parameter must still exist with the same kind.
    for i, base in enumerate(base_params):
        name = base["name"]
        live = live_by_name.get(name)
        if live is None:
            errors.append(f"  - parameter '{name}' was removed")
            continue
        if live["kind"] != base["kind"]:
            errors.append(
                f"  - parameter '{name}' changed kind: "
                f"{base['kind']} → {live['kind']}"
            )
        # 2. Required → optional is fine; optional → required is not.
        if base["has_default"] and not live["has_default"]:
            errors.append(
                f"  - parameter '{name}' became required "
                "(was optional in baseline)"
            )

    # 3. Positional order must be stable for POSITIONAL_OR_KEYWORD /
    # POSITIONAL_ONLY params. Keyword-only and varargs are unordered
    # by definition; skip those.
    def _positional_order(params: list[dict]) -> list[str]:
        return [
            p["name"]
            for p in params
            if p["kind"] in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}
        ]

    base_order = _positional_order(base_params)
    live_order = _positional_order(live_params)
    # Compare only the prefix length we share - additions at the end
    # are allowed if they have defaults (rule 1 in module docstring).
    shared = min(len(base_order), len(live_order))
    if base_order[:shared] != live_order[:shared]:
        errors.append(
            f"  - positional parameter order changed: "
            f"baseline {base_order} → live {live_order}"
        )

    return errors


def _diff_signature(
    fqname: str, base_sig: dict | None, live_sig: dict | None
) -> list[str]:
    if base_sig is None:
        # Baseline never knew the signature - we can't say anything.
        return []
    if live_sig is None:
        return [f"{fqname}:\n  - signature became unintrospectable"]

    errors: list[str] = []
    param_errors = _diff_parameters(
        base_sig["parameters"], live_sig["parameters"]
    )
    if param_errors:
        errors.append(fqname + ":\n" + "\n".join(param_errors))
    return errors


def _diff_class(
    fqname: str, base_cls: dict, live_cls: dict
) -> list[str]:
    errors: list[str] = []
    base_bases = set(base_cls.get("bases", []))
    live_bases = set(live_cls.get("bases", []))
    removed_bases = base_bases - live_bases
    if removed_bases:
        errors.append(
            f"{fqname}:\n  - bases removed: {sorted(removed_bases)}"
        )

    base_attrs = set(base_cls.get("attributes", []))
    live_attrs = set(live_cls.get("attributes", []))
    removed_attrs = base_attrs - live_attrs
    if removed_attrs:
        errors.append(
            f"{fqname}:\n  - attributes removed: {sorted(removed_attrs)}"
        )

    base_methods = base_cls.get("methods", {})
    live_methods = live_cls.get("methods", {})
    for method_name, base_sig in base_methods.items():
        live_sig = live_methods.get(method_name, "__MISSING__")
        if live_sig == "__MISSING__":
            errors.append(
                f"{fqname}:\n  - method '{method_name}' was removed"
            )
            continue
        method_errors = _diff_signature(
            f"{fqname}.{method_name}", base_sig, live_sig
        )
        errors.extend(method_errors)
    return errors


def _diff_module(
    mod_name: str, base_mod: dict, live_mod: dict
) -> list[str]:
    errors: list[str] = []
    base_syms = base_mod.get("symbols", {})
    live_syms = live_mod.get("symbols", {})

    for sym_name, base_record in base_syms.items():
        fqname = f"{mod_name}.{sym_name}"
        live_record = live_syms.get(sym_name)
        if live_record is None:
            errors.append(f"{fqname}:\n  - public symbol was removed")
            continue

        if base_record["kind"] != live_record["kind"]:
            errors.append(
                f"{fqname}:\n  - kind changed: "
                f"{base_record['kind']} → {live_record['kind']}"
            )
            continue

        if base_record["kind"] == "function":
            errors.extend(
                _diff_signature(
                    fqname,
                    base_record.get("signature"),
                    live_record.get("signature"),
                )
            )
        elif base_record["kind"] == "class":
            errors.extend(_diff_class(fqname, base_record, live_record))
        elif base_record["kind"] == "module":
            base_target = base_record.get("target")
            live_target = live_record.get("target")
            if base_target != live_target:
                errors.append(
                    f"{fqname}:\n  - re-export target changed: "
                    f"{base_target} → {live_target}"
                )
        else:
            # constant — only check type-name didn't drift
            base_type = base_record.get("type")
            live_type = live_record.get("type")
            if base_type and live_type and base_type != live_type:
                errors.append(
                    f"{fqname}:\n  - constant type changed: "
                    f"{base_type} → {live_type}"
                )
    return errors


def test_api_surface_freeze(baseline: dict, live: dict) -> None:
    base_mods = baseline["modules"]
    live_mods = live["modules"]

    errors: list[str] = []
    for mod_name, base_mod in base_mods.items():
        live_mod = live_mods.get(mod_name)
        if live_mod is None:
            errors.append(f"{mod_name}:\n  - public module was removed")
            continue
        errors.extend(_diff_module(mod_name, base_mod, live_mod))

    if errors:
        body = "\n\n".join(errors)
        pytest.fail(
            "SDK public surface changed in a breaking way "
            f"({len(errors)} issue{'s' if len(errors) != 1 else ''}) — "
            "this trips the 0.2.0 freeze guard.\n\n"
            f"{body}\n\n"
            "If this is intentional, bump the SDK major version, add "
            "a ### Removed / ### Breaking entry to the CHANGELOG, "
            "reset the freeze clock from /admin/clients, then "
            "regenerate the baseline:\n"
            "  python packages/sdk-python/scripts/snapshot_api_surface.py"
        )
