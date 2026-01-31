"""Write PQC run settings to a TOML file."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _clean_value(val: Any) -> Any:
    if isinstance(val, tuple):
        return list(val)
    return val


def _clean_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in mapping.items():
        if val is None:
            continue
        out[key] = _clean_value(val)
    return out


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return repr(float(val))
    if isinstance(val, str):
        return f'"{_toml_escape(val)}"'
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_toml_value(v) for v in val)
        return f"[{inner}]"
    return f'"{_toml_escape(str(val))}"'


def _write_section(fp, name: str, mapping: dict[str, Any]) -> None:
    if not mapping:
        return
    fp.write(f"[{name}]\n")
    for key, val in mapping.items():
        fp.write(f"{key} = {_toml_value(val)}\n")
    fp.write("\n")


def _as_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported settings type: {type(obj)}")


def write_run_settings_toml(
    path: str | Path,
    *,
    parfile: str | Path,
    backend_col: str,
    drop_unmatched: bool,
    bad_cfg: Any,
    tr_cfg: Any,
    dip_cfg: Any,
    merge_cfg: Any,
    feature_cfg: Any,
    struct_cfg: Any,
    step_cfg: Any,
    dm_cfg: Any,
    robust_cfg: Any,
    preproc_cfg: Any,
    gate_cfg: Any,
    solar_cfg: Any,
    orbital_cfg: Any,
) -> None:
    """Write pipeline settings to a TOML file (overwrites existing file)."""
    out_path = Path(path)
    settings = {
        "run": {
            "parfile": str(parfile),
            "backend_col": str(backend_col),
            "drop_unmatched": bool(drop_unmatched),
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "bad_cfg": _as_dict(bad_cfg),
        "tr_cfg": _as_dict(tr_cfg),
        "dip_cfg": _as_dict(dip_cfg),
        "merge_cfg": _as_dict(merge_cfg),
        "feature_cfg": _as_dict(feature_cfg),
        "struct_cfg": _as_dict(struct_cfg),
        "step_cfg": _as_dict(step_cfg),
        "dm_cfg": _as_dict(dm_cfg),
        "robust_cfg": _as_dict(robust_cfg),
        "preproc_cfg": _as_dict(preproc_cfg),
        "gate_cfg": _as_dict(gate_cfg),
        "solar_cfg": _as_dict(solar_cfg),
        "orbital_cfg": _as_dict(orbital_cfg),
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        for section, mapping in settings.items():
            _write_section(fp, section, _clean_mapping(mapping))
