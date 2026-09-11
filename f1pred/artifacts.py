"""Run manifests with data hashes, settings and actual execution provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from .config import DEFAULT_MODEL_CONFIG, RAW_DIR, ROOT


def write_manifest(path: Path, **details: object) -> None:
    """Record inputs and settings without asserting that an unrun check passed."""
    sources = list(Path(__file__).parent.rglob("*.py"))
    sources += list((ROOT / "scripts").glob("*.py"))
    sources += list((ROOT / "tests").glob("*.py"))
    inputs = list(RAW_DIR.glob("*.parquet"))
    inputs += list((ROOT / "data" / "reference").glob("*.csv"))
    inputs += list((ROOT / "configs").glob("*.csv"))
    inputs += [p for p in (ROOT / "pyproject.toml", ROOT / "requirements-lock.txt") if p.exists()]
    hashes = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(sources + inputs)
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_config": asdict(DEFAULT_MODEL_CONFIG),
        "dependencies": {
            p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")
        },
        "sha256": hashes,
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
