"""SHA1-based artifact cache utilities.

Adapted from task01_solution. Extended with per-file resume cache helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACTS_ROOT = Path("artifacts")
HASH_PREFIX_LEN = 12
RESUME_CACHE_DIR = ARTIFACTS_ROOT / "resume_cache"


def compute_sha1(file_path: Path) -> str:
    h = hashlib.sha1()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha1_str(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def get_run_dir(input_signature: str, root: Path = ARTIFACTS_ROOT) -> Path:
    digest = hashlib.sha1(input_signature.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]
    return root / digest


def stage_done(out_dir: Path, stage_filename: str) -> bool:
    return (out_dir / stage_filename).exists()


def resume_cache_path(resume_sha1: str, root: Path = RESUME_CACHE_DIR) -> Path:
    return root / f"{resume_sha1}.json"


def load_resume_cache(resume_sha1: str, root: Path = RESUME_CACHE_DIR) -> dict[str, Any] | None:
    path = resume_cache_path(resume_sha1, root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_resume_cache(resume_sha1: str, data: dict[str, Any], root: Path = RESUME_CACHE_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = resume_cache_path(resume_sha1, root)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
