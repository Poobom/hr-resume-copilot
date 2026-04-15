"""HR Resume Copilot — pipeline orchestration.

Stages:
  1. Extract (hybrid: pdfplumber / python-docx / LibreOffice / Vision)
  2. Classify (embedding + LLM tiebreak for borderline cases)
  3. Diagnose (triple-layer flags + LLM explanation for flagged JDs)
  4. Compare (supply-demand gap visualization data)
  5. Export (Notion-friendly markdown)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src import classify, compare, diagnose, export, extract
from src.cache import ARTIFACTS_ROOT, get_run_dir, stage_done
from src.llm import CostTracker, get_client

DEFAULT_INPUT_DIR = Path("../task02_data")
STAGE_FILES = {
    "extract": "01_extracted.json",
    "classify": "02_classified.json",
    "diagnose": "03_diagnosis.json",
    "compare": "04_comparison.json",
    "export": "05_notion_report.md",
}


def _signature(jds_csv: Path, funnel_csv: Path, resume_dir: Path) -> str:
    files = sorted(resume_dir.glob("*"))
    return "|".join(
        [
            jds_csv.read_text(encoding="utf-8"),
            funnel_csv.read_text(encoding="utf-8"),
            str(len(files)),
            files[0].name if files else "",
            files[-1].name if files else "",
        ]
    )


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline(
    input_dir: Path = DEFAULT_INPUT_DIR,
    force: bool = False,
    on_step: Callable[[int, str, dict], None] | None = None,
    skip_export: bool = False,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    jds_csv = input_dir / "jds.csv"
    funnel_csv = input_dir / "funnel_stats.csv"
    resume_dir = input_dir / "resumes"

    assert jds_csv.exists(), f"Missing {jds_csv}"
    assert funnel_csv.exists(), f"Missing {funnel_csv}"
    assert resume_dir.exists(), f"Missing {resume_dir}"

    run_dir = get_run_dir(_signature(jds_csv, funnel_csv, resume_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] run_dir = {run_dir}")

    cost = CostTracker()
    client = get_client()

    jds = classify.load_jds_csv(jds_csv)
    funnel_stats = pd.read_csv(funnel_csv)

    # Stage 1: Extract
    extract_path = run_dir / STAGE_FILES["extract"]
    if not force and stage_done(run_dir, STAGE_FILES["extract"]):
        print(f"[pipeline] stage 1 (extract) cached")
        extracted = _load_json(extract_path)
    else:
        print(f"[pipeline] stage 1 (extract) running on {resume_dir}")
        def _progress(cur, total, rid):
            if cur % 50 == 0 or cur == total:
                print(f"  extract progress: {cur}/{total} ({rid})")
        extracted = extract.extract_directory(resume_dir, client, cost, on_progress=_progress)
        _save_json(extract_path, extracted)
    success = sum(1 for e in extracted if e.get("ok"))
    print(f"[pipeline] extracted {success}/{len(extracted)} ok ({success/len(extracted)*100:.1f}%)")
    if on_step:
        on_step(1, "extract", {"total": len(extracted), "ok": success})

    # Stage 2: Classify
    classify_path = run_dir / STAGE_FILES["classify"]
    if not force and stage_done(run_dir, STAGE_FILES["classify"]):
        print(f"[pipeline] stage 2 (classify) cached")
        classified = _load_json(classify_path)
    else:
        print(f"[pipeline] stage 2 (classify) running")
        classified = classify.classify_resumes(extracted, jds, client, cost)
        _save_json(classify_path, classified)
    tiebreaks = sum(1 for c in classified if c.get("decision_path") == "llm_tiebreak")
    print(f"[pipeline] classified {len(classified)} ({tiebreaks} llm tiebreaks)")
    if on_step:
        on_step(2, "classify", {"total": len(classified), "tiebreaks": tiebreaks})

    # Stage 3: Diagnose
    diagnose_path = run_dir / STAGE_FILES["diagnose"]
    if not force and stage_done(run_dir, STAGE_FILES["diagnose"]):
        print(f"[pipeline] stage 3 (diagnose) cached")
        diagnosis = _load_json(diagnose_path)
    else:
        print(f"[pipeline] stage 3 (diagnose) running")
        diagnosis = diagnose.diagnose_all(classified, jds, funnel_stats, client, cost)
        _save_json(diagnose_path, diagnosis)
    flagged = sum(1 for d in diagnosis if not d.get("is_healthy"))
    print(f"[pipeline] diagnosed {len(diagnosis)} JDs ({flagged} flagged)")
    if on_step:
        on_step(3, "diagnose", {"total": len(diagnosis), "flagged": flagged})

    # Stage 4: Compare
    compare_path = run_dir / STAGE_FILES["compare"]
    if not force and stage_done(run_dir, STAGE_FILES["compare"]):
        print(f"[pipeline] stage 4 (compare) cached")
        comparison = _load_json(compare_path)
    else:
        print(f"[pipeline] stage 4 (compare) running")
        comparison = compare.build_comparison_data(classified, diagnosis)
        _save_json(compare_path, comparison)
    print(f"[pipeline] comparison built ({len(comparison['gap_table'])} gap rows)")
    if on_step:
        on_step(4, "compare", {"gap_rows": len(comparison["gap_table"])})

    # Save cost before export
    cost.save(run_dir)

    # Stage 5: Export
    export_path = run_dir / STAGE_FILES["export"]
    if skip_export:
        print(f"[pipeline] stage 5 (export) skipped")
    elif not force and stage_done(run_dir, STAGE_FILES["export"]):
        print(f"[pipeline] stage 5 (export) cached")
    else:
        print(f"[pipeline] stage 5 (export) rendering")
        meta = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "total_resumes": len(extracted),
            "extraction_success_rate": success / max(1, len(extracted)),
        }
        md = export.build_notion_markdown(
            classified=classified,
            extracted=extracted,
            diagnosis=diagnosis,
            comparison=comparison,
            cost=cost.to_dict(),
            jds=jds,
            meta=meta,
        )
        export.save_markdown(md, export_path)
        print(f"[pipeline] exported {export_path} ({len(md)} chars)")
    if on_step:
        on_step(5, "export", {"path": str(export_path)})

    print(f"[pipeline] total cost: ${cost.total:.4f}")
    return {k: run_dir / v for k, v in STAGE_FILES.items()} | {"cost": run_dir / "cost.json", "run_dir": run_dir}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    paths = run_pipeline(Path(args.input_dir), force=args.force, skip_export=args.skip_export)
    print("[pipeline] artifacts:")
    for key, p in paths.items():
        print(f"  {key}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
