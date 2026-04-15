"""Stage 4 — 수요-공급 갭 시각화 집계 (LLM 호출 없음)."""
from __future__ import annotations

from collections import Counter
from typing import Any


_STATUS_ORDER = {"UNDER_ATTRACTED": 0, "NORMAL": 1, "OVER_ATTRACTED": 2}


def _derive_status(layered_flags: dict[str, Any], fit_ratio: float) -> str:
    # Stage 3 플래그 우선, 없으면 fit_ratio로 fallback
    flag = (layered_flags or {}).get("supply_demand")
    if flag in ("UNDER_ATTRACTED", "OVER_ATTRACTED"):
        return flag
    if fit_ratio < 0.5:
        return "UNDER_ATTRACTED"
    if fit_ratio > 3.0:
        return "OVER_ATTRACTED"
    return "NORMAL"


def compute_gap_table(diagnosis: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for d in diagnosis:
        ev = (d.get("evidence") or {}).get("supply_demand") or {}
        fit_ratio = float(ev.get("fit_ratio", 0.0))
        status = _derive_status(d.get("layered_flags") or {}, fit_ratio)
        rows.append({
            "jd_id": d["jd_id"],
            "title": d.get("title", ""),
            "potential_fit": int(ev.get("potential_fit", 0)),
            "actual_applicants": int(ev.get("actual_applicants", 0)),
            "fit_ratio": fit_ratio,
            "attraction_status": status,
        })

    def sort_key(r: dict) -> tuple[int, float]:
        bucket = _STATUS_ORDER[r["attraction_status"]]
        if r["attraction_status"] == "UNDER_ATTRACTED":
            return (bucket, r["fit_ratio"])
        if r["attraction_status"] == "OVER_ATTRACTED":
            return (bucket, -r["fit_ratio"])
        return (bucket, r["fit_ratio"])

    rows.sort(key=sort_key)
    return rows


def compute_drift_distribution(
    jd_id: str,
    classified: list[dict],
    margin_threshold: float = 0.05,
) -> list[dict]:
    fitting_resumes: list[dict] = []
    for r in classified:
        top2 = r.get("top2") or []
        top1_id = top2[0][0] if len(top2) >= 1 else r.get("assigned_jd")
        top2_id = top2[1][0] if len(top2) >= 2 else None
        margin = float(r.get("margin", 1.0))

        if top1_id == jd_id:
            fitting_resumes.append(r)
        elif top2_id == jd_id and margin < margin_threshold:
            fitting_resumes.append(r)

    total = len(fitting_resumes)
    if total == 0:
        return []

    counts = Counter(r.get("assigned_jd") for r in fitting_resumes)
    entries = [
        {
            "assigned_jd": assigned,
            "count": count,
            "percentage": round(count / total * 100.0, 2),
        }
        for assigned, count in counts.items()
    ]
    entries.sort(key=lambda e: e["count"], reverse=True)
    return entries


def build_comparison_data(
    classified: list[dict],
    diagnosis: list[dict],
) -> dict:
    gap_table = compute_gap_table(diagnosis)
    drift: dict[str, list[dict]] = {}
    for row in gap_table:
        if row["attraction_status"] == "UNDER_ATTRACTED":
            drift[row["jd_id"]] = compute_drift_distribution(row["jd_id"], classified)
    return {"gap_table": gap_table, "drift_distributions": drift}


def summarize_attraction_status(gap_table: list[dict]) -> dict:
    counts = Counter(r["attraction_status"] for r in gap_table)
    under = [r for r in gap_table if r["attraction_status"] == "UNDER_ATTRACTED"]
    over = [r for r in gap_table if r["attraction_status"] == "OVER_ATTRACTED"]
    under_sorted = sorted(under, key=lambda r: r["fit_ratio"])[:3]
    over_sorted = sorted(over, key=lambda r: r["fit_ratio"], reverse=True)[:3]
    return {
        "counts": {
            "UNDER_ATTRACTED": counts.get("UNDER_ATTRACTED", 0),
            "NORMAL": counts.get("NORMAL", 0),
            "OVER_ATTRACTED": counts.get("OVER_ATTRACTED", 0),
        },
        "top_under_attracted": under_sorted,
        "top_over_attracted": over_sorted,
    }


if __name__ == "__main__":
    import json

    classified_sample = [
        {
            "resume_id": "R001",
            "assigned_jd": "JD002",
            "confidence": 0.82,
            "top2": [("JD001", 0.81), ("JD002", 0.80)],
            "margin": 0.01,
            "decision_path": "embedding+llm",
        },
        {
            "resume_id": "R002",
            "assigned_jd": "JD002",
            "confidence": 0.76,
            "top2": [("JD001", 0.78), ("JD003", 0.60)],
            "margin": 0.18,
            "decision_path": "embedding",
        },
        {
            "resume_id": "R003",
            "assigned_jd": "JD001",
            "confidence": 0.88,
            "top2": [("JD001", 0.88), ("JD002", 0.70)],
            "margin": 0.18,
            "decision_path": "embedding",
        },
        {
            "resume_id": "R004",
            "assigned_jd": "JD003",
            "confidence": 0.71,
            "top2": [("JD003", 0.71), ("JD001", 0.69)],
            "margin": 0.02,
            "decision_path": "embedding+llm",
        },
        {
            "resume_id": "R005",
            "assigned_jd": "JD002",
            "confidence": 0.65,
            "top2": [("JD002", 0.65), ("JD001", 0.63)],
            "margin": 0.02,
            "decision_path": "embedding+llm",
        },
    ]

    diagnosis_sample = [
        {
            "jd_id": "JD001",
            "title": "Backend Engineer",
            "layered_flags": {"supply_demand": "UNDER_ATTRACTED"},
            "evidence": {
                "supply_demand": {
                    "potential_fit": 3,
                    "actual_applicants": 10,
                    "fit_ratio": 0.3,
                }
            },
        },
        {
            "jd_id": "JD002",
            "title": "Frontend Engineer",
            "layered_flags": {"supply_demand": "OVER_ATTRACTED"},
            "evidence": {
                "supply_demand": {
                    "potential_fit": 3,
                    "actual_applicants": 1,
                    "fit_ratio": 3.5,
                }
            },
        },
        {
            "jd_id": "JD003",
            "title": "Data Scientist",
            "layered_flags": {},
            "evidence": {
                "supply_demand": {
                    "potential_fit": 5,
                    "actual_applicants": 5,
                    "fit_ratio": 1.0,
                }
            },
        },
    ]

    result = build_comparison_data(classified_sample, diagnosis_sample)
    result["summary"] = summarize_attraction_status(result["gap_table"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
