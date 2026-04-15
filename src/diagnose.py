"""JD bottleneck diagnosis via layered funnel flags + LLM explanation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .llm import CostTracker, chat_json

CLUSTER_MAP = {
    "JD001": "engineering", "JD002": "engineering", "JD003": "engineering",
    "JD004": "engineering", "JD005": "engineering", "JD006": "engineering",
    "JD007": "engineering", "JD008": "engineering", "JD012": "engineering",
    "JD009": "non-engineering", "JD010": "non-engineering", "JD011": "non-engineering",
}

# Layer A thresholds
LOW_APPLICANTS_ABS = 20
LOW_DOC_PASS_ABS = 0.15

# Layer B multipliers
LOW_APPLICANTS_REL = 0.5
LOW_DOC_PASS_REL = 0.7

# Layer C thresholds
UNDER_ATTRACTED_RATIO = 0.5
OVER_ATTRACTED_RATIO = 3.0

DESCRIPTION_REF_MAX_CHARS = 500


def aggregate_funnel(funnel_stats: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jd_id, g in funnel_stats.groupby("jd_id"):
        g_sorted = g.sort_values("month")
        total_applicants = int(g_sorted["applicant_count"].sum())
        total_doc_pass = int(g_sorted["doc_pass_count"].sum())
        total_hired = int(g_sorted["final_hired_count"].sum())
        avg_doc_pass_rate = float(g_sorted["doc_pass_rate"].mean())
        avg_final_hired_rate = float(g_sorted["final_hired_rate"].mean())
        monthly = [
            (str(r["month"]), int(r["applicant_count"]), float(r["doc_pass_rate"]), int(r["final_hired_count"]))
            for _, r in g_sorted.iterrows()
        ]
        out[str(jd_id)] = {
            "total_applicants": total_applicants,
            "total_doc_pass": total_doc_pass,
            "avg_doc_pass_rate": avg_doc_pass_rate,
            "total_hired": total_hired,
            "avg_final_hired_rate": avg_final_hired_rate,
            "monthly": monthly,
        }
    return out


def compute_layer_a(jd_id: str, agg: dict) -> tuple[list[str], dict]:
    flags: list[str] = []
    applicants = agg["total_applicants"]
    doc_rate = agg["avg_doc_pass_rate"]
    hired = agg["total_hired"]

    if applicants < LOW_APPLICANTS_ABS:
        flags.append("LOW_APPLICANTS_ABS")
    if doc_rate < LOW_DOC_PASS_ABS:
        flags.append("LOW_DOC_PASS_ABS")
    if hired == 0:
        flags.append("ZERO_HIRED")

    evidence = {
        "applicants": applicants,
        "doc_pass_rate": round(doc_rate, 4),
        "hired": hired,
    }
    return flags, evidence


def compute_layer_b(
    jd_id: str, all_aggs: dict, cluster_map: dict
) -> tuple[list[str], dict]:
    cluster = cluster_map.get(jd_id, "unknown")
    peers = [
        a for jid, a in all_aggs.items()
        if cluster_map.get(jid) == cluster and jid != jd_id
    ]
    this_agg = all_aggs[jd_id]

    flags: list[str] = []
    if not peers:
        evidence = {
            "cluster": cluster,
            "peer_count": 0,
            "peer_avg_applicants": None,
            "this_jd_applicants": this_agg["total_applicants"],
            "peer_avg_doc_pass_rate": None,
            "this_jd_doc_pass_rate": round(this_agg["avg_doc_pass_rate"], 4),
        }
        return flags, evidence

    peer_avg_applicants = sum(p["total_applicants"] for p in peers) / len(peers)
    peer_avg_doc_rate = sum(p["avg_doc_pass_rate"] for p in peers) / len(peers)

    if this_agg["total_applicants"] < peer_avg_applicants * LOW_APPLICANTS_REL:
        flags.append("LOW_APPLICANTS_REL")
    if this_agg["avg_doc_pass_rate"] < peer_avg_doc_rate * LOW_DOC_PASS_REL:
        flags.append("LOW_DOC_PASS_REL")

    evidence = {
        "cluster": cluster,
        "peer_count": len(peers),
        "peer_avg_applicants": round(peer_avg_applicants, 2),
        "this_jd_applicants": this_agg["total_applicants"],
        "peer_avg_doc_pass_rate": round(peer_avg_doc_rate, 4),
        "this_jd_doc_pass_rate": round(this_agg["avg_doc_pass_rate"], 4),
    }
    return flags, evidence


def compute_layer_c(
    jd_id: str,
    classified: list[dict],
    actual_applicants: int,
    margin_threshold: float = 0.05,
) -> tuple[list[str], dict]:
    potential_fit = 0
    for r in classified:
        top2 = r.get("top2", [])
        if not top2:
            continue
        top1_id = top2[0][0]
        if top1_id == jd_id:
            potential_fit += 1
            continue
        if len(top2) > 1 and top2[1][0] == jd_id and r.get("margin", 1.0) < margin_threshold:
            potential_fit += 1

    if actual_applicants <= 0:
        fit_ratio = 0.0
    else:
        fit_ratio = potential_fit / actual_applicants

    flags: list[str] = []
    interpretation = "balanced"
    if actual_applicants > 0:
        if fit_ratio < UNDER_ATTRACTED_RATIO:
            flags.append("UNDER_ATTRACTED")
            interpretation = "적합 후보 대비 실제 지원 부족 — JD 매력도 문제 의심"
        elif fit_ratio > OVER_ATTRACTED_RATIO:
            flags.append("OVER_ATTRACTED")
            interpretation = "부적합 지원이 과다 유입 — JD 문구 모호성 의심"

    evidence = {
        "potential_fit": potential_fit,
        "actual_applicants": actual_applicants,
        "fit_ratio": round(fit_ratio, 3),
        "interpretation": interpretation,
    }
    return flags, evidence


def pick_healthy_refs(
    cluster: str,
    all_flags: dict[str, list],
    agg: dict,
    jds: list[dict],
    top_n: int = 2,
) -> list[dict]:
    jd_by_id = {jd["jd_id"]: jd for jd in jds}
    candidates: list[tuple[str, float]] = []
    for jd_id, flags in all_flags.items():
        if CLUSTER_MAP.get(jd_id) != cluster:
            continue
        if flags:
            continue
        hired_rate = agg.get(jd_id, {}).get("avg_final_hired_rate", 0.0)
        candidates.append((jd_id, hired_rate))

    candidates.sort(key=lambda x: -x[1])
    refs: list[dict] = []
    for jd_id, _ in candidates[:top_n]:
        jd = jd_by_id.get(jd_id)
        if not jd:
            continue
        desc = str(jd.get("description", ""))[:DESCRIPTION_REF_MAX_CHARS]
        refs.append({"jd_id": jd_id, "title": jd.get("title", ""), "description": desc})
    return refs


def diagnose_jd(
    jd: dict,
    layered_flags: dict,
    evidence: dict,
    reference_examples: list[dict],
    client,
    cost: CostTracker,
) -> dict:
    system = (
        "당신은 HR JD 컨설턴트입니다. 주어진 JD와 병목 지표(플래그·근거)를 읽고, "
        "왜 지원자/서류통과/합격이 저조한지 가설과 근거 체인을 제시하고, "
        "JD 문구 중 의심 구절과 개선안을 JSON으로 반환하세요. "
        "반드시 reference_examples의 문구 스타일을 참고하여 개선안을 제시하세요. "
        "지표와 연결되지 않은 추측은 금지."
    )
    user = (
        f"[JD]\n"
        f"jd_id: {jd.get('jd_id')}\n"
        f"title: {jd.get('title')}\n"
        f"description:\n{jd.get('description', '')}\n\n"
        f"[Layered Flags]\n{json.dumps(layered_flags, ensure_ascii=False)}\n\n"
        f"[Evidence]\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"[Reference Examples — 같은 클러스터의 건강한 JD 문구]\n"
        f"{json.dumps(reference_examples, ensure_ascii=False, indent=2)}\n\n"
        "형식(JSON):\n"
        "{\n"
        '  "root_cause_hypothesis": "한두 문장 요약",\n'
        '  "evidence_chain": ["지표 → 해석 → 근거 인용", ...],\n'
        '  "suspected_quotes": [{"quote": "JD 원문 일부", "why": "문제 이유"}, ...],\n'
        '  "suggested_edits": [{"before": "...", "after": "...", "rationale": "레퍼런스 JD 참조"}, ...]\n'
        "}"
    )
    return chat_json(
        client,
        model="gpt-4o-mini",
        system=system,
        user=user,
        cost=cost,
        step="diagnose_jd",
        max_tokens=1500,
    )


def diagnose_all(
    classified: list[dict],
    jds: list[dict],
    funnel_stats: pd.DataFrame,
    client,
    cost: CostTracker,
) -> list[dict]:
    aggs = aggregate_funnel(funnel_stats)

    # First pass: compute all flags
    per_jd: dict[str, dict] = {}
    for jd in jds:
        jd_id = jd["jd_id"]
        agg = aggs.get(jd_id, {
            "total_applicants": 0, "total_doc_pass": 0, "avg_doc_pass_rate": 0.0,
            "total_hired": 0, "avg_final_hired_rate": 0.0, "monthly": [],
        })
        flags_a, ev_a = compute_layer_a(jd_id, agg)
        flags_b, ev_b = compute_layer_b(jd_id, aggs, CLUSTER_MAP)
        flags_c, ev_c = compute_layer_c(jd_id, classified, agg["total_applicants"])
        per_jd[jd_id] = {
            "jd": jd,
            "agg": agg,
            "flags_a": flags_a, "ev_a": ev_a,
            "flags_b": flags_b, "ev_b": ev_b,
            "flags_c": flags_c, "ev_c": ev_c,
        }

    all_flags_combined: dict[str, list] = {
        jid: p["flags_a"] + p["flags_b"] + p["flags_c"] for jid, p in per_jd.items()
    }

    results: list[dict] = []
    for jd in jds:
        jd_id = jd["jd_id"]
        p = per_jd[jd_id]
        cluster = CLUSTER_MAP.get(jd_id, "unknown")
        layered_flags = {
            "absolute": p["flags_a"],
            "cluster_relative": p["flags_b"],
            "supply_demand": p["flags_c"],
        }
        evidence = {
            "absolute": p["ev_a"],
            "cluster_relative": p["ev_b"],
            "supply_demand": p["ev_c"],
        }
        any_flag = bool(p["flags_a"] or p["flags_b"] or p["flags_c"])

        base = {
            "jd_id": jd_id,
            "title": jd.get("title", ""),
            "cluster": cluster,
            "layered_flags": layered_flags,
            "evidence": evidence,
            "is_healthy": not any_flag,
            "reference_examples": [],
            "root_cause_hypothesis": None,
            "evidence_chain": None,
            "suspected_quotes": None,
            "suggested_edits": None,
        }

        if not any_flag:
            results.append(base)
            continue

        refs = pick_healthy_refs(cluster, all_flags_combined, aggs, jds, top_n=2)
        base["reference_examples"] = refs

        llm_out = diagnose_jd(jd, layered_flags, evidence, refs, client, cost)
        base["root_cause_hypothesis"] = llm_out.get("root_cause_hypothesis")
        base["evidence_chain"] = llm_out.get("evidence_chain")
        base["suspected_quotes"] = llm_out.get("suspected_quotes")
        base["suggested_edits"] = llm_out.get("suggested_edits")
        results.append(base)

    return results


if __name__ == "__main__":
    import os

    data_dir = Path(__file__).parent.parent.parent / "task02_data"
    funnel = pd.read_csv(data_dir / "funnel_stats.csv")
    jds = pd.read_csv(data_dir / "jds.csv").to_dict("records")

    classified = [
        {"resume_id": "R001", "assigned_jd": "JD001", "confidence": 0.82,
         "top2": [("JD001", 0.82), ("JD003", 0.70)], "margin": 0.12, "decision_path": "embedding"},
        {"resume_id": "R002", "assigned_jd": "JD002", "confidence": 0.78,
         "top2": [("JD002", 0.78), ("JD006", 0.76)], "margin": 0.02, "decision_path": "llm_tiebreak"},
        {"resume_id": "R003", "assigned_jd": "JD005", "confidence": 0.65,
         "top2": [("JD005", 0.65), ("JD001", 0.61)], "margin": 0.04, "decision_path": "llm_tiebreak"},
        {"resume_id": "R004", "assigned_jd": "JD001", "confidence": 0.71,
         "top2": [("JD001", 0.71), ("JD007", 0.55)], "margin": 0.16, "decision_path": "embedding"},
        {"resume_id": "R005", "assigned_jd": "JD011", "confidence": 0.60,
         "top2": [("JD011", 0.60), ("JD009", 0.50)], "margin": 0.10, "decision_path": "embedding"},
    ]

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    cost = CostTracker()

    if has_key:
        from .llm import get_client
        client = get_client()
        results = diagnose_all(classified, jds, funnel, client, cost)
    else:
        # Flags-only dry run without LLM
        aggs = aggregate_funnel(funnel)
        per_jd_flags = {}
        for jd in jds:
            jd_id = jd["jd_id"]
            agg = aggs.get(jd_id, {"total_applicants": 0, "avg_doc_pass_rate": 0.0, "total_hired": 0, "avg_final_hired_rate": 0.0, "total_doc_pass": 0, "monthly": []})
            fa, ea = compute_layer_a(jd_id, agg)
            fb, eb = compute_layer_b(jd_id, aggs, CLUSTER_MAP)
            fc, ec = compute_layer_c(jd_id, classified, agg["total_applicants"])
            per_jd_flags[jd_id] = {
                "jd_id": jd_id,
                "title": jd.get("title"),
                "cluster": CLUSTER_MAP.get(jd_id),
                "layered_flags": {"absolute": fa, "cluster_relative": fb, "supply_demand": fc},
                "evidence": {"absolute": ea, "cluster_relative": eb, "supply_demand": ec},
                "is_healthy": not (fa or fb or fc),
            }
        results = list(per_jd_flags.values())

    print(f"OPENAI_API_KEY present: {has_key}")
    for r in results[:3]:
        print(f"\n=== {r['jd_id']} {r.get('title', '')} (cluster={r.get('cluster')}) ===")
        print(f"  healthy: {r.get('is_healthy')}")
        print(f"  flags: {r.get('layered_flags')}")
        print(f"  evidence: {json.dumps(r.get('evidence'), ensure_ascii=False)}")
        if has_key and not r.get("is_healthy"):
            print(f"  root_cause: {r.get('root_cause_hypothesis')}")
            print(f"  refs: {[x['jd_id'] for x in r.get('reference_examples', [])]}")
    print(f"\nTotal cost: ${cost.total:.6f}")
