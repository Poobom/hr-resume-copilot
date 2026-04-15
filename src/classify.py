"""Embedding-based resume→JD classification with LLM tiebreak for ambiguous cases."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .llm import CostTracker, chat_json, embed_batch

MARGIN_THRESHOLD = 0.05
EMBED_TEXT_MAX_CHARS = 3000
TIEBREAK_RESUME_MAX_CHARS = 2500


def load_jds_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def build_jd_embeddings(jds: list[dict], client, cost: CostTracker) -> dict[str, list[float]]:
    texts = [f"{jd['title']}. {jd['description']}" for jd in jds]
    vecs = embed_batch(client, texts, cost, step="classify_jd_embed")
    return {jd["jd_id"]: v for jd, v in zip(jds, vecs)}


def _cosine_matrix(resume_mat: np.ndarray, jd_mat: np.ndarray) -> np.ndarray:
    rn = resume_mat / (np.linalg.norm(resume_mat, axis=1, keepdims=True) + 1e-12)
    jn = jd_mat / (np.linalg.norm(jd_mat, axis=1, keepdims=True) + 1e-12)
    return rn @ jn.T


def _llm_tiebreak(
    resume_text: str,
    jd_a: dict,
    jd_b: dict,
    client,
    cost: CostTracker,
) -> str:
    system = (
        "당신은 채용 매칭 전문가. 이력서와 두 JD(A, B)를 읽고 "
        "어느 JD에 더 적합한지 선택하고 JSON으로 답."
    )
    user = (
        f"[이력서]\n{resume_text[:TIEBREAK_RESUME_MAX_CHARS]}\n\n"
        f"[JD A] id={jd_a['jd_id']} title={jd_a['title']}\n{jd_a['description']}\n\n"
        f"[JD B] id={jd_b['jd_id']} title={jd_b['title']}\n{jd_b['description']}\n\n"
        "형식: {\"choice\": \"A\" 또는 \"B\", \"reason\": \"짧은 근거\"}"
    )
    result = chat_json(
        client,
        model="gpt-4o-mini",
        system=system,
        user=user,
        cost=cost,
        step="classify_tiebreak",
        max_tokens=200,
    )
    choice = str(result.get("choice", "A")).strip().upper()
    return jd_b["jd_id"] if choice == "B" else jd_a["jd_id"]


def classify_resumes(
    extracted: list[dict],
    jds: list[dict],
    client,
    cost: CostTracker,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> list[dict]:
    valid = [e for e in extracted if e.get("ok")]
    if not valid:
        return []

    resume_texts = [e["masked_text"][:EMBED_TEXT_MAX_CHARS] for e in valid]
    resume_vecs = embed_batch(client, resume_texts, cost, step="classify_resume_embed")

    jd_emb = build_jd_embeddings(jds, client, cost)
    jd_ids = [jd["jd_id"] for jd in jds]
    jd_by_id = {jd["jd_id"]: jd for jd in jds}

    resume_mat = np.array(resume_vecs, dtype=np.float32)
    jd_mat = np.array([jd_emb[jid] for jid in jd_ids], dtype=np.float32)
    sims = _cosine_matrix(resume_mat, jd_mat)

    results: list[dict] = []
    for i, e in enumerate(valid):
        row = sims[i]
        order = np.argsort(-row)
        top1_idx, top2_idx = int(order[0]), int(order[1])
        top1_id, top2_id = jd_ids[top1_idx], jd_ids[top2_idx]
        sim1, sim2 = float(row[top1_idx]), float(row[top2_idx])
        margin = sim1 - sim2

        decision_path = "embedding"
        assigned = top1_id
        if margin < margin_threshold:
            assigned = _llm_tiebreak(
                e["masked_text"],
                jd_by_id[top1_id],
                jd_by_id[top2_id],
                client,
                cost,
            )
            decision_path = "llm_tiebreak"

        results.append(
            {
                "resume_id": e["resume_id"],
                "assigned_jd": assigned,
                "confidence": sim1,
                "top2": [(top1_id, sim1), (top2_id, sim2)],
                "margin": margin,
                "decision_path": decision_path,
            }
        )
    return results


if __name__ == "__main__":
    from .llm import get_client

    jds = load_jds_csv(Path(__file__).parent.parent.parent / "task02_data" / "jds.csv")
    print(f"Loaded {len(jds)} JDs")

    extracted = [
        {
            "resume_id": "R001",
            "masked_text": (
                "경력 7년 백엔드 엔지니어. Java/Spring Boot 기반 MSA 환경에서 "
                "대규모 트래픽 처리 경험. Kafka, Redis, Kubernetes 운영. "
                "주요 프로젝트: 이커머스 결제 시스템, 실시간 추천 파이프라인."
            ),
            "ok": True,
        },
        {
            "resume_id": "R002",
            "masked_text": (
                "프론트엔드 개발자 5년차. React, TypeScript, Next.js 기반 SSR 서비스. "
                "Core Web Vitals 최적화 경험. 디자인 시스템 구축 및 A/B 테스트 운영."
            ),
            "ok": True,
        },
        {
            "resume_id": "R003",
            "masked_text": (
                "데이터 분석 및 백엔드 혼합 경력 4년. Python, SQL, Airflow 기반 ETL. "
                "Java/Spring으로 사내 데이터 API 구축. ML 서빙 엔드포인트 경험."
            ),
            "ok": True,
        },
    ]

    client = get_client()
    cost = CostTracker()
    results = classify_resumes(extracted, jds, client, cost, margin_threshold=0.05)

    for r in results:
        print(
            f"{r['resume_id']} -> {r['assigned_jd']} "
            f"(conf={r['confidence']:.3f}, margin={r['margin']:.4f}, path={r['decision_path']})"
        )
        print(f"  top2: {r['top2']}")
    print(f"\nTotal cost: ${cost.total:.6f}")
    print(f"By step: {cost.by_step()}")
