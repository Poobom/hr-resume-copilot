"""HR Resume Copilot — Streamlit UI.

6 탭 구조: 대시보드 / 분류 / JD 병목 / 수요-공급 / PII / 다운로드.
pipeline.run_pipeline() 결과 artifacts 를 6개 JSON+MD 로 로드하여 렌더링.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

st.set_page_config(page_title="HR Resume Copilot", layout="wide", page_icon="📋")

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
ARTIFACTS_ROOT = ROOT / "artifacts"
DATA_DIR = (ROOT / ".." / "task02_data").resolve()
JDS_CSV = DATA_DIR / "jds.csv"
FUNNEL_CSV = DATA_DIR / "funnel_stats.csv"
RESUMES_DIR = DATA_DIR / "resumes"
PII_POLICY_MD = ROOT / "pii_policy.md"
UPLOAD_TMP_DIR = ROOT / ".tmp_uploads"

BUDGET_USD = 3.00
BUDGET_BLOCK_USD = 2.50

STAGE_FILES = {
    "extracted": "01_extracted.json",
    "classified": "02_classified.json",
    "diagnosis": "03_diagnosis.json",
    "comparison": "04_comparison.json",
    "report_md": "05_notion_report.md",
    "cost": "cost.json",
}

FLAG_ICONS = {
    "LOW_APPLICANTS_ABS": "🔴",
    "LOW_DOC_PASS_ABS": "🔴",
    "ZERO_HIRED": "🔴",
    "LOW_APPLICANTS_REL": "🟡",
    "LOW_DOC_PASS_REL": "🟡",
    "UNDER_ATTRACTED": "🟠",
    "OVER_ATTRACTED": "🟠",
}

FLAG_HUMAN = {
    "LOW_APPLICANTS_ABS": "지원자 < 20명",
    "LOW_DOC_PASS_ABS": "서류통과율 < 15%",
    "ZERO_HIRED": "합격자 0명",
    "LOW_APPLICANTS_REL": "클러스터 평균 50% 미만",
    "LOW_DOC_PASS_REL": "클러스터 평균 대비 낮은 통과율",
    "UNDER_ATTRACTED": "잠재 대비 지원 저조",
    "OVER_ATTRACTED": "부적합 지원 과다",
}

# ---------------------------------------------------------------------------
# 선택적 import (graceful degradation)
# ---------------------------------------------------------------------------
PIPELINE_AVAILABLE = True
PIPELINE_ERROR: str | None = None
try:
    from pipeline import run_pipeline  # type: ignore
except Exception as exc:
    run_pipeline = None  # type: ignore
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(exc)

PII_AVAILABLE = True
PII_ERROR: str | None = None
try:
    from src.pii import safe_display_row  # type: ignore
except Exception as exc:
    safe_display_row = None  # type: ignore
    PII_AVAILABLE = False
    PII_ERROR = str(exc)

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None  # type: ignore
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# I/O 유틸
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def discover_latest_run() -> Path | None:
    if not ARTIFACTS_ROOT.exists():
        return None
    candidates = [
        d for d in ARTIFACTS_ROOT.iterdir()
        if d.is_dir() and d.name != "resume_cache"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        # 최소 한 개 아티팩트라도 있어야 유효
        if any((c / f).exists() for f in STAGE_FILES.values()):
            return c
    return None


def load_artifacts(run_dir: Path) -> dict[str, Any]:
    b: dict[str, Any] = {"run_dir": run_dir, "run_id": run_dir.name}
    b["extracted"] = _load_json(run_dir / STAGE_FILES["extracted"]) or []
    b["classified"] = _load_json(run_dir / STAGE_FILES["classified"]) or []
    b["diagnosis"] = _load_json(run_dir / STAGE_FILES["diagnosis"]) or []
    b["comparison"] = _load_json(run_dir / STAGE_FILES["comparison"]) or {"gap_table": [], "drift_distributions": {}}
    b["report_md"] = _load_text(run_dir / STAGE_FILES["report_md"]) or ""
    b["cost"] = _load_json(run_dir / STAGE_FILES["cost"]) or {"total_cost_usd": 0.0, "records": [], "by_step": {}}
    return b


def load_jds() -> list[dict]:
    if not JDS_CSV.exists():
        return []
    try:
        return pd.read_csv(JDS_CSV).to_dict("records")
    except Exception:
        return []


def total_cost(bundle: dict | None) -> float:
    if not bundle:
        return 0.0
    cost = bundle.get("cost") or {}
    return float(cost.get("total_cost_usd") or cost.get("total") or 0.0)


# ---------------------------------------------------------------------------
# PII 안전 DataFrame 렌더
# ---------------------------------------------------------------------------
FORBIDDEN_COLS = {"pii_fields_separated", "raw_text"}


def render_safe_df(df: pd.DataFrame, extra_forbidden: set[str] | None = None, **kwargs) -> None:
    forbidden = FORBIDDEN_COLS | (extra_forbidden or set())
    safe_cols = [c for c in df.columns if c not in forbidden]
    st.dataframe(df[safe_cols], use_container_width=True, hide_index=True, **kwargs)


_YEAR_LINE_RE = re.compile(r".*?\d+\s*년.*")


def first_career_hint(masked_text: str, max_chars: int = 40) -> str:
    if not masked_text:
        return "-"
    for line in masked_text.splitlines():
        if "년" in line and re.search(r"\d", line):
            s = line.strip()
            if s:
                return s[: max_chars - 3] + "..." if len(s) > max_chars else s
    return "-"


def skill_snippet(masked_text: str, max_chars: int = 100) -> str:
    if not masked_text:
        return "-"
    # 앞 쪽 여러 라인 중 '경력' '기술' 키워드 주변 또는 첫 유효 라인
    lines = [ln.strip() for ln in masked_text.splitlines() if ln.strip()]
    for ln in lines:
        if any(k in ln for k in ("기술", "스킬", "Skill", "skill", "Stack", "stack")):
            return ln[:max_chars] + ("..." if len(ln) > max_chars else "")
    # fallback: 가장 긴 라인 앞부분
    if not lines:
        return "-"
    longest = max(lines, key=len)
    return longest[:max_chars] + ("..." if len(longest) > max_chars else "")


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[bool, bool, list]:
    st.sidebar.title("📋 HR Resume Copilot")
    st.sidebar.caption("이력서 분류 & JD 진단 도구")

    st.sidebar.divider()
    st.sidebar.subheader("📂 데이터 소스")
    jds_ok = JDS_CSV.exists()
    funnel_ok = FUNNEL_CSV.exists()
    resume_count = 0
    if RESUMES_DIR.exists():
        resume_count = sum(1 for _ in RESUMES_DIR.iterdir() if _.is_file())
    st.sidebar.markdown(
        f"- {'✅' if jds_ok else '❌'} `jds.csv`\n"
        f"- {'✅' if funnel_ok else '❌'} `funnel_stats.csv`\n"
        f"- {'✅' if resume_count else '❌'} 이력서 `{resume_count}`건"
    )

    st.sidebar.divider()
    st.sidebar.subheader("💰 비용")
    bundle = st.session_state.get("bundle")
    cost_total = total_cost(bundle)
    st.sidebar.markdown(f"**누적: `${cost_total:.2f}` / 예산: `${BUDGET_USD:.2f}`**")
    st.sidebar.progress(min(1.0, cost_total / BUDGET_USD))

    st.sidebar.divider()
    st.sidebar.subheader("📥 이력서 업로드")
    uploaded = st.sidebar.file_uploader(
        "업로드 (PDF/DOCX/HWP)",
        type=["pdf", "docx", "hwp"],
        accept_multiple_files=True,
        help="Cloud 배포 환경에서는 텍스트 PDF/DOCX 만 실시간 처리됩니다.",
    )
    if uploaded:
        warned = False
        for uf in uploaded:
            ext = Path(uf.name).suffix.lower()
            if ext == ".hwp":
                warned = True
        if warned:
            st.sidebar.warning(
                "⚠️ HWP / 스캔 PDF 는 Cloud 환경에서 처리할 수 없습니다. "
                "로컬에서 `pipeline.py` 재실행 후 artifacts 를 커밋해주세요."
            )

    st.sidebar.divider()
    run_blocked = cost_total > BUDGET_BLOCK_USD
    force = st.sidebar.checkbox("↻ 강제 재실행", value=False, key="force_rerun")
    run_clicked = st.sidebar.button(
        "▶ 파이프라인 실행",
        type="primary",
        use_container_width=True,
        disabled=run_blocked or not PIPELINE_AVAILABLE,
        help="비용이 $2.50 을 초과하면 실행이 차단됩니다." if run_blocked else None,
    )

    if not PIPELINE_AVAILABLE:
        st.sidebar.error(f"⚠️ pipeline 모듈 로드 실패 — 데모 모드 (artifacts 만 조회 가능).\n\n{PIPELINE_ERROR}")
    if not PII_AVAILABLE:
        st.sidebar.warning(f"⚠️ `src.pii` 미사용 가능 — PII 가드가 제한됩니다.\n\n{PII_ERROR}")

    return run_clicked, force, uploaded or []


# ---------------------------------------------------------------------------
# 파이프라인 실행
# ---------------------------------------------------------------------------
def execute_pipeline(force: bool) -> None:
    if not PIPELINE_AVAILABLE or run_pipeline is None:
        st.error("pipeline 모듈을 불러올 수 없어 실행 불가.")
        return
    status = st.status("파이프라인 실행 중…", expanded=True)
    stage_names = {1: "추출", 2: "분류", 3: "JD 진단", 4: "수요-공급", 5: "Notion Export"}

    def on_step(idx: int, label: str, info: dict) -> None:
        with status:
            detail = ", ".join(f"{k}={v}" for k, v in info.items())
            st.write(f"✅ [{idx}/5] {stage_names.get(idx, label)} — {detail}")

    try:
        paths = run_pipeline(input_dir=DATA_DIR, force=force, on_step=on_step)
        run_dir = Path(paths["run_dir"])
        status.update(label="완료", state="complete", expanded=False)
        st.session_state.bundle = load_artifacts(run_dir)
        st.session_state.last_run_at = datetime.now().isoformat(timespec="seconds")
        st.rerun()
    except Exception as exc:
        status.update(label="실패", state="error", expanded=True)
        st.exception(exc)


# ---------------------------------------------------------------------------
# 탭 1: 대시보드
# ---------------------------------------------------------------------------
def render_dashboard(bundle: dict, jds: list[dict]) -> None:
    extracted = bundle.get("extracted") or []
    classified = bundle.get("classified") or []
    diagnosis = bundle.get("diagnosis") or []

    total_res = len(extracted)
    ok_count = sum(1 for e in extracted if e.get("ok"))
    rate = (ok_count / total_res) if total_res else 0.0
    flagged = sum(1 for d in diagnosis if not d.get("is_healthy", True))
    cost_total = total_cost(bundle)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 이력서", f"{total_res:,}")
    c2.metric("추출 성공률", f"{rate * 100:.1f}%")
    c3.metric("플래그된 JD", f"{flagged}")
    c4.metric("누적 비용", f"${cost_total:.3f}")

    st.divider()

    # JD별 지원자 수 + flag 상태 색상
    jd_title_map = {jd["jd_id"]: jd.get("title", jd["jd_id"]) for jd in jds}
    jd_flag_map: dict[str, str] = {}
    for d in diagnosis:
        jid = d.get("jd_id")
        if d.get("is_healthy", True):
            jd_flag_map[jid] = "정상"
        else:
            lf = d.get("layered_flags") or {}
            # 가장 심각한 레이어 우선
            if lf.get("absolute"):
                jd_flag_map[jid] = "Layer A (절대)"
            elif lf.get("cluster_relative"):
                jd_flag_map[jid] = "Layer B (상대)"
            elif lf.get("supply_demand"):
                jd_flag_map[jid] = "Layer C (수요-공급)"
            else:
                jd_flag_map[jid] = "기타"

    counts: dict[str, int] = {}
    for c in classified:
        jid = c.get("assigned_jd", "UNASSIGNED")
        counts[jid] = counts.get(jid, 0) + 1

    chart_rows = [
        {
            "JD": jd_title_map.get(jid, jid),
            "jd_id": jid,
            "지원자 수": n,
            "상태": jd_flag_map.get(jid, "정상"),
        }
        for jid, n in counts.items()
    ]
    chart_rows.sort(key=lambda r: -r["지원자 수"])

    col_chart, col_top = st.columns([3, 2])
    with col_chart:
        st.markdown("### JD별 분류된 이력서 수")
        if chart_rows and PLOTLY_AVAILABLE:
            df_chart = pd.DataFrame(chart_rows)
            color_map = {
                "정상": "#2ca02c",
                "Layer A (절대)": "#d62728",
                "Layer B (상대)": "#ff7f0e",
                "Layer C (수요-공급)": "#ffbb33",
                "기타": "#888888",
            }
            fig = px.bar(
                df_chart,
                x="JD",
                y="지원자 수",
                color="상태",
                color_discrete_map=color_map,
                hover_data=["jd_id"],
            )
            fig.update_layout(xaxis_tickangle=-30, height=400, margin=dict(t=20, b=80))
            st.plotly_chart(fig, use_container_width=True)
        elif chart_rows:
            st.bar_chart(pd.DataFrame(chart_rows).set_index("JD")["지원자 수"])
        else:
            st.info("분류 결과가 아직 없습니다.")

    with col_top:
        st.markdown("### 🚨 Top-3 문제 JD")
        ranked = []
        for d in diagnosis:
            if d.get("is_healthy", True):
                continue
            lf = d.get("layered_flags") or {}
            total_flags = len(lf.get("absolute") or []) + len(lf.get("cluster_relative") or []) + len(lf.get("supply_demand") or [])
            ranked.append((total_flags, d))
        ranked.sort(key=lambda x: -x[0])
        if not ranked:
            st.success("모든 JD 가 정상입니다.")
        for n, d in ranked[:3]:
            title = d.get("title", d.get("jd_id", "-"))
            lf = d.get("layered_flags") or {}
            all_flags = (lf.get("absolute") or []) + (lf.get("cluster_relative") or []) + (lf.get("supply_demand") or [])
            icons = " ".join(FLAG_ICONS.get(f, "⚪") + f" `{f}`" for f in all_flags)
            st.markdown(f"**{d.get('jd_id')} — {title}** ({n}개 플래그)")
            st.caption(icons)


# ---------------------------------------------------------------------------
# 탭 2: 이력서 분류
# ---------------------------------------------------------------------------
def render_classification(bundle: dict, jds: list[dict]) -> None:
    classified = bundle.get("classified") or []
    extracted = bundle.get("extracted") or []
    ext_by_id = {e.get("resume_id"): e for e in extracted}

    with st.expander("ℹ️ Confidence 점수는 어떻게 산출되나요?"):
        st.markdown(
            "- **산출**: 이력서 ↔ JD 임베딩(`text-embedding-3-small`) 간 **코사인 유사도** (0~1).\n"
            "- **구간 해석**:\n"
            "  - `0.90+` 매우 확실\n"
            "  - `0.70 ~ 0.90` 확실\n"
            "  - `0.50 ~ 0.70` 보통\n"
            "  - `< 0.50` 낮음\n"
            "- **🔁 LLM 뱃지**: top-1 과 top-2 JD 차이(margin)가 0.05 미만인 애매 케이스에 한해 gpt-4o-mini 가 재판정."
        )

    if not classified:
        st.info("분류 결과가 없습니다. 좌측 `▶ 파이프라인 실행` 을 눌러주세요.")
        return

    jd_options = [("전체", None)] + [(f"{jd['jd_id']} — {jd.get('title', '')}", jd["jd_id"]) for jd in jds]
    label_to_id = {lbl: jid for lbl, jid in jd_options}
    pick = st.selectbox("JD 선택", [lbl for lbl, _ in jd_options], index=1 if len(jd_options) > 1 else 0)
    target = label_to_id.get(pick)

    rows: list[dict] = []
    for c in classified:
        if target is not None and c.get("assigned_jd") != target:
            continue
        ext = ext_by_id.get(c.get("resume_id"), {})
        masked = ext.get("masked_text", "") or ""
        conf = float(c.get("confidence") or 0.0)
        path = c.get("decision_path", "embedding")
        rows.append({
            "ID": c.get("resume_id", "-"),
            "경력": first_career_hint(masked, 40),
            "기술": skill_snippet(masked, 100),
            "Confidence": round(conf, 3),
            "JD": c.get("assigned_jd", "-"),
            "재판정": "🔁 LLM" if path == "llm_tiebreak" else "",
        })

    st.caption(f"필터 결과: **{len(rows)}** 건")
    if not rows:
        st.info("해당 조건에 맞는 이력서가 없습니다.")
        return

    df = pd.DataFrame(rows)
    render_safe_df(
        df,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="임베딩 코사인 유사도",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# 탭 3: JD 병목 분석
# ---------------------------------------------------------------------------
def render_bottleneck(bundle: dict, jds: list[dict]) -> None:
    diagnosis = bundle.get("diagnosis") or []

    with st.expander("ℹ️ 삼중 레이어 플래그는 무엇인가요?"):
        st.markdown(
            "- **Layer A — 절대 임계치 (결정론적)**: 지원자 <20명, 서류통과율 <15%, 합격자 0명.\n"
            "- **Layer B — 클러스터 상대 비교**: 같은 직군군(엔지니어/비개발) 평균 대비 50%/70% 미만.\n"
            "- **Layer C — 수요-공급 갭**: 이력서 임베딩 매칭 기반 잠재 후보 대비 실제 지원 비율. "
            "`<0.5` UNDER_ATTRACTED (JD 매력도 문제), `>3.0` OVER_ATTRACTED (JD 문구 모호)."
        )

    if not diagnosis:
        st.info("진단 결과가 없습니다.")
        return

    labels = [f"{d.get('jd_id')} — {d.get('title', '')}" for d in diagnosis]
    pick = st.selectbox("JD 선택", labels, key="bottleneck_jd")
    d = diagnosis[labels.index(pick)]

    # Funnel
    ev_a = (d.get("evidence") or {}).get("absolute") or {}
    applicants = int(ev_a.get("applicants", 0) or 0)
    doc_rate = float(ev_a.get("doc_pass_rate", 0.0) or 0.0)
    hired = int(ev_a.get("hired", 0) or 0)
    doc_pass = int(round(applicants * doc_rate))

    st.markdown("### 📉 3단계 Funnel")
    max_val = max(applicants, 1)
    def bar(val: int, label: str) -> str:
        w = int(40 * val / max_val) if max_val > 0 else 0
        return f"`{'█' * max(w, 0):<40}` **{val}** {label}"

    st.markdown(bar(applicants, "지원자"))
    st.markdown(bar(doc_pass, f"서류통과 ({doc_rate * 100:.1f}%)"))
    st.markdown(bar(hired, "최종합격"))

    st.divider()

    # Layer flags (3 columns)
    st.markdown("### 🚦 플래그")
    lf = d.get("layered_flags") or {}
    ev = d.get("evidence") or {}
    col_a, col_b, col_c = st.columns(3)
    layer_specs = [
        (col_a, "Layer A — 절대", lf.get("absolute") or [], ev.get("absolute") or {}),
        (col_b, "Layer B — 상대", lf.get("cluster_relative") or [], ev.get("cluster_relative") or {}),
        (col_c, "Layer C — 수요-공급", lf.get("supply_demand") or [], ev.get("supply_demand") or {}),
    ]
    for col, title, flags, evidence in layer_specs:
        with col:
            st.markdown(f"**{title}**")
            if not flags:
                st.markdown("🟢 정상")
            for f in flags:
                st.markdown(f"{FLAG_ICONS.get(f, '⚪')} `{f}` — {FLAG_HUMAN.get(f, '')}")
            with st.expander("근거"):
                st.json(evidence)

    st.divider()

    if d.get("is_healthy", True):
        st.success("모든 레이어가 정상입니다.")
        return

    # LLM output
    if d.get("root_cause_hypothesis"):
        st.markdown("### 💡 Root Cause 가설")
        st.info(d["root_cause_hypothesis"])

    chain = d.get("evidence_chain") or []
    if chain:
        st.markdown("### 🔗 Evidence Chain")
        for step in chain:
            st.markdown(f"- {step}")

    quotes = d.get("suspected_quotes") or []
    if quotes:
        st.markdown("### 🔍 의심 문구")
        for q in quotes:
            if isinstance(q, dict):
                st.markdown(f"> {q.get('quote', '')}")
                if q.get("why"):
                    st.caption(f"이유: {q['why']}")
            else:
                st.markdown(f"> {q}")

    edits = d.get("suggested_edits") or []
    if edits:
        st.markdown("### ✏️ 수정 제안")
        for i, e in enumerate(edits, 1):
            if isinstance(e, dict):
                with st.container(border=True):
                    st.markdown(f"**{i}.**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Before**")
                        st.markdown(f"> {e.get('before', '-')}")
                    with c2:
                        st.markdown("**After**")
                        st.markdown(f"> {e.get('after', '-')}")
                    if e.get("rationale"):
                        st.caption(f"근거: {e['rationale']}")
            else:
                st.markdown(f"{i}. {e}")

    refs = d.get("reference_examples") or []
    if refs:
        with st.expander(f"참고한 헬시 JD ({len(refs)}개)"):
            for ref in refs:
                st.markdown(f"**{ref.get('jd_id')} — {ref.get('title', '')}**")
                st.caption(ref.get("description", ""))


# ---------------------------------------------------------------------------
# 탭 4: 수요-공급 갭
# ---------------------------------------------------------------------------
def render_gap(bundle: dict, jds: list[dict]) -> None:
    comparison = bundle.get("comparison") or {}
    gap_table = comparison.get("gap_table") or []
    drift = comparison.get("drift_distributions") or {}

    with st.expander("ℹ️ '잠재 후보'는 어떻게 산출했나요?"):
        st.markdown(
            "- **정의**: 이력서 임베딩 매칭 결과 해당 JD 가 `top-1` 이거나, "
            "`top-2` 중이면서 margin `< 0.05` (애매 케이스)인 이력서 수.\n"
            "- **의미**: 실제 지원 여부와 **무관**한 이론적 적합 후보 수.\n"
            "- **fit_ratio = 잠재 후보 / 실제 지원자**:\n"
            "  - `< 0.5` → UNDER_ATTRACTED (JD 매력도 문제)\n"
            "  - `> 3.0` → OVER_ATTRACTED (JD 문구 모호)"
        )

    if not gap_table:
        st.info("비교 데이터가 없습니다.")
        return

    # Horizontal bar: per JD 2 bars (잠재 / 실제)
    df_gap = pd.DataFrame(gap_table)
    df_long = pd.concat([
        df_gap.assign(metric="잠재 후보", value=df_gap["potential_fit"]),
        df_gap.assign(metric="실제 지원자", value=df_gap["actual_applicants"]),
    ])
    df_long["label"] = df_long["jd_id"] + " — " + df_long["title"].astype(str)

    st.markdown("### 📊 잠재 후보 vs 실제 지원자")
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df_long,
            x="value",
            y="label",
            color="metric",
            barmode="group",
            orientation="h",
            hover_data=["attraction_status", "fit_ratio"],
            color_discrete_map={"잠재 후보": "#1f77b4", "실제 지원자": "#ff7f0e"},
        )
        fig.update_layout(height=max(300, 40 * len(df_gap)), margin=dict(t=20, l=220))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_long.pivot_table(index="label", columns="metric", values="value"))

    # Gap table
    st.markdown("### 📋 상세 테이블")
    display = df_gap[["jd_id", "title", "potential_fit", "actual_applicants", "fit_ratio", "attraction_status"]].rename(
        columns={
            "jd_id": "JD",
            "title": "제목",
            "potential_fit": "잠재",
            "actual_applicants": "실제",
            "fit_ratio": "ratio",
            "attraction_status": "상태",
        }
    )
    render_safe_df(display)

    # Drift drill-down
    under_options = [row["jd_id"] for row in gap_table if row.get("attraction_status") == "UNDER_ATTRACTED"]
    if under_options:
        st.divider()
        st.markdown("### 🔎 UNDER_ATTRACTED JD 의 잠재 후보들은 어디로 지원했나?")
        sel = st.selectbox("JD 선택", under_options, key="drift_jd")
        dist = drift.get(sel, [])
        if not dist:
            st.info("분포 데이터가 없습니다.")
        else:
            df_dist = pd.DataFrame(dist)
            jd_title_map = {jd["jd_id"]: jd.get("title", jd["jd_id"]) for jd in jds}
            df_dist["JD"] = df_dist["assigned_jd"].map(lambda x: f"{x} — {jd_title_map.get(x, '')}")
            if PLOTLY_AVAILABLE:
                fig2 = px.bar(df_dist, x="count", y="JD", orientation="h", text="percentage")
                fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig2.update_layout(height=max(200, 30 * len(df_dist)), margin=dict(t=20, l=200))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(df_dist.set_index("JD")["count"])


# ---------------------------------------------------------------------------
# 탭 5: PII 정책
# ---------------------------------------------------------------------------
def render_pii(bundle: dict) -> None:
    st.markdown("### 📄 PII 정책 문서")
    if PII_POLICY_MD.exists():
        txt = _load_text(PII_POLICY_MD) or ""
        st.markdown(txt)
    else:
        st.warning("📝 `pii_policy.md` 문서 미완성 — 배포 전 작성 필요.")

    st.divider()
    st.markdown("### 🧪 마스킹 감사 (Masking Audit)")
    st.caption("실제 화면에 노출되는 `masked_text` 샘플 5건 — 원본 PII 가 마스킹 토큰으로 치환됐는지 확인합니다.")

    extracted = bundle.get("extracted") or []
    ok_rows = [e for e in extracted if e.get("ok") and e.get("masked_text")]
    if not ok_rows:
        st.info("표시할 추출 결과가 없습니다.")
        return

    rng = random.Random(42)
    samples = rng.sample(ok_rows, k=min(5, len(ok_rows)))
    for s in samples:
        rid = s.get("resume_id", "-")
        fmt = s.get("format", "-")
        masked = (s.get("masked_text") or "")[:150]
        pii_sep = s.get("pii_fields_separated") or {}
        # 절대 내용 노출 금지 — 카운트만
        if isinstance(pii_sep, dict):
            counts = {k: len(v) if isinstance(v, list) else 1 for k, v in pii_sep.items() if v}
        else:
            counts = {}
        with st.expander(f"📄 {rid} · {fmt}"):
            st.code(masked + (" …" if len(s.get("masked_text") or "") > 150 else ""), language="text")
            if counts:
                st.caption("격리된 PII 필드 (카운트만): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
            else:
                st.caption("격리된 PII 필드: 없음")


# ---------------------------------------------------------------------------
# 탭 6: 다운로드
# ---------------------------------------------------------------------------
def render_download(bundle: dict) -> None:
    run_id = bundle.get("run_id", "unknown")
    classified = bundle.get("classified") or []
    diagnosis = bundle.get("diagnosis") or []
    cost = bundle.get("cost") or {}
    report = bundle.get("report_md") or ""

    st.markdown("### ⬇️ 산출물 다운로드")
    st.caption(f"Run ID: `{run_id}`")

    c1, c2 = st.columns(2)

    # Classified CSV (PII-safe: only whitelist columns)
    if classified:
        df_cls = pd.DataFrame([
            {
                "resume_id": c.get("resume_id"),
                "assigned_jd": c.get("assigned_jd"),
                "confidence": c.get("confidence"),
                "margin": c.get("margin"),
                "decision_path": c.get("decision_path"),
            }
            for c in classified
        ])
        csv_bytes = df_cls.to_csv(index=False).encode("utf-8-sig")
    else:
        csv_bytes = b""

    with c1:
        st.download_button(
            "📊 분류 결과 CSV",
            data=csv_bytes,
            file_name=f"classified_{run_id}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not csv_bytes,
        )
        st.download_button(
            "📄 Notion Markdown",
            data=report or "리포트 미생성.",
            file_name=f"notion_report_{run_id}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not report,
        )
    with c2:
        st.download_button(
            "🔍 JD 진단 JSON",
            data=json.dumps(diagnosis, ensure_ascii=False, indent=2),
            file_name=f"diagnosis_{run_id}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not diagnosis,
        )
        st.download_button(
            "💰 비용 로그",
            data=json.dumps(cost, ensure_ascii=False, indent=2),
            file_name=f"cost_{run_id}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not cost,
        )

    if report:
        with st.expander("📖 Notion Markdown 미리보기"):
            st.markdown(report)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    # 초기 번들 로드 (자동 discovery)
    if "bundle" not in st.session_state:
        latest = discover_latest_run()
        st.session_state.bundle = load_artifacts(latest) if latest else None

    run_clicked, force, uploaded = render_sidebar()

    if run_clicked:
        execute_pipeline(force=force)
        return

    bundle = st.session_state.bundle
    jds = load_jds()

    st.title("📋 HR Resume Copilot")
    st.caption("이력서 800건을 JD 12개로 분류하고, 지원자 병목 JD 를 funnel 지표 기반으로 진단합니다.")

    if not bundle:
        st.info(
            "아직 파이프라인 실행 전입니다. 좌측 **▶ 파이프라인 실행** 버튼을 눌러주세요.\n\n"
            "이미 실행된 artifacts 가 있다면 `artifacts/` 하위 최신 run 이 자동 로드됩니다."
        )
        return

    st.caption(f"📂 Run: `{bundle.get('run_id')}` · 최종 수정: "
               f"{datetime.fromtimestamp(Path(bundle['run_dir']).stat().st_mtime).isoformat(timespec='seconds')}")

    tabs = st.tabs([
        "📊 대시보드",
        "📄 이력서 분류",
        "🔍 JD 병목 분석",
        "⚖️ 수요-공급 갭",
        "🔒 PII 정책",
        "⬇️ 다운로드",
    ])
    with tabs[0]:
        render_dashboard(bundle, jds)
    with tabs[1]:
        render_classification(bundle, jds)
    with tabs[2]:
        render_bottleneck(bundle, jds)
    with tabs[3]:
        render_gap(bundle, jds)
    with tabs[4]:
        render_pii(bundle)
    with tabs[5]:
        render_download(bundle)


if __name__ == "__main__":
    main()
