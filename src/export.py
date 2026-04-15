"""Notion-friendly Markdown export for HR Resume Copilot.

Pure template rendering (no LLM calls). Generates a single Markdown string
composed of five sections that paste cleanly into Notion (callouts via
blockquote + emoji, toggles via <details>, standard pipe tables).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

try:
    from src.pii import assert_pii_safe, PIILeakError  # type: ignore
except Exception:  # pragma: no cover - fall back when run standalone
    class PIILeakError(RuntimeError):
        pass

    _FALLBACK_PHONE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
    _FALLBACK_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

    def assert_pii_safe(text: str) -> None:
        leaks: list[str] = []
        if _FALLBACK_PHONE.search(text):
            leaks.append("phone")
        if _FALLBACK_EMAIL.search(text):
            leaks.append("email")
        if leaks:
            raise PIILeakError(f"PII leak detected: {leaks}")


_FLAG_ICONS: dict[str, str] = {
    "ZERO_HIRED": "🔴",
    "LOW_APPLICANTS_ABS": "🔴",
    "LOW_DOC_PASS_ABS": "🔴",
    "LOW_APPLICANTS_REL": "🟡",
    "LOW_DOC_PASS_REL": "🟡",
    "UNDER_ATTRACTED": "🟠",
    "OVER_ATTRACTED": "🟠",
}

_CATEGORY_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("백엔드", "backend", "서버"), "🔧"),
    (("프론트", "frontend", "front-end"), "⚛️"),
    (("ml", "머신러닝", "ai", "데이터 사이언"), "🤖"),
    (("데이터 엔지니어", "데이터엔지니어", "data engineer"), "🗄️"),
    (("데이터 분석", "데이터분석", "analyst"), "📊"),
    (("보안", "security"), "🔒"),
    (("devops", "sre", "인프라"), "⚙️"),
    (("ios", "android", "모바일", "mobile"), "📱"),
    (("qa", "테스트", "test"), "🧪"),
    (("디자인", "design"), "🎨"),
    (("pm", "기획", "product manager"), "📋"),
    (("hr", "인사", "people"), "🧑\u200d💼"),
    (("마케팅", "marketing"), "📣"),
    (("영업", "sales"), "💼"),
]


_YEAR_LINE_RE = re.compile(r".*?\d+\s*년.*")
_LEAK_PHONE_RE = re.compile(r"\b\d{3}-\d{4}-\d{4}\b")
_LEAK_EMAIL_RE = re.compile(r"@\w+\.\w+")


def _format_flag_icon(flag: str) -> str:
    return _FLAG_ICONS.get(flag, "⚪")


def _category_emoji(title: str) -> str:
    low = (title or "").lower()
    for keywords, emoji in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in low:
                return emoji
    return "📌"


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def _escape_cell(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ")


def _first_career_hint(masked_text: str) -> str:
    if not masked_text:
        return "-"
    for line in masked_text.splitlines():
        if "년" in line and re.search(r"\d", line):
            cleaned = line.strip()
            if cleaned:
                return _truncate(cleaned, 40)
    return "-"


def _fmt_pct(x: float | int | None) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x) * 100:.1f}%" if float(x) <= 1.0 else f"{float(x):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_money(x: float | int | None) -> str:
    if x is None:
        return "$0.00"
    try:
        return f"${float(x):.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmt_ratio(x: float | int | None) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "-"


def _jd_lookup(jds: Iterable[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jd in jds or []:
        jid = jd.get("jd_id") or jd.get("id")
        if jid:
            out[jid] = jd
    return out


def _section_header(md: list[str]) -> None:
    md.append("# HR Resume Copilot 리포트")
    md.append("")


def _section_summary(md: list[str], meta: dict, cost: dict, diagnosis: list[dict], jds_count: int) -> None:
    flagged = sum(1 for d in diagnosis if not d.get("is_healthy", True))
    total_cost = 0.0
    if isinstance(cost, dict):
        total_cost = float(cost.get("total_cost_usd") or cost.get("total_usd") or cost.get("total") or 0.0)
    date_str = meta.get("date", "-")
    total_res = meta.get("total_resumes", 0)
    rate = meta.get("extraction_success_rate", 0.0)
    md.append(
        f"> 💡 **요약**: {date_str} 기준 이력서 {total_res}건을 {jds_count}개 JD로 분류했습니다. "
        f"추출 성공률 {_fmt_pct(rate)}, 병목 플래그가 붙은 JD {flagged}개. LLM 비용 {_fmt_money(total_cost)}."
    )
    md.append("")


def _section_overview(md: list[str], meta: dict, jds: list[dict], diagnosis: list[dict], cost: dict) -> None:
    md.append("## §1 전체 개요")
    md.append("")
    flagged = sum(1 for d in diagnosis if not d.get("is_healthy", True))
    total_cost = 0.0
    if isinstance(cost, dict):
        total_cost = float(cost.get("total_cost_usd") or cost.get("total_usd") or cost.get("total") or 0.0)
    md.append("| 지표 | 값 |")
    md.append("|---|---|")
    md.append(f"| 총 이력서 | {meta.get('total_resumes', 0)}건 |")
    md.append(f"| 추출 성공률 | {_fmt_pct(meta.get('extraction_success_rate', 0.0))} |")
    md.append(f"| 총 JD | {len(jds)}개 |")
    md.append(f"| 플래그된 JD | {flagged}개 |")
    md.append(f"| LLM 비용 | {_fmt_money(total_cost)} |")
    md.append("")


def _section_classification(
    md: list[str],
    classified: list[dict],
    extracted: list[dict],
    jds: list[dict],
) -> None:
    md.append("## §2 포지션별 이력서 분류")
    md.append("")
    ext_by_id = {e.get("resume_id"): e for e in extracted or []}
    by_jd: dict[str, list[dict]] = {}
    for c in classified or []:
        by_jd.setdefault(c.get("assigned_jd", "UNASSIGNED"), []).append(c)

    jd_order = [jd.get("jd_id") for jd in jds] + [k for k in by_jd if k not in {jd.get("jd_id") for jd in jds}]
    jd_lookup = _jd_lookup(jds)

    for jd_id in jd_order:
        if not jd_id:
            continue
        rows = by_jd.get(jd_id, [])
        jd = jd_lookup.get(jd_id, {"title": jd_id})
        title = jd.get("title", jd_id)
        emoji = _category_emoji(title)
        md.append(f"### {emoji} {title} ({len(rows)}건)")
        md.append("")
        if not rows:
            md.append("_분류된 이력서 없음._")
            md.append("")
            continue
        md.append("<details><summary>이력서 목록</summary>")
        md.append("")
        md.append("| ID | 경력 | Confidence |")
        md.append("|---|---|---|")
        shown = rows[:20]
        for r in shown:
            rid = r.get("resume_id", "-")
            ext = ext_by_id.get(rid, {})
            career = _first_career_hint(ext.get("masked_text", ""))
            conf = r.get("confidence")
            conf_str = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "-"
            md.append(f"| {_escape_cell(rid)} | {_escape_cell(career)} | {conf_str} |")
        if len(rows) > 20:
            md.append(f"")
            md.append(f"_(+ {len(rows) - 20}건 생략)_")
        md.append("")
        md.append("</details>")
        md.append("")


def _section_diagnosis(md: list[str], diagnosis: list[dict]) -> None:
    md.append("## §3 JD 병목 진단")
    md.append("")
    for d in diagnosis or []:
        title = d.get("title", d.get("jd_id", "-"))
        if d.get("is_healthy", False):
            md.append(f"### ✅ {title}")
            md.append("")
            md.append("> 정상 (모든 layer flag clear)")
            md.append("")
            continue

        md.append(f"### ⚠️ {title}")
        md.append("")

        flags = d.get("layered_flags") or []
        if isinstance(flags, dict):
            flat: list[str] = []
            for v in flags.values():
                if isinstance(v, list):
                    flat.extend(v)
                elif v:
                    flat.append(str(v))
            flags = flat
        if flags:
            md.append("**플래그**")
            md.append("")
            for f in flags:
                md.append(f"- {_format_flag_icon(f)} `{f}`")
            md.append("")

        chain = d.get("evidence_chain")
        if chain:
            md.append("**근거 체인**")
            md.append("")
            if isinstance(chain, list):
                for step in chain:
                    md.append(f"- {step}")
            else:
                md.append(f"{chain}")
            md.append("")

        quotes = d.get("suspected_quotes") or []
        if quotes:
            md.append("**문제 문구 인용**")
            md.append("")
            for q in quotes:
                md.append(f"> {q}")
            md.append("")

        edits = d.get("suggested_edits") or []
        if edits:
            md.append("**수정 제안**")
            md.append("")
            for i, e in enumerate(edits, 1):
                if isinstance(e, dict):
                    before = _truncate(e.get("before", ""), 180)
                    after = _truncate(e.get("after", ""), 180)
                    md.append(f"{i}. **Before**: {before}")
                    md.append(f"   **After**: {after}")
                else:
                    md.append(f"{i}. {e}")
            md.append("")


def _section_gap(md: list[str], comparison: dict, diagnosis: list[dict]) -> None:
    md.append("## §4 수요-공급 갭 상세")
    md.append("")
    gap_table = (comparison or {}).get("gap_table") or []
    drift = (comparison or {}).get("drift_distributions") or {}

    md.append("| JD | 잠재후보 | 실제지원 | ratio | 상태 |")
    md.append("|---|---|---|---|---|")
    for row in gap_table:
        jd_label = row.get("title") or row.get("jd_id", "-")
        potential = row.get("potential_fit", row.get("potential", "-"))
        actual = row.get("actual_applicants", row.get("actual", "-"))
        ratio = _fmt_ratio(row.get("fit_ratio", row.get("ratio")))
        status = row.get("attraction_status", row.get("status", "-"))
        md.append(
            f"| {_escape_cell(jd_label)} | {_escape_cell(potential)} | "
            f"{_escape_cell(actual)} | {ratio} | {_escape_cell(status)} |"
        )
    md.append("")

    diag_lookup = {d.get("jd_id"): d for d in diagnosis or []}
    under = [r for r in gap_table if (r.get("attraction_status") or r.get("status")) == "UNDER_ATTRACTED"]
    if under:
        md.append("### 실제 지원자는 어디로 갔나? (UNDER_ATTRACTED JD)")
        md.append("")
        for row in under:
            jd_id = row.get("jd_id")
            jd_title = row.get("title") or diag_lookup.get(jd_id, {}).get("title", jd_id)
            md.append(f"#### {jd_title}")
            md.append("")
            dist = drift.get(jd_id) or []
            if not dist:
                md.append("_분포 데이터 없음._")
                md.append("")
                continue
            md.append("| 실제 지원 JD | 건수 |")
            md.append("|---|---|")
            for item in dist:
                if isinstance(item, dict):
                    label = item.get("title") or item.get("jd_id", "-")
                    cnt = item.get("count", item.get("n", "-"))
                else:
                    label, cnt = str(item), "-"
                md.append(f"| {_escape_cell(label)} | {_escape_cell(cnt)} |")
            md.append("")


def _section_pii(md: list[str]) -> None:
    md.append("## §5 PII 처리 정책 요약")
    md.append("")
    md.append(
        "본 리포트의 모든 이력서 텍스트는 `src/pii.py`의 2-tier 정책에 따라 처리되었습니다. "
        "이름·전화·이메일·주소는 마스킹 토큰으로 치환되고, 학력·나이·성별·사진·출신지역은 "
        "텍스트에서 제거된 뒤 별도 필드로 격리되어 LLM 프롬프트에 절대 포함되지 않습니다."
    )
    md.append("")
    md.append("자세한 내용: [pii_policy.md](pii_policy.md)")
    md.append("")


def _footer(md: list[str], meta: dict, cost: dict) -> None:
    total_cost = 0.0
    if isinstance(cost, dict):
        total_cost = float(cost.get("total_cost_usd") or cost.get("total_usd") or cost.get("total") or 0.0)
    md.append("---")
    md.append(f"*생성일: {meta.get('date', '-')} | 생성 비용: {_fmt_money(total_cost)}*")
    md.append("")


def _pii_guard(md_text: str) -> None:
    phone_m = _LEAK_PHONE_RE.search(md_text)
    if phone_m:
        raise PIILeakError(f"PII leak (phone) in Markdown: {phone_m.group(0)!r}")
    email_m = _LEAK_EMAIL_RE.search(md_text)
    if email_m:
        snippet = md_text[max(0, email_m.start() - 20) : email_m.end() + 20]
        raise PIILeakError(f"PII leak (email-like) in Markdown near: {snippet!r}")
    try:
        assert_pii_safe(md_text)
    except PIILeakError:
        raise
    except Exception:
        pass


def build_notion_markdown(
    classified: list[dict],
    extracted: list[dict],
    diagnosis: list[dict],
    comparison: dict,
    cost: dict,
    jds: list[dict],
    meta: dict,
) -> str:
    md: list[str] = []
    _section_header(md)
    _section_summary(md, meta, cost, diagnosis, len(jds))
    _section_overview(md, meta, jds, diagnosis, cost)
    _section_classification(md, classified, extracted, jds)
    _section_diagnosis(md, diagnosis)
    _section_gap(md, comparison, diagnosis)
    _section_pii(md)
    _footer(md, meta, cost)
    text = "\n".join(md)
    _pii_guard(text)
    return text


def save_markdown(md: str, output_path: Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8", newline="\n")
    return p


if __name__ == "__main__":
    import tempfile

    jds = [
        {"jd_id": "JD001", "title": "백엔드 엔지니어"},
        {"jd_id": "JD002", "title": "프론트엔드 엔지니어"},
        {"jd_id": "JD003", "title": "보안 엔지니어"},
    ]

    classified = [
        {"resume_id": f"R{i:03d}", "assigned_jd": "JD001", "confidence": 0.91 - i * 0.01,
         "top2": [("JD001", 0.9), ("JD002", 0.6)], "margin": 0.3, "decision_path": "embed"}
        for i in range(25)
    ] + [
        {"resume_id": "R100", "assigned_jd": "JD002", "confidence": 0.82,
         "top2": [], "margin": 0.2, "decision_path": "embed"},
        {"resume_id": "R101", "assigned_jd": "JD003", "confidence": 0.74,
         "top2": [], "margin": 0.1, "decision_path": "llm_tiebreak"},
    ]

    extracted = [
        {"resume_id": f"R{i:03d}", "masked_text": f"[MASK_NAME]\n경력 {i % 10 + 1}년 Python/Django",
         "format": "pdf", "ok": True}
        for i in range(25)
    ] + [
        {"resume_id": "R100", "masked_text": "[MASK_NAME]\n경력 3년 React/TypeScript", "format": "docx", "ok": True},
        {"resume_id": "R101", "masked_text": "[MASK_NAME]\n경력 5년 침투테스트", "format": "pdf", "ok": True},
    ]

    diagnosis = [
        {
            "jd_id": "JD001", "title": "백엔드 엔지니어", "cluster": "engineering",
            "layered_flags": [], "evidence": {}, "is_healthy": True,
            "root_cause_hypothesis": "", "evidence_chain": [], "suspected_quotes": [],
            "suggested_edits": [], "reference_examples": [],
        },
        {
            "jd_id": "JD002", "title": "프론트엔드 엔지니어", "cluster": "engineering",
            "layered_flags": ["LOW_DOC_PASS_REL", "UNDER_ATTRACTED"],
            "evidence": {"doc_pass_rate": 0.08},
            "is_healthy": False,
            "root_cause_hypothesis": "JD 문구가 기술 스택을 명시하지 않아 매력도가 낮음",
            "evidence_chain": [
                "서류 통과율 8% (클러스터 평균 18% 대비 44%)",
                "fit_ratio 0.3 — 적합 후보 50명 중 15명만 실제 지원",
            ],
            "suspected_quotes": [
                "‘다양한 프론트엔드 기술에 능숙한 분’ — 구체 스택 부재",
            ],
            "suggested_edits": [
                {"before": "다양한 프론트엔드 기술", "after": "React 18+ / TypeScript / Next.js 경험"},
                {"before": "커뮤니케이션 능력 우수", "after": "디자이너·PM과 주 2회 이상 싱크 경험"},
            ],
            "reference_examples": [],
        },
        {
            "jd_id": "JD003", "title": "보안 엔지니어", "cluster": "engineering",
            "layered_flags": ["ZERO_HIRED"], "evidence": {"hired": 0},
            "is_healthy": False,
            "root_cause_hypothesis": "요구 경력이 비현실적으로 높음",
            "evidence_chain": ["합격자 0명", "지원자 수는 평균 수준"],
            "suspected_quotes": ["10년 이상 침투테스트 경력 필수"],
            "suggested_edits": [{"before": "10년 이상", "after": "5년 이상 또는 동등한 프로젝트 이력"}],
            "reference_examples": [],
        },
    ]

    comparison = {
        "gap_table": [
            {"jd_id": "JD001", "title": "백엔드 엔지니어", "potential_fit": 80, "actual_applicants": 90,
             "fit_ratio": 0.89, "attraction_status": "HEALTHY"},
            {"jd_id": "JD002", "title": "프론트엔드 엔지니어", "potential_fit": 50, "actual_applicants": 15,
             "fit_ratio": 0.30, "attraction_status": "UNDER_ATTRACTED"},
            {"jd_id": "JD003", "title": "보안 엔지니어", "potential_fit": 10, "actual_applicants": 40,
             "fit_ratio": 4.00, "attraction_status": "OVER_ATTRACTED"},
        ],
        "drift_distributions": {
            "JD002": [
                {"jd_id": "JD001", "title": "백엔드 엔지니어", "count": 20},
                {"jd_id": "JD003", "title": "보안 엔지니어", "count": 5},
            ],
        },
    }

    cost = {"total_cost_usd": 0.42, "calls": 12}
    meta = {"date": "2026-04-15", "total_resumes": 800, "extraction_success_rate": 0.93}

    md = build_notion_markdown(classified, extracted, diagnosis, comparison, cost, jds, meta)

    out = Path(tempfile.gettempdir()) / "test_report.md"
    save_markdown(md, out)
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(f"length={len(md)} path={out}")
    print("--- first 500 chars ---")
    sys.stdout.write(md[:500])
    sys.stdout.write("\n")
