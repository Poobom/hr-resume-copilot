"""PII masking and isolation for HR Resume Copilot.

Implements two-tier protection per 2026-04-15 법무팀 공지:
  - Tier 1 (mask):   이름 / 전화 / 이메일 / 주소 → tokenized in text
  - Tier 2 (isolate): 학력 / 나이 / 성별 / 사진 / 출신지역 → removed from text,
    separated into `pii_fields_separated` that must never enter an LLM prompt.

Call `assert_pii_safe()` immediately before any prompt construction.
Call `safe_display_row()` in the Streamlit layer to whitelist displayable columns.
"""
from __future__ import annotations

import re
from typing import Iterable


class PIILeakError(RuntimeError):
    pass


class ForbiddenFieldError(RuntimeError):
    pass


_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_ADDR_RE = re.compile(
    r"(?:서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시"
    r"|서울시|부산시|대구시|인천시|광주시|대전시|울산시"
    r"|서울|부산|대구|인천|광주|대전|울산|세종"
    r"|경기도|강원도|강원특별자치도|충청북도|충청남도|전라북도|전북특별자치도|전라남도|경상북도|경상남도|제주특별자치도"
    r"|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
    r"\s*[가-힣]{1,10}(?:시|군|구)"
)
_NAME_LABELED_RE = re.compile(r"(성명|이름|지원자명|성함)\s*[:：]\s*([가-힣]{2,4})")
_NAME_HEADER_RE = re.compile(r"(?m)^\s*([가-힣]{2,4})\s*(?:\(.*?\))?\s*$")

_EDU_RE = re.compile(
    r"(?:[가-힣A-Za-z]+(?:대학교|대학|고등학교|고교|중학교))"
    r"(?:\s*(?:[가-힣A-Za-z]+학과|[가-힣A-Za-z]+학부|[가-힣A-Za-z]+전공))?"
    r"(?:\s*(?:\d{4}\s*년?\s*(?:졸업|입학|수료)?|졸업|재학|수료))?"
)
_AGE_RE = re.compile(r"(?:만\s*)?\d{1,3}\s*(?:세|살)\b")
_BIRTH_RE = re.compile(
    r"(?:생년월일|생일)?\s*[:：]?\s*"
    r"(?:\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{4}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?|\d{6}-\d{7})"
)
_GENDER_RE = re.compile(r"성별\s*[:：]\s*[가-힣A-Za-z]+|\b(?:남자|여자|남성|여성)\b")
_PHOTO_B64_RE = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]{40,}")
_PHOTO_LABEL_RE = re.compile(r"(?m)^\s*(?:사진|증명사진|프로필\s*사진)\s*[:：]?.*$")
_HOMETOWN_RE = re.compile(r"(?:출신(?:지역|지)?|고향|본적|본관)\s*[:：]?\s*[가-힣]{2,15}")


def _collect(pattern: re.Pattern[str], text: str) -> list[str]:
    out: list[str] = []
    for m in pattern.finditer(text):
        out.append(m.group(0).strip())
    return out


def mask_pii(text: str) -> tuple[str, dict[str, list[str]]]:
    separated: dict[str, list[str]] = {
        "education": [],
        "age": [],
        "gender": [],
        "photo": [],
        "hometown": [],
    }

    separated["photo"].extend(_collect(_PHOTO_B64_RE, text))
    text = _PHOTO_B64_RE.sub("[FILTERED]", text)
    separated["photo"].extend(_collect(_PHOTO_LABEL_RE, text))
    text = _PHOTO_LABEL_RE.sub("[FILTERED]", text)

    separated["education"].extend(_collect(_EDU_RE, text))
    text = _EDU_RE.sub("[FILTERED]", text)

    separated["hometown"].extend(_collect(_HOMETOWN_RE, text))
    text = _HOMETOWN_RE.sub("[FILTERED]", text)

    separated["age"].extend(_collect(_BIRTH_RE, text))
    text = _BIRTH_RE.sub("[FILTERED]", text)
    separated["age"].extend(_collect(_AGE_RE, text))
    text = _AGE_RE.sub("[FILTERED]", text)

    separated["gender"].extend(_collect(_GENDER_RE, text))
    text = _GENDER_RE.sub("[FILTERED]", text)

    text = _PHONE_RE.sub("[MASK_PHONE]", text)
    text = _EMAIL_RE.sub("[MASK_EMAIL]", text)
    text = _ADDR_RE.sub("[MASK_ADDR]", text)

    text = _NAME_LABELED_RE.sub(lambda m: f"{m.group(1)}: [MASK_NAME]", text)
    text = _NAME_HEADER_RE.sub("[MASK_NAME]", text)

    for key, vals in separated.items():
        seen: set[str] = set()
        deduped: list[str] = []
        for v in vals:
            if v and v not in seen:
                seen.add(v)
                deduped.append(v)
        separated[key] = deduped

    return text, separated


def assert_pii_safe(text: str) -> None:
    leaks: list[str] = []
    if _PHONE_RE.search(text):
        leaks.append("phone")
    if _EMAIL_RE.search(text):
        leaks.append("email")
    if _NAME_HEADER_RE.search(text):
        leaks.append("name_header")
    if leaks:
        raise PIILeakError(f"PII leak detected before LLM call: {leaks}")


def safe_display_row(row: dict, allowed_fields: set[str]) -> dict:
    forbidden = set(row.keys()) - allowed_fields
    if forbidden:
        raise ForbiddenFieldError(f"Forbidden fields in display row: {sorted(forbidden)}")
    return {k: row[k] for k in row if k in allowed_fields}


def get_masking_samples(before_after_pairs: Iterable[tuple[str, str]], n: int = 5) -> str:
    pairs = list(before_after_pairs)[:n]
    lines: list[str] = ["# PII 마스킹 감사 샘플", ""]
    for i, (before, after) in enumerate(pairs, 1):
        lines.append(f"## 샘플 {i}")
        lines.append("")
        lines.append("**Before**")
        lines.append("")
        lines.append("```")
        lines.append(before)
        lines.append("```")
        lines.append("")
        lines.append("**After**")
        lines.append("")
        lines.append("```")
        lines.append(after)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    samples = [
        "성명: 김민수\n연락처: 010-1234-5678\n이메일: minsu.kim@example.com\n주소: 서울시 강남구 테헤란로 123\n성별: 남자\n나이: 32세\n서울대학교 컴퓨터공학과 2015년 졸업\n출신지역: 부산",
        "이정훈\n010-9876-5432\nlee@corp.co.kr\n경기도 성남시 분당구\n생년월일: 1990-05-20\n여성 / 고려대학교 경영학과 졸업\n고향: 대구",
        "박서연 (Seoyeon Park)\nphone: 02-555-1234\nemail: park@test.io\n사진: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==\n연세대학교 전기전자공학부 2018 졸업\n만 28세 남성\n본적: 전라남도 순천시",
    ]

    pairs: list[tuple[str, str]] = []
    for i, s in enumerate(samples, 1):
        masked, separated = mask_pii(s)
        print(f"=== Sample {i} ===")
        print("[Before]")
        print(s)
        print("[After]")
        print(masked)
        print("[Separated]")
        for k, v in separated.items():
            print(f"  {k}: {v}")
        try:
            assert_pii_safe(masked)
            print("[assert_pii_safe] OK")
        except PIILeakError as e:
            print(f"[assert_pii_safe] LEAK: {e}")
        print()
        pairs.append((s, masked))

    print("=== safe_display_row ===")
    row = {"resume_id": "R001", "skills": "Python", "name": "김민수", "age": "32"}
    try:
        print(safe_display_row(row, {"resume_id", "skills", "name", "age"}))
        print(safe_display_row(row, {"resume_id", "skills"}))
    except ForbiddenFieldError as e:
        print(f"blocked: {e}")

    try:
        safe_display_row(row, {"resume_id"})
    except ForbiddenFieldError as e:
        print(f"blocked: {e}")

    print()
    print("=== get_masking_samples (truncated) ===")
    print(get_masking_samples(pairs, n=2)[:400])
