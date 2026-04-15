# PII 정책 (HR Resume Copilot)

본 문서는 이력서 800건과 JD 12건을 처리하는 HR Resume Copilot의 개인정보(PII) 보호 정책을 정의한다. 2026-04-15 법무팀 공지(가산 채용 폐지)를 기준 근거로 한다.

## 1. 적용 범위

본 정책은 다음 구성요소 전체에 적용된다.

- 입력 데이터: `task02_data/resumes/` 내 PDF/DOCX/HWP(X) 이력서 800건, `task02_data/jds.csv`(JD 12건), `task02_data/funnel_stats.csv`
- 파이프라인 전 단계: Stage 1 추출(`src/extract.py`) → Stage 2 분류(`src/classify.py`) → Stage 3 진단(`src/diagnose.py`) → Stage 4 수요-공급 갭(`src/compare.py`) → Stage 5 Notion Markdown export(`src/export.py`)
- UI: `app.py` Streamlit 6탭 (대시보드 / 이력서 분류 / JD 병목 분석 / 수요-공급 갭 / PII 정책 / 다운로드)
- LLM 호출 경로: OpenAI gpt-4o-mini(재판정·진단) 및 text-embedding-3-small(임베딩), Vision API(스캔 PDF OCR)

## 2. PII 분류

이력서에 등장하는 정보를 아래 3개 등급으로 나눈다.

### 2.1 마스킹 대상 (원문 → `[MASK_*]` 치환)

정규식 기반 치환. 실제 구현은 `src/pii.py:mask_pii()`.

| 항목 | 탐지 패턴 | 치환 토큰 | 근거 |
|---|---|---|---|
| 이름 | 라벨형(`성명|이름|지원자명|성함 : 홍길동`) / 문서 상단 단독 헤더(`^[가-힣]{2,4}$`) | `[MASK_NAME]` | 개인 식별 직접 정보 |
| 전화번호 | `\d{2,3}-\d{3,4}-\d{4}` | `[MASK_PHONE]` | 직접 연락처 |
| 이메일 | `[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}` | `[MASK_EMAIL]` | 직접 연락처 |
| 주소 | 시·도(서울/부산/경기/제주 등) + `구|군|시` 조합 | `[MASK_ADDR]` | 거주지 추정 가능 |

### 2.2 완전 격리 대상 (LLM/UI 모두 미노출)

**근거**: 법무팀 2026-04-15 공지 — 학력·나이·성별·외모·출신지역을 채용 평가 변수로 사용 금지(가산 채용 폐지).

본문에서 `[FILTERED]`로 치환한 뒤, 별도 딕셔너리(`pii_fields_separated`)로 격리하여 **어떤 LLM 프롬프트에도 포함하지 않는다.** 격리된 필드는 디버깅 목적의 로컬 검수 이외에는 UI에도 전달되지 않는다.

| 항목 | 처리 방법 | 법적·정책 근거 |
|---|---|---|
| 학력 (학교·학과·졸업년도) | `_EDU_RE`로 추출 후 `[FILTERED]`, `pii_fields_separated.education`에 격리 | 2026-04-15 공지 |
| 나이·생년월일 | `_AGE_RE` / `_BIRTH_RE`로 추출 후 `[FILTERED]`, `.age` 격리 | 2026-04-15 공지 |
| 성별 | `_GENDER_RE`(`성별:`, `남/여성` 등) 추출 후 `[FILTERED]`, `.gender` 격리 | 2026-04-15 공지 |
| 증명사진(base64/바이너리) | `_PHOTO_B64_RE`, `_PHOTO_LABEL_RE`로 추출 후 `[FILTERED]`, `.photo` 격리 | 2026-04-15 공지 + 생체정보 취급 회피 |
| 출신지역 | `_HOMETOWN_RE`(`출신|고향|본적|본관`) 추출 후 `[FILTERED]`, `.hometown` 격리 | 2026-04-15 공지 |

### 2.3 평가 사용 허용 (마스킹 없이)

다음은 채용 적합도 판단에 필요한 정보로, 마스킹·격리 대상이 아니다(단, 위 PII 치환 이후의 텍스트에서만 사용).

- 경력 연수·재직 회사명
- 기술 스택·자격증
- 프로젝트 설명 및 직무 키워드

## 3. 데이터 흐름 및 보호 지점

```
[원본 PDF/DOCX/HWP] (task02_data/resumes/, .gitignore 차단)
    |
    v  Stage 1 추출 (src/extract.py)
    |  - PDF: pdfplumber → <50자면 PyMuPDF 렌더 + Vision OCR
    |  - DOCX: python-docx
    |  - HWP: HWPX(ZIP) → Preview/PrvText.txt → Contents/section*.xml <hs:t>
    |         (실패 시 LibreOffice CLI fallback)
    |
[raw_text — 메모리 내에서만 존재, 파일에 저장되지 않음]
    |
    v  *** 마스킹 발생 지점: src/pii.py:mask_pii() ***
    |
[masked_text + pii_fields_separated]
    |
    v  캐시 저장: artifacts/resume_cache/{sha1}.json
    |  (raw_text 필드 미포함 — masked_text와 pii_fields_separated만 기록)
    |
    v  Stage 2 분류 / Stage 3 진단 / Stage 4 갭 집계
    |  - 모든 LLM 프롬프트 직전: assert_pii_safe(prompt) 호출
    |
    v  Stage 5 Notion Markdown export (src/export.py)
    |  - 최종 텍스트에 대해 _pii_guard(md_text) 실행
    |
[UI / Markdown 다운로드]
    - safe_display_row(row, allowed_fields)로 화이트리스트된 컬럼만 표시
```

가드레일 함수:

- `src.pii.mask_pii(text) -> (masked_text, pii_fields_separated)` — 본문 치환 및 격리
- `src.pii.assert_pii_safe(text)` — LLM 프롬프트 직전 전화·이메일·이름 헤더 잔존 시 `PIILeakError`
- `src.pii.safe_display_row(row, allowed_fields)` — UI 표시 전 허용되지 않은 키 존재 시 `ForbiddenFieldError`
- `src.export._pii_guard(md_text)` — 최종 Markdown에 전화번호 등 누설 시 `PIILeakError`

## 4. 저장·보관·폐기

- **원본 이력서**: `task02_data/resumes/`, `resumes_raw/`는 `.gitignore`로 차단(리포 커밋 금지). 로컬 개발자 머신에서만 보관.
- **캐시**: `artifacts/resume_cache/{sha1}.json`에는 `raw_text`를 저장하지 않고 `raw_text_length`, `masked_text`, `pii_fields_separated`, 추출 메타데이터만 기록. 선처리(pre-bake) 결과를 Git에 커밋하여 Streamlit Cloud에서 재사용.
- **런 아티팩트**: `artifacts/{run_sha1}/01_extracted.json` 등 스테이지별 산출물 역시 마스킹된 상태만 포함. 원본 텍스트 디렉토리(`artifacts/**/raw_texts/`)는 `.gitignore`로 차단.
- **비밀**: `.env`는 `.gitignore`. Streamlit Cloud 배포 시 API 키는 `secrets.toml`로 주입하며 리포에 커밋하지 않음.
- **로컬 폐기**: 과제 종료 시 `rm -rf artifacts/` 및 `task02_data/resumes/` 삭제 권장.

## 5. 마스킹 감사 샘플

아래는 `src/pii.py` `__main__` 블록과 동일한 구조로 생성한 **가상 예시**다. 실제 이력서가 아니며, 실 데이터 감사 로그는 별도 로컬 파일로만 검수한다.

### 샘플 1

**Before**
```
성명: 김민수
연락처: 010-1234-5678
이메일: minsu.kim@example.com
주소: 서울시 강남구 테헤란로 123
성별: 남자
나이: 32세
서울대학교 컴퓨터공학과 2015년 졸업
출신지역: 부산
```

**After**
```
성명: [MASK_NAME]
연락처: [MASK_PHONE]
이메일: [MASK_EMAIL]
주소: [MASK_ADDR] 테헤란로 123
[FILTERED]
[FILTERED]
[FILTERED] [FILTERED]
[FILTERED]
```

### 샘플 2

**Before**
```
이정훈
010-9876-5432
lee@corp.co.kr
경기도 성남시 분당구
생년월일: 1990-05-20
여성 / 고려대학교 경영학과 졸업
```

**After**
```
[MASK_NAME]
[MASK_PHONE]
[MASK_EMAIL]
[MASK_ADDR]
[FILTERED]
[FILTERED] / [FILTERED]
```

### 샘플 3

**Before**
```
박서연 (Seoyeon Park)
phone: 02-555-1234
email: park@test.io
사진: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEA...
연세대학교 전기전자공학부 2018 졸업
만 28세 남성
본적: 전라남도 순천시
```

**After**
```
[MASK_NAME]
phone: [MASK_PHONE]
email: [MASK_EMAIL]
[FILTERED]
[FILTERED]
[FILTERED] [FILTERED]
[FILTERED]
```

### 샘플 4 (영문 이름 헤더 — 한계 시연)

**Before**
```
Name: Alice Chen
HP: 010-5555-1212
email: alice@tech.io
인천광역시 연수구 송도동
여성 28세
한양대학교 전산학부 2019 졸업
```

**After**
```
Name: Alice Chen
HP: [MASK_PHONE]
email: [MASK_EMAIL]
[MASK_ADDR] 송도동
[FILTERED] [FILTERED]
[FILTERED]
```

> ⚠️ `_NAME_LABELED_RE`의 라벨 집합이 한글(`성명|이름|지원자명|성함`)에 한정되고 `_NAME_HEADER_RE`가 `[가-힣]{2,4}`이라 영문 이름("Alice Chen") 및 영문 라벨("Name:")은 마스킹되지 않는다. 6장 "영문 이름 마스킹" 한계에 대응.

### 샘플 5 (생년월일·본관·증명사진 base64 혼합)

**Before**
```
지원자명: 최영희
Tel: 02-1234-5678
choi@mail.com
대구광역시 수성구 범어동
1988년 3월 15일생 / 남성
경희대학교 경영학부 졸업
본관: 경주
사진: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...
```

**After**
```
지원자명: [MASK_NAME]
Tel: [MASK_PHONE]
[MASK_EMAIL]
[MASK_ADDR] 범어동
[FILTERED]생 / [FILTERED]
[FILTERED]
[FILTERED]
[FILTERED]
```

> 생년월일은 `_BIRTH_RE`가 "1988년 3월 15일"까지만 포획하므로 뒤의 "생" 글자가 잔존한다. 본관(`본관: 경주`)은 `_HOMETOWN_RE`, 증명사진은 `_PHOTO_B64_RE` 우선 치환 후 `_PHOTO_LABEL_RE`가 라벨 라인을 통째로 재치환하여 base64 페이로드가 UI·캐시에 남지 않는다.

## 6. 한계 및 미해결 사항

- **구 HWP 바이너리(HWP 2007 이전)**: HWPX(ZIP) 구조가 아닌 바이너리 포맷은 `zipfile` 경로로 열리지 않으며, 로컬에 LibreOffice(`libreoffice`/`soffice`)가 설치되어 있어야 변환 가능. Streamlit Cloud 런타임에서는 이 경로가 작동하지 않으므로 업로드 시 "로컬 재처리 필요" 안내를 노출.
- **영문 이름 마스킹**: 한글 이름 대비 정확도가 낮다. 라벨 없이 단독 헤더로 등장하는 영문 이름은 `_NAME_HEADER_RE`에 매칭되지 않는다. 라벨형(`Name: ...`)이 아니면 누락될 수 있다.
- **스캔 PDF**: 원본 해상도에 의존. 흐린 스캔은 Vision OCR 품질이 떨어져 추출 실패(`vision_empty_output`)로 기록될 수 있다.
- **정규식 오탐/누락 가능성**: 주소 패턴은 시·도 + 구/군/시 조합만 커버한다. "도" 단독이거나 외국 주소는 누락될 수 있으며, 주소 뒤의 상세 도로명(예: "테헤란로 123")은 마스킹 대상이 아니므로 문맥상 노출 위험이 일부 남는다.
- **격리 필드의 로컬 캐시 노출**: `pii_fields_separated`는 LLM·UI로 전달되지 않지만 캐시 JSON에는 기록된다. 캐시 파일은 Git에 커밋되므로 실제 이력서 데이터로는 마스킹된 본문만 재구성 가능한 항목인지 반드시 사전 검수 후 커밋해야 한다.
