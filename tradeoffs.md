# Tradeoffs (HR Resume Copilot)

3시간·$3 예산 내에서 과제 핵심("분류 + JD 진단 + PII 안전")을 유지하기 위해 의도적으로 포기한 기능과 그 대체 방식을 기록한다.

## Scope Cuts

### 1. 같은 직군 JD 상대 비교 (Stage 4 원안)
- **왜 포기**: `task02_data/jds.csv` 확인 결과 JD001~JD012가 모두 서로 다른 직군(백엔드/프론트/데이터/ML/DevOps/게임 클라이언트/게임 서버/QA/PM/UX/사업개발/보안)이라 "같은 직군 페어"가 성립하지 않음. 상대 비교의 전제가 무너진다.
- **대체**: Stage 3 삼중 레이어(절대 임계 / 느슨한 클러스터 상대 / 수요-공급 갭)로 설계 변경 + "헬시 JD 레퍼런스 주입"(`pick_healthy_refs`). 같은 클러스터(engineering / non-engineering)의 플래그 0개 + 합격률 상위 1~2개 JD description을 LLM 프롬프트에 포함시켜 비교 기반 개선안 가치를 복원.

### 2. Notion API 자동 sync
- **왜 포기**: Notion integration 토큰 발급·페이지 ID 매핑·API 블록 구조 맞추기가 3시간 예산에 비해 불확실. 실패 시 데모 리스크.
- **대체**: `src/export.py:build_notion_markdown()`이 토글/콜아웃/테이블 Notion 페이스트 호환 Markdown을 생성. UI "다운로드" 탭에서 HR이 1회 복붙하는 흐름.

### 3. 이력서 원본 뷰어
- **왜 포기**: 원본 PDF/DOCX를 UI로 노출하면 마스킹 파이프라인을 우회하게 된다. PII 누설 위험이 구현 이득보다 크다.
- **대체**: `safe_display_row(row, allowed_fields)` 화이트리스트를 통해 `resume_id`, 경력 요약, 기술 스택, confidence 등 마스킹된 필드만 표시.

### 4. 지원자 랭킹/점수화
- **왜 포기**: 과제 요구사항(분류 + 진단)에 포함되지 않음. 자동화된 점수화는 2026-04-15 법무팀 공지와 충돌 가능성(자동화된 차별) — 범위를 명시적으로 벗어남.
- **대체**: 없음. JD별 분류 결과 + Stage 2 confidence만 제공하고 최종 판단은 HR에 위임.

### 5. LibreOffice 의존 HWP 경로
- **왜 포기**: Streamlit Cloud에 LibreOffice(~500MB) 설치 시 빌드 타임아웃 위험. 로컬에서도 Windows 환경에 미설치 가능성.
- **대체**: `src/extract.py:_extract_hwp_text()`에서 파일 시그니처(`PK\x03\x04`)로 HWPX(ZIP) 판별 후 `Preview/PrvText.txt` → `Contents/section*.xml`의 `<hs:t>` 노드를 직접 파싱. LibreOffice는 구 바이너리 HWP 전용 fallback으로만 유지. 본 과제 데이터 확인 결과 HWP 파일은 전부 HWPX 포맷이었음.

## Quality Cuts

- **탭 2 필터 축소**: "JD + 경력구간 + 기술 다중 체크박스" 구성을 **JD 드롭다운 + 단순 텍스트 검색**으로 축소. 3시간 예산 내에서 UI 복잡도를 줄이되 핵심 질의("이 JD로 분류된 이력서 보기")는 보장.
- **LLM retry 정책**: task01에서 사용한 3회 재시도 정책을 `src/llm.py`에 그대로 재사용. 과제별로 튜닝하지 않음(동일 모델·동일 호출 패턴이라 재튜닝 이득 작음).
- **Stage 3 LLM 비용 제어**: 플래그 없는 JD(`is_healthy=True`)는 LLM 호출 자체를 생략. 12개 중 플래그가 붙은 JD만 `diagnose_jd()` 호출.

## Known Limits

- **스캔 PDF 품질**: Vision API 결과는 원본 해상도에 의존. 저해상도 스캔은 `vision_empty_output`으로 실패 가능하며, 성공률 90% 목표에서 일부 차감될 수 있음.
- **영문 이름 정규식 불완전**: `_NAME_HEADER_RE`는 `[가-힣]{2,4}`에 맞춰져 있어 영문 단독 이름 헤더는 마스킹 누락 가능. 라벨 없는 영문 이름은 누락 리스크 잔존.
- **funnel_stats.csv 검증 없음**: 입력 CSV의 월별 합·비율을 그대로 신뢰. 데이터 오염이 있을 경우 Layer A/B 플래그가 잘못 발화할 수 있음.
- **HWP 2007 이전 바이너리 포맷 미지원**: 본 과제 시료는 전부 HWPX였으나, 실환경에서 구 바이너리가 섞이면 LibreOffice가 필요하고 Streamlit Cloud에서는 실패.
- **주소 부분 마스킹**: `[MASK_ADDR]` 치환 후에도 뒤의 상세 도로명·번지는 남을 수 있음(`서울시 강남구 테헤란로 123` → `[MASK_ADDR] 테헤란로 123`).
- **Streamlit Cloud 런타임 한계**: 업로드된 HWP/스캔 PDF 중 처리 불가 건은 "로컬 재처리 필요" 메시지 후 스킵. 증분 업데이트는 로컬 개발자의 수동 재실행 + commit에 의존.
