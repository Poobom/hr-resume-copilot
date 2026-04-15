# Progress Log (HR Resume Copilot)

3시간 구현 타임라인 실시간 기록. 기준 시각은 세션 시작 00:00.

## 시간 로그

### 00:00 — 00:20 | 스캐폴딩 + 환경
- [00:00] `task02_solution/` 디렉토리 생성, `.env.example` 구성, `.gitignore`에 `task02_data/resumes/`·`.env`·`artifacts/**/raw_texts/` 차단 추가
- [00:05] `requirements.txt` 작성(streamlit, openai, pandas, numpy, scikit-learn, pdfplumber, python-docx, pdf2image→PyMuPDF, Pillow, plotly, python-dotenv)
- [00:10] task01_solution에서 `src/llm.py`(CostTracker + chat_json + Vision 래퍼) · `src/cache.py`(SHA1 run_dir + 파일별 resume_cache) 복사 후 재사용 경로로 수정
- [00:15] `src/` 패키지 초기화(`__init__.py`), `.streamlit/config.toml` 추가

### 00:20 — 00:45 | PII + 추출
- [00:20] `src/pii.py` 작성(sub-agent): 이름·전화·이메일·주소 마스킹 + 학력·나이·성별·사진·출신지역 격리 + `assert_pii_safe` + `safe_display_row`
- [00:28] `src/pii.py` `__main__` 샘플 실행으로 3종 케이스 스팟 체크
- [00:30] `src/extract.py` 하이브리드 라우팅 작성(sub-agent): PDF pdfplumber→Vision 폴백, DOCX python-docx, HWP HWPX-ZIP
- [00:38] 데이터 디렉토리 스캔 중 HWP 파일 시그니처가 모두 `PK\x03\x04`(ZIP)인 것 확인 → LibreOffice 경로를 fallback으로 강등
- [00:42] `extract.extract_resume()` 단위 실행(3건)으로 masked_text 길이·`ok=true` 확인

### 00:45 — 01:10 | 분류
- [00:45] `src/classify.py` 임베딩 + LLM 재판정 작성(sub-agent): text-embedding-3-small로 JD·resume 임베딩, 코사인 top-2, margin<0.05 시 gpt-4o-mini tiebreak, `top2`·`margin`·`decision_path` 기록
- [01:00] 샘플 5건 분류 실행, confidence 분포·tiebreak 진입률 확인
- [01:05] `classify.load_jds_csv()` + 파이프라인 hook 결선 완료

### 01:10 — 02:15 | Stage 3 / 4 / 5
- [01:10] 3개 sub-agent 병렬 실행
  - `src/diagnose.py`: 삼중 레이어 플래그(절대/클러스터 상대/수요-공급 갭) + `pick_healthy_refs` + `diagnose_jd` LLM 호출
  - `src/compare.py`: Stage 3 결과 재사용해 `gap_table`·drift 분포 집계(추가 LLM 없음)
  - `src/export.py`: Notion 친화 Markdown 빌더 + `_pii_guard` 최종 스캔
- [01:45] Layer C에서 Stage 2 `top2`·`margin` 의존을 확인, classify 출력 스키마 동기화
- [01:55] `pipeline.py` 오케스트레이션 작성(5단계 + stage_done 캐시 + `run_dir = get_run_dir(signature)`)
- [02:00] 800건 pre-bake 실행 시작 (`python pipeline.py --input-dir ../task02_data`), 진행 로그 50건 단위 출력
- [02:10] 진단 + 비교 + Markdown export 완료, `cost.json` 저장 확인

### 02:15 — 02:45 | UI
- [02:15] `app.py` Streamlit 6탭 작성(sub-agent): 대시보드 / 이력서 분류 / JD 병목 분석 / 수요-공급 갭 / PII 정책 / 다운로드
- [02:30] 탭 2에 confidence 해석 expander 추가, 탭 3에 삼중 레이어 플래그 expander 추가, 탭 4에 "잠재 후보" 산출 방식 expander 추가
- [02:38] `safe_display_row` 화이트리스트로 각 테이블 컬럼 잠금
- [02:40] 로컬 `streamlit run app.py`로 전 탭 시각 확인, KPI 카드·funnel·drift 차트 렌더링 검증

### 02:45 — 03:00 | 배포 + 데모
- [02:47] GitHub 리포 푸시(의미있는 커밋 단위로 정리)
- [02:50] Streamlit Cloud 연결, `secrets.toml`에 `OPENAI_API_KEY` 주입, 로그인 없는 공개 URL 확인
- [02:55] 2분 데모 영상 녹화(업로드 → 분류 → JD 병목 → 갭 탭 → 다운로드)
- [02:58] `pii_policy.md` / `tradeoffs.md` / `progress.md` 최종 다듬기

## Decisions Made

- **[00:40] HWP 처리 경로**: HWP 파일이 전부 HWPX(ZIP) 포맷으로 판명 → LibreOffice 의존성 제거. `zipfile` + `Preview/PrvText.txt` 우선, 없으면 `Contents/section*.xml`의 `<hs:t>` 노드 직접 파싱. LibreOffice는 구 바이너리용 fallback만 유지. Streamlit Cloud 빌드 안정성 확보.
- **[00:50] Vision OCR 렌더러**: `pdf2image`(poppler 의존)가 Streamlit Cloud 빌드에서 위험 → PyMuPDF(`fitz`)로 PDF 페이지를 PNG로 렌더하는 방식으로 교체. poppler 의존성 제거.
- **[01:20] Layer B 클러스터 매핑**: 12개 JD가 모두 다른 직군이라 "같은 직군 평균" 불가능. `CLUSTER_MAP` 상수를 `src/diagnose.py`에 하드코딩(JD001~008·012 = engineering / JD009·010·011 = non-engineering)하여 느슨한 비교만 유지. 원래 Stage 4 "상대 비교"는 Stage 3 레퍼런스 주입으로 대체.
- **[02:10] Notion 연동 축소**: Notion API 자동 sync 포기, Markdown export 버튼으로 축소. 토글·콜아웃·테이블 구조만 맞추면 HR이 1회 복붙으로 동일 결과 달성.

## Blockers Encountered

- **[00:30] HWP 추출 리스크**: 로컬 Windows 환경에 LibreOffice 미설치 → 전 건 실패 우려. → **해결**: HWPX는 사실상 ZIP이라는 점을 확인하고 `zipfile`로 `Preview/PrvText.txt` 직접 추출.
- **[00:45] poppler 의존성**: `pdf2image`가 Streamlit Cloud에서 poppler 설치 필요. → **해결**: PyMuPDF(`fitz`)로 대체하여 OS 의존성 0.
- **[01:50] Stage 2 ↔ Stage 3 스키마 결합**: Layer C가 `classified[i].top2`·`margin`에 의존하는데 초기 classify 출력엔 누락. → **해결**: classify 결과 dict에 `top2: [(jd_id, sim), ...]`와 `margin` 필드 추가, diagnose 측 접근자를 이에 맞춤.

## Risks Flagged

- **Streamlit Cloud에서 구 바이너리 HWP 업로드**: HWPX가 아닐 경우 추출 실패. → UI에 "이 형식은 로컬 재처리 필요" 안내 메시지 노출, 해당 파일은 `ok=false` + `error=hwp_unavailable`로 기록.
- **LLM 비용 상한**: $3 상한 초과 방지를 위해 `cost.total > $2.5`이면 UI에서 실행 버튼 비활성화. 800건 pre-bake는 로컬에서 1회 수행하고 Cloud는 캐시만 사용하여 재호출 0 유지.
- **PII 회귀**: 향후 정규식 수정 시 감사 샘플(본 문서 Section 5 및 `src/pii.py` `__main__`) 실행을 PR 체크리스트로 관리.
