# KRX 고래 추적기

한국 주식 실시간 고래(대량 체결) 추적 및 AI 세력 분석 웹앱.

## Run & Operate

- `KRX 고래 추적기` 워크플로우로 실행 (포트 8000)
- `artifacts/korean-stock-tracker/main.py` — FastAPI 메인 앱
- 환경변수: `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO`, `KIS_ACCOUNT_SUFFIX`, `DART_API_KEY`
- AI: `AI_INTEGRATIONS_ANTHROPIC_BASE_URL`, `AI_INTEGRATIONS_ANTHROPIC_API_KEY` (Replit 자동 설정)

## Stack

- **백엔드**: Python 3.11, FastAPI, uvicorn, asyncio
- **실시간**: KIS WebSocket (pipe-delimited 프로토콜), SSE (sse-starlette)
- **데이터**: pykrx, pandas, numpy
- **AI**: Anthropic Claude Sonnet (Replit AI 통합)
- **프론트**: Vanilla JS, TailwindCSS CDN, Chart.js CDN, FontAwesome CDN

## Where things live

```
artifacts/korean-stock-tracker/
├── main.py               — FastAPI 앱, 모든 API 엔드포인트
├── startup_event.py      — 앱 시작 시 초기화 (ticker 캐시, DART)
├── kis_auth.py           — KIS OAuth 토큰 관리, 자동 갱신
├── kis_realtime.py       — KIS WebSocket 고래 감지, SSE 큐
├── screening.py          — pykrx 기반 종목 스캐너
├── scoring.py            — 세력 점수 계산 (A~H 8개 항목)
├── dart_client.py        — OpenDART 공시 조회
├── claude_analyst.py     — Claude AI 세력 분석
├── strategy.py           — 자동 매매 전략 생성
├── market_filter.py      — KOSPI/KOSDAQ 시장 상태
├── static/index.html     — SPA 프론트엔드
└── cache/corp_codes.json — DART corp code 캐시
```

## Architecture decisions

- pykrx는 동기 라이브러리 → `asyncio.to_thread()` 로 항상 감싸서 이벤트 루프 블로킹 방지
- KIS WebSocket 메시지는 JSON이 아닌 pipe-delimited 문자열 → `_parse_pipe_message()` 로 파싱
- 고래 SSE는 `asyncio.Queue` 기반으로 실시간 푸시
- startup_event는 서버 포트 오픈을 블로킹하지 않도록 백그라운드 태스크로 실행
- Claude 호출은 점수 5 이상 또는 고래 신호 있을 때만 실행 (API 비용 절감)
- DART corpCode.xml은 ZIP 압축 응답 → zipfile + XML 파싱 후 cache/corp_codes.json 캐싱

## Product

- **탭 1 고래 피드**: KIS 실시간 WebSocket에서 1억+ 체결 감지, SSE로 브라우저에 푸시
- **탭 2 종목 스캐너**: pykrx로 시총/거래량 필터링 후 8개 항목 세력 점수 계산
- **탭 3 AI 분석**: 종목별 세력 단계 판단 + 자동 전략 생성 + AI 채팅
- **탭 4 설정**: KIS 환경 전환 (실전/모의), 고래 임계값 설정

## User preferences

- 모든 종목코드는 6자리 문자열 유지 (`str(ticker).zfill(6)`)
- AI: Replit 내장 Anthropic 연동 사용 (별도 키 없음)
- KIS_ENV=PROD (실전 환경)

## Gotchas

- pykrx는 KRX 서버에서 데이터를 가져오므로 장 마감 후에는 일부 API가 빈 응답 반환
- KIS WebSocket은 장 시간(09:00~15:30)에만 연결 시도
- `get_index_ohlcv_by_date`는 pykrx 내부 로깅 버그로 stderr에 오류 출력될 수 있으나 예외 처리됨
- DART corpCode.xml은 앱 시작 시 백그라운드로 다운로드 → 첫 실행 시 수십초 소요

## API Endpoints

- `GET /` — SPA 프론트엔드
- `GET /api/search?q=` — 종목 검색
- `GET /api/scan` — 시장 스캔 (최대 2분)
- `GET /api/analyze/{ticker}` — AI 종목 분석
- `GET /api/dart/{ticker}` — 공시 조회
- `GET /api/market-status` — 시장 상태
- `GET /api/whale/realtime` — 고래 SSE 스트림
- `GET /api/whale/summary` — 5분 누적 고래 요약
- `POST /api/chat` — AI 채팅
- `POST /api/settings` — 설정 업데이트

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
