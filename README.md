# 학생용 튜링 테스트 실습 웹앱

교사 PC를 로컬 서버로 실행하고, 학생은 각자 브라우저로 접속해
"상대가 학생인지 AI인지"를 맞히는 튜링 테스트 게임을 진행할 수 있는 Flask 앱입니다.

## 주요 기능

- **두 가지 매칭 방식 지원**
  - 학생-학생 매칭
  - 학생-AI 매칭
- 입장 시 `자동/학생-학생/학생-AI` 선택 가능
- 학생-학생 선택 시 대기열로 관리되어 2명씩 자동 페어링
- 교사용 대시보드에서 대기열/방/판정 통계 확인
- AI 답변은 **랜덤 지연(기본 1.2~4.2초 + 문장 길이 보정)** 후 전송되어
  즉답 패턴을 줄임

## 빠른 실행

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력 (앱이 자동 로드)
python app.py
```

- 학생 접속: `http://교사IP:8000`
- 교사 접속: `http://교사IP:8000/teacher`

## 환경변수

- `OPENAI_API_KEY`: ChatGPT API 키(없으면 데모 모드)
- `OPENAI_MODEL`: 기본 `gpt-4o-mini`
- `AUTO_AI_RATIO`: 자동 배정 시 학생-AI 비율(기본 `0.35`)
- `AI_MIN_DELAY`: AI 최소 지연(초)
- `AI_MAX_DELAY`: AI 최대 지연(초)
- `FLASK_SECRET_KEY`: 세션 보안 키


## 주의

- `.env.example`은 예시 파일입니다. 실제 실행 시에는 `.env` 파일로 복사해 사용해야 합니다.
