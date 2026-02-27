# 학생용 튜링 테스트 실습 웹앱

교사 PC를 로컬 서버로 실행하고, 학생은 각자 브라우저로 접속해
"상대가 사람인지 AI인지"를 맞히는 튜링 테스트 게임을 진행할 수 있는 Flask 앱입니다.

## 주요 기능

- 학생 화면: 대화 + 판정(사람/AI, 근거 제출)
- 교사용 대시보드: 라운드 열기/닫기/초기화, 실시간 대화 확인
- 혼합 모드: 일부 방은 사람 역할(교사), 일부는 AI 자동 응답
- OpenAI API 키가 없으면 데모 응답 모드로 동작

## 빠른 실행

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # 선택
python app.py
```

- 학생 접속: `http://교사IP:8000`
- 교사 접속: `http://교사IP:8000/teacher`

## 환경변수

- `OPENAI_API_KEY`: ChatGPT API 키(없으면 데모 모드)
- `OPENAI_MODEL`: 기본 `gpt-4o-mini`
- `HUMAN_ROOM_RATIO`: 사람 역할 방 비율(기본 `0.3`)
- `FLASK_SECRET_KEY`: 세션 보안 키
