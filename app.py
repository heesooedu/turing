import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class Message:
    sender_id: str
    text: str
    ts: float


@dataclass
class Judgment:
    guess: str
    reason: str
    ts: float


@dataclass
class Participant:
    participant_id: str
    name: str
    joined_at: float
    preference: str  # auto | student | ai


@dataclass
class Room:
    room_id: str
    match_type: str  # student | ai
    participant_ids: List[str]
    created_at: float
    messages: List[Message] = field(default_factory=list)
    judgments: Dict[str, Judgment] = field(default_factory=dict)


class GameState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.round_open = True
        self.participants: Dict[str, Participant] = {}
        self.rooms: Dict[str, Room] = {}
        self.room_by_participant: Dict[str, str] = {}
        self.waiting_student_ids: List[str] = []

    def reset(self) -> None:
        with self.lock:
            self.round_open = True
            self.participants = {}
            self.rooms = {}
            self.room_by_participant = {}
            self.waiting_student_ids = []


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
state = GameState()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
AUTO_AI_RATIO = float(os.environ.get("AUTO_AI_RATIO", "0.35"))
AI_MIN_DELAY = float(os.environ.get("AI_MIN_DELAY", "1.2"))
AI_MAX_DELAY = float(os.environ.get("AI_MAX_DELAY", "4.2"))
openai_client = OpenAI() if OpenAI and os.environ.get("OPENAI_API_KEY") else None

SYSTEM_PROMPT = (
    "너는 중고등학생 대상 튜링 테스트 게임의 익명 대화 상대다. "
    "답변은 보통 1~3문장으로 자연스럽게, 가끔 말투 변화를 주고, 공격적/부적절 요청은 거절하라."
)


def _generate_id() -> str:
    return str(uuid.uuid4())[:8]


def _decide_auto_match_type() -> str:
    return "ai" if random.random() < AUTO_AI_RATIO else "student"


def _ai_delay_seconds(user_text: str) -> float:
    base = random.uniform(AI_MIN_DELAY, AI_MAX_DELAY)
    length_adjustment = min(len(user_text) / 80.0, 1.8)
    jitter = random.uniform(-0.3, 0.5)
    return max(0.8, base + length_adjustment + jitter)


def _ai_reply(user_text: str, room: Room, participant_id: str) -> str:
    if not openai_client:
        samples = [
            "음, 그 질문 재밌다. 네 생각은 어때?",
            "그럴 수도 있지. 왜 그렇게 생각했는지 더 듣고 싶어.",
            "상황에 따라 다를 것 같아. 예시를 하나 들어줄래?",
            "완전히 단정하긴 어렵지만, 나는 가능하다고 봐.",
        ]
        return random.choice(samples)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in room.messages[-12:]:
        role = "assistant" if m.sender_id == "ai" else "user"
        speaker = state.participants.get(m.sender_id)
        text = m.text
        if speaker and speaker.participant_id != participant_id:
            text = f"(상대 학생 메시지) {m.text}"
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_text})

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=180,
    )
    return (response.choices[0].message.content or "").strip()


def _make_student_room(pid1: str, pid2: str) -> Room:
    room = Room(
        room_id=_generate_id(),
        match_type="student",
        participant_ids=[pid1, pid2],
        created_at=time.time(),
    )
    state.rooms[room.room_id] = room
    state.room_by_participant[pid1] = room.room_id
    state.room_by_participant[pid2] = room.room_id
    return room


def _make_ai_room(pid: str) -> Room:
    room = Room(
        room_id=_generate_id(),
        match_type="ai",
        participant_ids=[pid],
        created_at=time.time(),
    )
    state.rooms[room.room_id] = room
    state.room_by_participant[pid] = room.room_id
    return room


def _try_match_waiting_students() -> None:
    while len(state.waiting_student_ids) >= 2:
        pid1 = state.waiting_student_ids.pop(0)
        pid2 = state.waiting_student_ids.pop(0)
        if pid1 in state.room_by_participant or pid2 in state.room_by_participant:
            continue
        _make_student_room(pid1, pid2)


@app.get("/")
def home():
    return render_template("index.html", round_open=state.round_open)


@app.post("/join")
def join():
    name = request.form.get("name", "").strip()
    preference = request.form.get("preference", "auto").strip()
    if not name or preference not in {"auto", "student", "ai"}:
        return redirect(url_for("home"))

    participant_id = _generate_id()
    participant = Participant(
        participant_id=participant_id,
        name=name,
        joined_at=time.time(),
        preference=preference,
    )

    with state.lock:
        if not state.round_open:
            return redirect(url_for("home"))

        state.participants[participant_id] = participant
        match_type = _decide_auto_match_type() if preference == "auto" else preference

        if match_type == "ai":
            _make_ai_room(participant_id)
        else:
            state.waiting_student_ids.append(participant_id)
            _try_match_waiting_students()

    session["participant_id"] = participant_id
    return redirect(url_for("student_chat"))


@app.get("/student")
def student_chat():
    participant = _get_participant()
    if not participant:
        return redirect(url_for("home"))
    return render_template("student.html", participant=participant, round_open=state.round_open)


@app.get("/api/messages")
def get_messages():
    participant = _get_participant()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    with state.lock:
        room = _get_room_by_participant(participant.participant_id)
        if not room:
            waiting_index = _waiting_order(participant.participant_id)
            return jsonify(
                {
                    "matched": False,
                    "waiting_order": waiting_index,
                    "round_open": state.round_open,
                }
            )

        peer_name = _peer_name(room, participant.participant_id)
        messages = []
        for m in room.messages:
            if m.sender_id == "ai":
                role = "opponent"
            elif m.sender_id == participant.participant_id:
                role = "self"
            else:
                role = "opponent"
            messages.append({"role": role, "text": m.text, "ts": m.ts})

        my_judgment = room.judgments.get(participant.participant_id)

    return jsonify(
        {
            "matched": True,
            "room_id": room.room_id,
            "match_type": room.match_type,
            "peer_name": peer_name,
            "messages": messages,
            "my_judgment": {
                "guess": my_judgment.guess,
                "reason": my_judgment.reason,
            }
            if my_judgment
            else None,
            "round_open": state.round_open,
        }
    )


@app.post("/api/send")
def send_message():
    participant = _get_participant()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400

    with state.lock:
        if not state.round_open:
            return jsonify({"error": "round closed"}), 400
        room = _get_room_by_participant(participant.participant_id)
        if not room:
            return jsonify({"error": "not matched yet"}), 400
        room.messages.append(Message(sender_id=participant.participant_id, text=text, ts=time.time()))
        should_ai_reply = room.match_type == "ai"

    if should_ai_reply:
        time.sleep(_ai_delay_seconds(text))
        with state.lock:
            room = _get_room_by_participant(participant.participant_id)
            if not room:
                return jsonify({"ok": True})
            reply = _ai_reply(text, room, participant.participant_id)
            room.messages.append(Message(sender_id="ai", text=reply, ts=time.time()))

    return jsonify({"ok": True})


@app.post("/api/judgment")
def submit_judgment():
    participant = _get_participant()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    guess = (request.json or {}).get("guess", "").strip()
    reason = (request.json or {}).get("reason", "").strip()
    if guess not in {"student", "ai"}:
        return jsonify({"error": "invalid guess"}), 400

    with state.lock:
        room = _get_room_by_participant(participant.participant_id)
        if not room:
            return jsonify({"error": "not matched yet"}), 400
        room.judgments[participant.participant_id] = Judgment(guess=guess, reason=reason, ts=time.time())

    return jsonify({"ok": True})


@app.get("/teacher")
def teacher_dashboard():
    with state.lock:
        waiting_students = [state.participants[pid] for pid in state.waiting_student_ids if pid in state.participants]
        rooms = list(state.rooms.values())
        total_judgments = sum(len(r.judgments) for r in rooms)
        correct = 0
        for r in rooms:
            answer = "student" if r.match_type == "student" else "ai"
            correct += sum(1 for j in r.judgments.values() if j.guess == answer)

        stats = {
            "participants": len(state.participants),
            "waiting": len(waiting_students),
            "rooms": len(rooms),
            "student_rooms": sum(1 for r in rooms if r.match_type == "student"),
            "ai_rooms": sum(1 for r in rooms if r.match_type == "ai"),
            "judgments": total_judgments,
            "correct": correct,
        }

        room_rows = []
        for room in rooms:
            participant_names = [state.participants[pid].name for pid in room.participant_ids if pid in state.participants]
            room_rows.append((room, participant_names))

    return render_template(
        "teacher.html",
        waiting_students=waiting_students,
        room_rows=room_rows,
        stats=stats,
        round_open=state.round_open,
    )


@app.post("/teacher/round/open")
def open_round():
    with state.lock:
        state.round_open = True
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/round/close")
def close_round():
    with state.lock:
        state.round_open = False
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/round/reset")
def reset_round():
    state.reset()
    return redirect(url_for("teacher_dashboard"))


@app.get("/teacher/api/rooms")
def teacher_rooms_api():
    with state.lock:
        rooms = []
        for r in state.rooms.values():
            rooms.append(
                {
                    "room_id": r.room_id,
                    "match_type": r.match_type,
                    "participants": [state.participants[pid].name for pid in r.participant_ids if pid in state.participants],
                    "messages": [{"sender_id": m.sender_id, "text": m.text, "ts": m.ts} for m in r.messages],
                    "judgments": {
                        state.participants[pid].name: {"guess": j.guess, "reason": j.reason}
                        for pid, j in r.judgments.items()
                        if pid in state.participants
                    },
                }
            )
    return jsonify({"rooms": rooms, "round_open": state.round_open})


def _get_participant() -> Optional[Participant]:
    participant_id = session.get("participant_id")
    if not participant_id:
        return None
    with state.lock:
        return state.participants.get(participant_id)


def _get_room_by_participant(participant_id: str) -> Optional[Room]:
    room_id = state.room_by_participant.get(participant_id)
    if not room_id:
        return None
    return state.rooms.get(room_id)


def _peer_name(room: Room, participant_id: str) -> str:
    if room.match_type == "ai":
        return "익명 상대"
    for pid in room.participant_ids:
        if pid != participant_id and pid in state.participants:
            return state.participants[pid].name
    return "익명 상대"


def _waiting_order(participant_id: str) -> int:
    try:
        return state.waiting_student_ids.index(participant_id) + 1
    except ValueError:
        return 0


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
