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
except Exception:  # openai is optional during local dry runs
    OpenAI = None


@dataclass
class Message:
    role: str
    text: str
    ts: float


@dataclass
class Room:
    room_id: str
    mode: str  # "ai" or "human"
    student_name: str
    created_at: float
    messages: List[Message] = field(default_factory=list)
    judgment: Optional[str] = None
    judgment_reason: str = ""


class GameState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.round_open = True
        self.rooms: Dict[str, Room] = {}

    def reset(self) -> None:
        with self.lock:
            self.rooms = {}
            self.round_open = True


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
state = GameState()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HUMAN_ROOM_RATIO = float(os.environ.get("HUMAN_ROOM_RATIO", "0.3"))
openai_client = OpenAI() if OpenAI and os.environ.get("OPENAI_API_KEY") else None

SYSTEM_PROMPT = (
    "너는 중고등학생 대상 튜링 테스트 게임의 대화 상대다. "
    "짧고 친절하게 답하고, 유해하거나 개인정보를 요구하는 대화는 피한다."
)


def _pick_room_mode() -> str:
    return "human" if random.random() < HUMAN_ROOM_RATIO else "ai"


def _ai_reply(user_text: str, history: List[Message]) -> str:
    if not openai_client:
        return "(데모 모드) 흥미로운 질문이네요. 조금 더 구체적으로 말해줄래요?"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-10:]:
        role = "assistant" if m.role == "opponent" else "user"
        messages.append({"role": role, "content": m.text})
    messages.append({"role": "user", "content": user_text})

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=180,
    )
    return response.choices[0].message.content or ""


@app.get("/")
def home():
    return render_template("index.html", round_open=state.round_open)


@app.post("/join")
def join():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("home"))

    with state.lock:
        if not state.round_open:
            return redirect(url_for("home"))

        room_id = str(uuid.uuid4())[:8]
        room = Room(
            room_id=room_id,
            mode=_pick_room_mode(),
            student_name=name,
            created_at=time.time(),
        )
        state.rooms[room_id] = room

    session["room_id"] = room_id
    session["student_name"] = name
    return redirect(url_for("student_chat"))


@app.get("/student")
def student_chat():
    room = _get_student_room()
    if not room:
        return redirect(url_for("home"))
    return render_template("student.html", room=room, round_open=state.round_open)


@app.get("/api/messages")
def get_messages():
    room = _get_student_room()
    if not room:
        return jsonify({"error": "room not found"}), 404

    with state.lock:
        data = [
            {"role": m.role, "text": m.text, "ts": m.ts}
            for m in state.rooms[room.room_id].messages
        ]
    return jsonify({"messages": data, "mode": room.mode, "round_open": state.round_open})


@app.post("/api/send")
def send_message():
    room = _get_student_room()
    if not room:
        return jsonify({"error": "room not found"}), 404

    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400

    if not state.round_open:
        return jsonify({"error": "round closed"}), 400

    with state.lock:
        live_room = state.rooms.get(room.room_id)
        if not live_room:
            return jsonify({"error": "room not found"}), 404
        live_room.messages.append(Message(role="student", text=text, ts=time.time()))
        mode = live_room.mode
        history_copy = list(live_room.messages)

    if mode == "ai":
        reply = _ai_reply(text, history_copy)
        with state.lock:
            live_room = state.rooms.get(room.room_id)
            if live_room:
                live_room.messages.append(Message(role="opponent", text=reply, ts=time.time()))

    return jsonify({"ok": True})


@app.post("/api/judgment")
def submit_judgment():
    room = _get_student_room()
    if not room:
        return jsonify({"error": "room not found"}), 404

    guess = request.json.get("guess", "").strip()
    reason = request.json.get("reason", "").strip()
    if guess not in {"human", "ai"}:
        return jsonify({"error": "invalid guess"}), 400

    with state.lock:
        live_room = state.rooms.get(room.room_id)
        if not live_room:
            return jsonify({"error": "room not found"}), 404
        live_room.judgment = guess
        live_room.judgment_reason = reason

    return jsonify({"ok": True})


@app.get("/teacher")
def teacher_dashboard():
    with state.lock:
        rooms = list(state.rooms.values())
        stats = {
            "total": len(rooms),
            "human_rooms": sum(1 for r in rooms if r.mode == "human"),
            "ai_rooms": sum(1 for r in rooms if r.mode == "ai"),
            "judged": sum(1 for r in rooms if r.judgment),
            "correct": sum(1 for r in rooms if r.judgment and r.judgment == r.mode),
        }
    return render_template("teacher.html", rooms=rooms, stats=stats, round_open=state.round_open)


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


@app.post("/teacher/room/<room_id>/reply")
def human_reply(room_id: str):
    text = request.form.get("text", "").strip()
    if not text:
        return redirect(url_for("teacher_dashboard"))

    with state.lock:
        room = state.rooms.get(room_id)
        if room and room.mode == "human":
            room.messages.append(Message(role="opponent", text=text, ts=time.time()))
    return redirect(url_for("teacher_dashboard"))


@app.get("/teacher/api/rooms")
def teacher_rooms_api():
    with state.lock:
        rooms = [
            {
                "room_id": r.room_id,
                "student_name": r.student_name,
                "mode": r.mode,
                "messages": [{"role": m.role, "text": m.text, "ts": m.ts} for m in r.messages],
                "judgment": r.judgment,
                "judgment_reason": r.judgment_reason,
            }
            for r in state.rooms.values()
        ]
    return jsonify({"rooms": rooms, "round_open": state.round_open})


def _get_student_room() -> Optional[Room]:
    room_id = session.get("room_id")
    if not room_id:
        return None
    with state.lock:
        return state.rooms.get(room_id)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
