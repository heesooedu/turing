import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:  # openai is optional during local dry runs
    OpenAI = None


@dataclass
class Message:
    sender_id: str
    text: str
    ts: float


@dataclass
class Participant:
    participant_id: str
    name: str
    joined_at: float
    room_id: Optional[str] = None


@dataclass
class Room:
    room_id: str
    mode: str  # "ai" or "human"
    participant_ids: List[str]
    created_at: float
    messages: List[Message] = field(default_factory=list)
    awaiting_ai: bool = False
    judgments: Dict[str, str] = field(default_factory=dict)
    judgment_reasons: Dict[str, str] = field(default_factory=dict)
    ai_persona: str = ""


class GameState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.accepting_joins = True
        self.matching_started = False
        self.default_ai_persona = (
            "You are a middle school student in a Turing-test style chat game. "
            "Sound natural and varied. Mix short and medium answers. "
            "Do not reveal you are an AI unless directly asked."
        )
        self.ai_personas = [self.default_ai_persona, "", "", ""]
        self.participants: Dict[str, Participant] = {}
        self.rooms: Dict[str, Room] = {}

    def reset(self) -> None:
        with self.lock:
            self.participants = {}
            self.rooms = {}
            self.accepting_joins = True
            self.matching_started = False


if load_dotenv:
    load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
state = GameState()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HUMAN_ROOM_RATIO = float(os.environ.get("HUMAN_ROOM_RATIO", "0.3"))
AI_REPLY_DELAY_MIN_SEC = float(os.environ.get("AI_REPLY_DELAY_MIN_SEC", "1.8"))
AI_REPLY_DELAY_MAX_SEC = float(os.environ.get("AI_REPLY_DELAY_MAX_SEC", "4.2"))
openai_client = OpenAI() if OpenAI and os.environ.get("OPENAI_API_KEY") else None

if AI_REPLY_DELAY_MAX_SEC < AI_REPLY_DELAY_MIN_SEC:
    AI_REPLY_DELAY_MIN_SEC, AI_REPLY_DELAY_MAX_SEC = (
        AI_REPLY_DELAY_MAX_SEC,
        AI_REPLY_DELAY_MIN_SEC,
    )


def _ai_reply(user_text: str, history: List[Message], my_id: str, persona: str) -> str:
    if not openai_client:
        return "(demo mode) Could you say that in a bit more detail?"

    messages = [{"role": "system", "content": persona}]
    for m in history[-10:]:
        role = "user" if m.sender_id == my_id else "assistant"
        messages.append({"role": role, "content": m.text})
    messages.append({"role": "user", "content": user_text})

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.9,
            max_tokens=180,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return "(ai unavailable) I cannot reply right now."


def _post_ai_reply_after_delay(
    room_id: str, user_text: str, history: List[Message], my_id: str, persona: str
) -> None:
    try:
        reply = _ai_reply(user_text, history, my_id=my_id, persona=persona)
    except Exception:
        reply = "(ai unavailable) I cannot reply right now."

    time.sleep(random.uniform(AI_REPLY_DELAY_MIN_SEC, AI_REPLY_DELAY_MAX_SEC))

    with state.lock:
        room = state.rooms.get(room_id)
        if not room:
            return
        room.messages.append(Message(sender_id="ai", text=reply, ts=time.time()))
        room.awaiting_ai = False


def _pick_ai_persona_unlocked() -> str:
    candidates = [p.strip() for p in state.ai_personas if p and p.strip()]
    if not candidates:
        return state.default_ai_persona
    return random.choice(candidates)


def _run_matching_unlocked() -> int:
    unmatched = [p.participant_id for p in state.participants.values() if not p.room_id]
    random.shuffle(unmatched)
    total = len(unmatched)
    if total == 0:
        state.matching_started = True
        state.accepting_joins = False
        return 0

    human_pair_count = int((total * HUMAN_ROOM_RATIO) // 2)
    created = 0

    for _ in range(human_pair_count):
        if len(unmatched) < 2:
            break
        a_id = unmatched.pop()
        b_id = unmatched.pop()
        room_id = str(uuid.uuid4())[:8]
        room = Room(
            room_id=room_id,
            mode="human",
            participant_ids=[a_id, b_id],
            created_at=time.time(),
        )
        state.rooms[room_id] = room
        state.participants[a_id].room_id = room_id
        state.participants[b_id].room_id = room_id
        created += 1

    while unmatched:
        a_id = unmatched.pop()
        room_id = str(uuid.uuid4())[:8]
        room = Room(
            room_id=room_id,
            mode="ai",
            participant_ids=[a_id],
            created_at=time.time(),
            ai_persona=_pick_ai_persona_unlocked(),
        )
        state.rooms[room_id] = room
        state.participants[a_id].room_id = room_id
        created += 1

    state.matching_started = True
    state.accepting_joins = False
    return created


def _get_participant_from_session() -> Optional[Participant]:
    participant_id = session.get("participant_id")
    if not participant_id:
        return None
    with state.lock:
        return state.participants.get(participant_id)


@app.get("/")
def home():
    with state.lock:
        accepting = state.accepting_joins
    return render_template("index.html", accepting_joins=accepting)


@app.post("/join")
def join():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("home"))

    with state.lock:
        if not state.accepting_joins:
            return redirect(url_for("home"))
        participant_id = str(uuid.uuid4())[:8]
        state.participants[participant_id] = Participant(
            participant_id=participant_id,
            name=name,
            joined_at=time.time(),
        )

    session["participant_id"] = participant_id
    return redirect(url_for("student_chat"))


@app.get("/student")
def student_chat():
    participant = _get_participant_from_session()
    if not participant:
        return redirect(url_for("home"))
    return render_template("student.html", student_name=participant.name)


@app.get("/api/messages")
def get_messages():
    participant = _get_participant_from_session()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    with state.lock:
        live_participant = state.participants.get(participant.participant_id)
        if not live_participant:
            return jsonify({"error": "participant not found"}), 404

        if not live_participant.room_id:
            return jsonify(
                {
                    "messages": [],
                    "waiting": True,
                    "matching_started": state.matching_started,
                    "mode": None,
                    "can_send": False,
                    "awaiting_reply": False,
                }
            )

        room = state.rooms.get(live_participant.room_id)
        if not room:
            return jsonify({"error": "room not found"}), 404

        my_id = live_participant.participant_id
        last_sender_id = room.messages[-1].sender_id if room.messages else None

        can_send = True
        awaiting_reply = False
        if room.mode == "ai":
            awaiting_reply = room.awaiting_ai
            can_send = can_send and (not room.awaiting_ai)
        can_send = can_send and (last_sender_id != my_id)

        data = [
            {
                "role": "student" if m.sender_id == my_id else "opponent",
                "text": m.text,
                "ts": m.ts,
            }
            for m in room.messages
        ]

        return jsonify(
            {
                "messages": data,
                "waiting": False,
                "mode": room.mode,
                "matching_started": state.matching_started,
                "can_send": can_send,
                "awaiting_reply": awaiting_reply,
            }
        )


@app.post("/api/send")
def send_message():
    participant = _get_participant_from_session()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400

    with state.lock:
        live_participant = state.participants.get(participant.participant_id)
        if not live_participant:
            return jsonify({"error": "participant not found"}), 404
        if not live_participant.room_id:
            return jsonify({"error": "waiting for teacher matching"}), 400

        live_room = state.rooms.get(live_participant.room_id)
        if not live_room:
            return jsonify({"error": "room not found"}), 404

        my_id = live_participant.participant_id
        last_sender_id = live_room.messages[-1].sender_id if live_room.messages else None
        if last_sender_id == my_id:
            return jsonify({"error": "wait for opponent reply before sending again"}), 400
        if live_room.mode == "ai" and live_room.awaiting_ai:
            return jsonify({"error": "ai is typing, please wait"}), 400

        room_id = live_room.room_id
        live_room.messages.append(Message(sender_id=my_id, text=text, ts=time.time()))
        mode = live_room.mode
        history_copy = list(live_room.messages)
        persona = live_room.ai_persona or state.default_ai_persona
        if mode == "ai":
            live_room.awaiting_ai = True

    if mode == "ai":
        t = threading.Thread(
            target=_post_ai_reply_after_delay,
            args=(room_id, text, history_copy, my_id, persona),
            daemon=True,
        )
        t.start()

    return jsonify({"ok": True})


@app.post("/api/judgment")
def submit_judgment():
    participant = _get_participant_from_session()
    if not participant:
        return jsonify({"error": "participant not found"}), 404

    payload = request.get_json(silent=True) or {}
    guess = payload.get("guess", "").strip()
    reason = payload.get("reason", "").strip()
    if guess not in {"human", "ai"}:
        return jsonify({"error": "invalid guess"}), 400

    with state.lock:
        live_participant = state.participants.get(participant.participant_id)
        if not live_participant:
            return jsonify({"error": "participant not found"}), 404
        if not live_participant.room_id:
            return jsonify({"error": "waiting for teacher matching"}), 400

        room = state.rooms.get(live_participant.room_id)
        if not room:
            return jsonify({"error": "room not found"}), 404

        room.judgments[live_participant.participant_id] = guess
        room.judgment_reasons[live_participant.participant_id] = reason

    return jsonify({"ok": True})


@app.get("/teacher")
def teacher_dashboard():
    with state.lock:
        participants = list(state.participants.values())
        rooms = list(state.rooms.values())

        waiting_students = [p.name for p in participants if not p.room_id]
        judged = sum(len(r.judgments) for r in rooms)
        correct = 0
        for r in rooms:
            for guess in r.judgments.values():
                if guess == r.mode:
                    correct += 1

        room_rows = []
        for r in rooms:
            names = [state.participants[pid].name for pid in r.participant_ids if pid in state.participants]

            judgments = []
            for pid in r.participant_ids:
                participant = state.participants.get(pid)
                if not participant:
                    continue
                judgments.append(
                    {
                        "student_name": participant.name,
                        "guess": r.judgments.get(pid, "-"),
                        "reason": r.judgment_reasons.get(pid, "-"),
                    }
                )

            messages = []
            for m in r.messages:
                if m.sender_id == "ai":
                    sender_name = "AI"
                else:
                    sender = state.participants.get(m.sender_id)
                    sender_name = sender.name if sender else "Unknown"
                messages.append({"sender_name": sender_name, "text": m.text})

            room_rows.append(
                {
                    "room_id": r.room_id,
                    "mode": r.mode,
                    "student_names": ", ".join(names),
                    "ai_persona": r.ai_persona,
                    "messages": messages,
                    "judgments": judgments,
                }
            )

        stats = {
            "participants": len(participants),
            "waiting": len(waiting_students),
            "rooms": len(rooms),
            "human_rooms": sum(1 for r in rooms if r.mode == "human"),
            "ai_rooms": sum(1 for r in rooms if r.mode == "ai"),
            "judged": judged,
            "correct": correct,
        }

        context = {
            "rooms": room_rows,
            "stats": stats,
            "accepting_joins": state.accepting_joins,
            "matching_started": state.matching_started,
            "waiting_students": waiting_students,
            "ai_personas": state.ai_personas,
        }

    return render_template("teacher.html", **context)


@app.post("/teacher/round/open")
def open_round():
    with state.lock:
        state.accepting_joins = True
        state.matching_started = False
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/round/close")
def close_round():
    with state.lock:
        state.accepting_joins = False
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/round/reset")
def reset_round():
    state.reset()
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/match")
def match_students():
    with state.lock:
        _run_matching_unlocked()
    return redirect(url_for("teacher_dashboard"))


@app.post("/teacher/persona")
def set_persona():
    texts = [request.form.get(f"persona_{i}", "").strip() for i in range(1, 5)]
    legacy_text = request.form.get("persona", "").strip()
    if legacy_text and not any(texts):
        texts[0] = legacy_text

    with state.lock:
        state.ai_personas = texts
    return redirect(url_for("teacher_dashboard"))


@app.get("/teacher/api/rooms")
def teacher_rooms_api():
    with state.lock:
        rows = []
        for r in state.rooms.values():
            names = [state.participants[pid].name for pid in r.participant_ids if pid in state.participants]
            messages = []
            for m in r.messages:
                if m.sender_id == "ai":
                    sender_name = "AI"
                else:
                    sender = state.participants.get(m.sender_id)
                    sender_name = sender.name if sender else "Unknown"
                messages.append({"sender_name": sender_name, "text": m.text, "ts": m.ts})

            rows.append(
                {
                    "room_id": r.room_id,
                    "mode": r.mode,
                    "ai_persona": r.ai_persona,
                    "students": names,
                    "messages": messages,
                    "judgments": r.judgments,
                    "judgment_reasons": r.judgment_reasons,
                }
            )

        return jsonify(
            {
                "rooms": rows,
                "accepting_joins": state.accepting_joins,
                "matching_started": state.matching_started,
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
