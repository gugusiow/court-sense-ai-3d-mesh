"""CourtSense API.

Accounts and kept sessions live in SQLite. Analysis calls the existing
pipeline. A professional mesh file is not produced here yet.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

import jwt
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import db
from courtsense.domain.models import ConsentRecord, Session
from courtsense.pipeline.runner import run_synthetic_analysis

app = FastAPI(title="CourtSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "courtsense-demo-secret-key-123"
ALGORITHM = "HS256"
security = HTTPBearer()
VIDEO_DIR = Path(__file__).resolve().parent / "data" / "videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".m4v": "video/mp4",
}

PLACEHOLDER_TIPS = [
    {
        "text": "Coaching tips are not produced by the pipeline yet.",
        "source": "placeholder",
    }
]
class SignupRequest(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    email: str
    password: str = Field(min_length=8)
    age: int = Field(ge=1, le=120)
    height_cm: float = Field(gt=0, le=300)
    weight_kg: float = Field(gt=0, le=500)


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdate(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    age: int = Field(ge=1, le=120)
    height_cm: float = Field(gt=0, le=300)
    weight_kg: float = Field(gt=0, le=500)


def clean_email(email: str) -> str:
    cleaned = email.strip().lower()
    if "@" not in cleaned or cleaned.startswith("@") or cleaned.endswith("@"):
        raise HTTPException(status_code=400, detail="Enter a valid email")
    return cleaned


def issue_token(user_id: str) -> str:
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    with db.connect() as connection:
        row = connection.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (payload["user_id"],)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["user_id"]


def build_result(user_id: str, session_id: str) -> dict:
    result = run_synthetic_analysis(
        Session(session_id, user_id),
        ConsentRecord(
            participant_id=user_id,
            history_retention_allowed=True,
            training_retention_allowed=False,
        ),
    )
    return {
        "features": [
            {
                "name": feature.feature_name,
                "value": feature.value,
                "units": feature.units,
                "valid": feature.valid,
            }
            for feature in result.features
        ],
        "tips": PLACEHOLDER_TIPS,
    }


def video_response(path: Path) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file is missing")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))


def session_view(row: sqlite3.Row, include_result: bool) -> dict:
    view = {
        "session_id": row["session_id"],
        "retention": row["retention"],
        "status": row["status"],
        "video_name": row["video_name"],
        "error": row["error"],
        "created_at": row["created_at"],
        "result": None,
    }
    if include_result and row["result_json"]:
        view["result"] = json.loads(row["result_json"])
    return view


def run_stored_session(session_id: str, user_id: str) -> None:
    try:
        with db.connect() as connection:
            connection.execute(
                "UPDATE sessions SET status = 'analyzing', error = NULL WHERE session_id = ?",
                (session_id,),
            )
        payload = build_result(user_id, session_id)
        with db.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET status = 'ready', result_json = ?, error = NULL
                WHERE session_id = ?
                """,
                (json.dumps(payload), session_id),
            )
    except Exception as exc:
        with db.connect() as connection:
            connection.execute(
                "UPDATE sessions SET status = 'failed', error = ? WHERE session_id = ?",
                (str(exc), session_id),
            )


db.init()


@app.post("/api/signup")
def signup(body: SignupRequest):
    email = clean_email(body.email)
    user_id = f"user-{uuid.uuid4().hex[:12]}"
    try:
        with db.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id, email, password_hash, first_name, last_name, age, height_cm, weight_kg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email,
                    db.hash_password(body.password),
                    body.first_name.strip(),
                    body.last_name.strip(),
                    body.age,
                    body.height_cm,
                    body.weight_kg,
                ),
            )
            row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with that email already exists")
    return {"access_token": issue_token(user_id), "user": db.public_user(row)}


@app.post("/api/login")
def login(body: LoginRequest):
    email = clean_email(body.email)
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not db.verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    return {"access_token": issue_token(row["user_id"]), "user": db.public_user(row)}


@app.get("/api/me")
def me(user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return db.public_user(row)


@app.patch("/api/me")
def update_me(body: ProfileUpdate, user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET first_name = ?, last_name = ?, age = ?, height_cm = ?, weight_kg = ?
            WHERE user_id = ?
            """,
            (body.first_name.strip(), body.last_name.strip(), body.age, body.height_cm, body.weight_kg, user_id),
        )
        row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return db.public_user(row)


@app.delete("/api/me")
def delete_me(user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT video_path FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchall()
        connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    for row in rows:
        if row["video_path"]:
            Path(row["video_path"]).unlink(missing_ok=True)
    return {"deleted": True}


@app.get("/api/sessions")
def list_sessions(user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [session_view(row, include_result=False) for row in rows]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_view(row, include_result=True)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        row = connection.execute(
            "SELECT video_path FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        connection.execute(
            "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
    if row["video_path"]:
        Path(row["video_path"]).unlink(missing_ok=True)
    return {"deleted": True}


@app.get("/api/sessions/{session_id}/video")
def session_video(session_id: str, user_id: str = Depends(verify_token)):
    with db.connect() as connection:
        row = connection.execute(
            "SELECT video_path FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if row is None or not row["video_path"]:
        raise HTTPException(status_code=404, detail="This run has no saved video")
    return video_response(Path(row["video_path"]))


@app.post("/api/sessions")
async def create_session(
    video: UploadFile = File(...),
    user_id: str = Depends(verify_token),
):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload an mp4, mov, webm, or m4v video")
    contents = await video.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Video file is empty")

    session_id = f"session-{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc).isoformat()
    folder = VIDEO_DIR / user_id
    folder.mkdir(parents=True, exist_ok=True)
    video_path = folder / f"{session_id}{suffix}"
    video_path.write_bytes(contents)
    with db.connect() as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                session_id, user_id, retention, status, video_name, video_path, created_at
            ) VALUES (?, ?, ?, 'received', ?, ?, ?)
            """,
            (session_id, user_id, "history", video.filename, str(video_path), created_at),
        )
    threading.Thread(
        target=run_stored_session,
        args=(session_id, user_id),
        daemon=True,
    ).start()
    return {
        "session_id": session_id,
        "retention": "history",
        "status": "received",
        "video_name": video.filename,
        "error": None,
        "created_at": created_at,
        "result": None,
    }
