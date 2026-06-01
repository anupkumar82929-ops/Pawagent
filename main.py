from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from chat_service import chat_reply, chat_reply_stream, sanitize_hint  # noqa: E402
from config import TOP_K  # noqa: E402
from inference import model_ready, predict_bytes  # noqa: E402

STATIC = ROOT / "static"

app = FastAPI(title="Pawgent", description="Dog breed ID + care assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str
    content: str = Field("", max_length=8000)

    @field_validator("role")
    @classmethod
    def role_ok(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("history role must be user or assistant")
        return v


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    breed_hint: str | None = None
    history: list[ChatTurn] = Field(default_factory=list)

    @field_validator("history")
    @classmethod
    def trim_history(cls, v: list[ChatTurn]) -> list[ChatTurn]:
        if len(v) > 20:
            return v[-20:]
        return v


def _history_payload(body: ChatBody) -> list[dict]:
    return [
        {"role": t.role, "content": t.content.strip()}
        for t in body.history
        if t.role in ("user", "assistant") and t.content.strip()
    ]


@app.get("/api/health")
def health():
    return {"ok": True, "model_ready": model_ready()}


@app.get("/api/status")
def status():
    return {"model_ready": model_ready(), "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip())}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not model_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run: python scripts/build_class_names.py && python train.py",
        )
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 12MB).")
    try:
        preds = predict_bytes(raw, top_k=TOP_K)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e
    top = preds[0] if preds else None
    return {"predictions": preds, "top": top}


@app.post("/api/chat")
async def chat(body: ChatBody):
    hint = sanitize_hint(body.breed_hint)
    text = await chat_reply(body.message, hint, _history_payload(body))
    return {"reply": text}


@app.post("/api/chat/stream")
async def chat_stream(body: ChatBody):
    hint = sanitize_hint(body.breed_hint)
    hist = _history_payload(body)

    async def gen():
        try:
            async for piece in chat_reply_stream(body.message, hint, hist):
                yield f"data: {json.dumps({'c': piece})}\n\n".encode()
            yield b'data: {"done": true}\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        return JSONResponse({"error": "Missing static/index.html"}, status_code=500)
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
