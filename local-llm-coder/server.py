"""Gurasees Ollama: a local coding chatbot. FastAPI backend in front of Ollama.

- Serves the chat UI from ./static
- Streams model replies from Ollama (everything stays on your machine)
- Keeps the model loaded in GPU memory so replies start instantly
- Saves rated / corrected answers to data/feedback.jsonl (your training data)

Run:  python server.py   then open http://localhost:8000
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PREFERRED_MODEL = os.getenv("MODEL", "gurasees-coder")
FALLBACK_MODEL = "qwen2.5-coder:3b"
PORT = int(os.getenv("PORT", "8000"))
# How long Ollama keeps the model in memory after a request. Reloading costs 10-30s.
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "2h")
# Tokens reserved for the reply when trimming old history to fit the context window.
REPLY_RESERVE = 1024

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

BASE_SYSTEM = """You are Gurasees, an expert software engineer and competitive programmer who solves coding problems.

For every problem:
1. Restate the goal and constraints in one or two lines. If something is ambiguous, state your assumption.
2. Explain the approach and why it works. Give the time and space complexity.
3. Write complete, runnable code in a fenced code block with the language tag. Use clear names and no placeholders.
4. Trace one example through the code and list the important edge cases.

When debugging, find the root cause first, then show the minimal fix.
Be concise and accurate. Never invent library functions; if you are unsure, say so."""

# Extra instructions per mode, appended to BASE_SYSTEM.
MODES = {
    "solve": "",
    "debug": ("Mode: DEBUG. The user has broken code or an error message. Identify the root cause in 1-3 sentences, "
              "show the corrected code, and explain how to avoid the bug in future."),
    "explain": ("Mode: EXPLAIN. Teach the code or concept clearly to a learner. Walk through it step by step, "
                "use a small concrete example, and skip the complexity analysis unless it matters."),
    "optimize": ("Mode: OPTIMIZE. Give the current time/space complexity, then a faster or leaner version with its new "
                 "complexity, and note any trade-offs."),
    "review": ("Mode: CODE REVIEW. List problems ordered by severity (bugs, security, performance, readability), "
               "each with a one-line fix, then show the improved code."),
    "tests": ("Mode: TESTS. Write thorough unit tests using the idiomatic framework for the language (pytest, Jest, "
              "JUnit, etc.). Cover normal cases, edge cases and error cases."),
}

state = {"model": FALLBACK_MODEL}
client: httpx.AsyncClient  # shared connection pool, created in lifespan


async def list_models() -> List[str]:
    r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def resolve_default(names: List[str]) -> str:
    for want in (PREFERRED_MODEL, f"{PREFERRED_MODEL}:latest", FALLBACK_MODEL):
        if want in names:
            return want.removesuffix(":latest")
    return names[0] if names else FALLBACK_MODEL


async def warm_up():
    """Load the default model into GPU memory so the first chat reply is instant."""
    try:
        state["model"] = resolve_default(await list_models())
        await client.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": state["model"], "prompt": "", "keep_alive": KEEP_ALIVE}, timeout=180)
        print(f"[gurasees] model '{state['model']}' loaded and ready")
    except Exception as e:  # noqa: BLE001
        print(f"[gurasees] warm-up skipped: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None))
    task = asyncio.create_task(warm_up())
    yield
    task.cancel()
    await client.aclose()


app = FastAPI(title="Gurasees Ollama", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    mode: str = "solve"
    instructions: Optional[str] = None  # user's extra custom instructions
    temperature: float = 0.15
    num_ctx: int = 4096


class FeedbackRequest(BaseModel):
    # Full conversation up to and including the assistant reply being rated.
    messages: List[Message]
    rating: str = "up"          # "up" or "down"
    edited: bool = False        # True if the user corrected the reply
    model: Optional[str] = None


def count_feedback() -> int:
    if not FEEDBACK_FILE.exists():
        return 0
    with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_system(mode: str, instructions: Optional[str]) -> str:
    parts = [BASE_SYSTEM, MODES.get(mode, "")]
    if instructions and instructions.strip():
        parts.append("Additional instructions from the user:\n" + instructions.strip())
    return "\n\n".join(p for p in parts if p)


def trim_history(messages: List[dict], system: str, num_ctx: int) -> List[dict]:
    """Keep the newest messages that fit in the context window (~3.5 chars per token).

    A shorter prompt means faster prompt processing and no silent truncation by Ollama.
    """
    budget = int((num_ctx - REPLY_RESERVE) * 3.5) - len(system)
    kept, used = [], 0
    for m in reversed(messages):
        used += len(m["content"])
        if kept and used > budget:
            break
        kept.append(m)
    return list(reversed(kept))


@app.get("/api/health")
async def health():
    try:
        names = await list_models()
        state["model"] = resolve_default(names)
        return {"ollama": True, "models": names, "default_model": state["model"], "examples": count_feedback()}
    except Exception:  # noqa: BLE001
        return {"ollama": False, "models": [], "default_model": state["model"], "examples": count_feedback()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    system = build_system(req.mode, req.instructions)
    history = trim_history([m.model_dump() for m in req.messages], system, req.num_ctx)
    payload = {
        "model": req.model or state["model"],
        "messages": [{"role": "system", "content": system}] + history,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": req.temperature, "num_ctx": req.num_ctx},
    }

    def line(obj) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    # Streams NDJSON: {"c": "<text>"} chunks, then one {"done": true, ...stats}.
    async def token_stream():
        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")
                    try:
                        body = json.loads(body).get("error", body)
                    except Exception:  # noqa: BLE001
                        pass
                    yield line({"error": f"Ollama error ({r.status_code}): {body}"})
                    return
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if data.get("error"):
                        yield line({"error": data["error"]})
                        return
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield line({"c": chunk})
                    if data.get("done"):
                        tokens = data.get("eval_count", 0)
                        secs = data.get("eval_duration", 0) / 1e9
                        yield line({"done": True, "tokens": tokens,
                                    "tps": round(tokens / secs, 1) if secs else 0,
                                    "seconds": round(data.get("total_duration", 0) / 1e9, 2),
                                    "model": payload["model"]})
        except httpx.ConnectError:
            yield line({"error": f"Can't reach Ollama at {OLLAMA_URL}. Make sure the Ollama app is running."})
        except Exception as e:  # noqa: BLE001
            yield line({"error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(token_stream(), media_type="application/x-ndjson")


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    record = {
        "ts": int(time.time()),
        "model": req.model or state["model"],
        "rating": req.rating,
        "edited": req.edited,
        "messages": [m.model_dump() for m in req.messages],
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "examples": count_feedback()}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=PORT, reload=False)
