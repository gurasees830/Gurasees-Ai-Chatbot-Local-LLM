"""Gurasees's Deep Learning Local LLM Chatbot: a local chat app for gpt-oss, Qwen and any other Ollama model. FastAPI backend.

- Serves the chat UI from ./static
- Streams the model's reasoning ("thinking") and its answer separately
- Strength control (fast / balanced / max): reasoning effort for gpt-oss, answer style for other models
- Reads uploaded documents (PDF, Word, Excel, PowerPoint, text, code, CSV, JSON...) into the prompt
- Photos, screenshots and video frames: text-only models get a description from a local vision model
  (e.g. gemma3, qwen2.5vl, llava). If you pick a vision model as the chat model, images go to it directly.
- Projects: chats grouped under shared instructions, files and a memory the model keeps updating
- Web links: URLs in a message are fetched and their text is given to the model
- Image generation: SD-Turbo in a separate worker process (imagegen_worker.py, its own .venv-image)
- Download ZIP: code blocks of a reply are packed into a .zip file

Run:  python server.py   then open http://localhost:8001
"""
import asyncio
import base64
import hashlib
import hmac
import io
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PREFERRED_MODEL = os.getenv("MODEL", "gpt-oss:20b")
VISION_MODEL = os.getenv("VISION_MODEL", "")  # empty = auto-pick an installed vision model
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "127.0.0.1")
# Hosting online. With APP_PASSWORD set, every /api call needs a login; without it the app is open (local use only).
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
# Websites allowed to call this server from a browser, comma separated, e.g. https://my-chat.netlify.app
# Not needed when the page is served by this server itself (the tunnel / same-origin setup).
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
# The web-link reader may open private addresses (localhost, 192.168.x.x) only when the app is not password-protected.
ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "0" if APP_PASSWORD else "1") == "1"
# How long Ollama keeps the model in memory after a request. Reloading a 20b model costs 10-60s.
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "2h")
# Tokens reserved for reasoning + reply when trimming old history to fit the context window.
REPLY_RESERVE = 4096
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DOC_CHARS = 400_000
DEFAULT_CTX = 8192
# Models too big for the graphics card run fully on the CPU, so they never push the small GPU model out of
# memory and several models can stay loaded at once. On a small GPU this costs them little or no speed.
BIG_MODELS_ON_CPU = os.getenv("BIG_MODELS_ON_CPU", "1") == "1"

BASE_DIR = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "data" / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = BASE_DIR / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MAX_MEMORY_ITEMS = 40

# Image generation runs in its own Python (PyTorch has no wheels for the chat server's Python).
IMAGE_PORT = int(os.getenv("IMAGE_PORT", "8002"))
IMAGE_URL = f"http://127.0.0.1:{IMAGE_PORT}"
IMAGE_PYTHON = BASE_DIR / ".venv-image" / "Scripts" / "python.exe"
if not IMAGE_PYTHON.exists():
    IMAGE_PYTHON = BASE_DIR / ".venv-image" / "bin" / "python"

# Web pages linked in a message.
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+")
MAX_URLS_PER_MESSAGE = 3
MAX_PAGE_CHARS = 60_000
PAGE_CACHE_SECS = 1800
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")

DEFAULT_SYSTEM = """You are a helpful, knowledgeable assistant running fully offline on the user's computer.
Answer accurately and clearly. Use Markdown: headings for long answers, bullet lists, tables when comparing things,
and fenced code blocks with a language tag for code. When the user attaches files, base your answer on their
contents and refer to them by file name. If something is not in the provided files and you are unsure, say so."""

# Always added, even to a custom system prompt: what this app can do around the model, so it doesn't refuse.
APP_ABILITIES = """About the app you are running in (Gurasees's Deep Learning Local LLM Chatbot):
- Web links: when the user's message contains URLs, the app downloads those pages and gives you their text inside
  <webpage url="..."> blocks. Use that content to answer and cite the URL. Never say you can't open links; if a
  page block reports an error, tell the user the page couldn't be read.
- Downloadable ZIP files: the app shows a "Download ZIP" button under every reply that contains code, which packs all
  the code blocks of the reply into a .zip file. So when the user asks for a zip, a project or several files, do NOT
  refuse or say you can't create files. Write every file completely, each in its own fenced code block, and put the
  file's path on the line right before its block, exactly like:  **File: src/app.py**
  Include config files (package.json, requirements.txt, .env.example, README.md) when they are needed, then tell the
  user to click "Download ZIP" below your reply.
- Images: the user can create images with the "Create image" option in the + menu (a local image model draws them).
  If asked to draw or generate a picture, suggest a detailed prompt and tell them to use "Create image"."""

IMAGE_PROMPT = ("Describe this image in thorough detail for someone who cannot see it. "
                "First transcribe ALL visible text exactly as written (keep line breaks, tables and numbers). "
                "Then describe the layout, objects, people, colours, charts/diagrams (with their values) and anything notable.")

VIDEO_PROMPT = ("These {n} images are frames taken in order from a video that is {dur} seconds long "
                "(at {times}). Describe what happens in the video from start to finish: the scene, people, "
                "actions, changes between frames, and transcribe any visible text exactly.")

# Strength for models without built-in reasoning: changes how they are asked to answer.
STRENGTH_PROMPTS = {
    "fast": "Be brief. Give the answer directly with only the explanation that is needed.",
    "balanced": "",
    "max": ("Work carefully. Think the problem through step by step before giving your final answer, "
            "consider edge cases, and double-check any code, maths or facts before you finish."),
}
# Strength for gpt-oss: its native reasoning effort.
STRENGTH_EFFORT = {"fast": "low", "balanced": "medium", "max": "high"}

LEARN_PROMPT = """You maintain the long-term memory of the project "{name}". Every future chat in this project reads it.

CURRENT MEMORY:
{memory}

NEW EXCHANGE (from the chat "{title}"):
USER: {user}

ASSISTANT: {assistant}

TASK: output the updated memory as a bullet list.
1. Start from the current memory. Keep each bullet unless the new exchange shows it is wrong or outdated.
2. Add only facts that are clearly stated in the exchange and will matter in later chats: the user's goals,
   what they are building, preferences, decisions, names, tools and tech stack, constraints.
3. Never guess or invent. Never write a bullet saying something is unknown, missing or not mentioned.
4. Each bullet is one plain fact under 25 words. No headings, no "Goal:" style labels, do not repeat these rules.
5. At most {limit} bullets. If nothing new is worth keeping, output the current memory unchanged.

Output ONLY the bullet list, one "- " per line."""
# Bullets a weak model writes that aren't memories ("No constraints mentioned", echoed rules).
JUNK_MEMORY = re.compile(r"\b(not (been )?(mentioned|specified|provided|stated)|no specific|none (yet|mentioned)|unknown"
                         r"|keep (each|existing)|current memory|new exchange)\b", re.I)

state = {"model": PREFERRED_MODEL}
caps_cache: Dict[str, List[str]] = {}
project_locks: Dict[str, asyncio.Lock] = {}
page_cache: Dict[str, tuple] = {}   # url -> (fetched_at, text or None, error or None)
image_proc: Optional[subprocess.Popen] = None
image_lock = asyncio.Lock()
client: httpx.AsyncClient  # shared connection pool, created in lifespan


# ---------------------------------------------------------------- Ollama helpers

async def list_model_info() -> List[dict]:
    r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return r.json().get("models", [])


async def list_models() -> List[str]:
    return [m["name"] for m in await list_model_info()]


def detect_vram_bytes() -> int:
    """Total memory of the first NVIDIA GPU, or 0 when there isn't one (then nothing is forced onto the CPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001
        return 0


VRAM_BYTES = int(float(os.getenv("GPU_VRAM_GB", "0")) * 1024 ** 3) or detect_vram_bytes()
model_sizes: Dict[str, int] = {}   # filled from /api/tags


def on_cpu(model: str) -> bool:
    size = model_sizes.get(model, 0)
    return BIG_MODELS_ON_CPU and VRAM_BYTES > 0 and size > VRAM_BYTES * 0.85


def run_options(model: str, num_ctx: int, **extra) -> dict:
    """Ollama options for a chat model. Every request to a model must use the same num_ctx and placement,
    otherwise Ollama reloads it."""
    opts = {"num_ctx": num_ctx, **extra}
    if on_cpu(model):
        opts["num_gpu"] = 0
    return opts


async def loaded_models() -> List[str]:
    r = await client.get(f"{OLLAMA_URL}/api/ps", timeout=5)
    return [m["name"] for m in r.json().get("models", [])]


async def preload(model: str, num_ctx: int) -> None:
    await client.post(f"{OLLAMA_URL}/api/generate", timeout=900, json={
        "model": model, "prompt": "", "keep_alive": KEEP_ALIVE, "options": run_options(model, num_ctx)})


def is_gpt_oss(model: str) -> bool:
    return model.startswith("gpt-oss")


def default_temperature(model: str) -> float:
    # OpenAI recommends 1.0 for gpt-oss; coding / chat models are more accurate when cooler.
    return 1.0 if is_gpt_oss(model) else 0.3


async def capabilities(model: str) -> List[str]:
    if model not in caps_cache:
        try:
            r = await client.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
            caps_cache[model] = r.json().get("capabilities", []) if r.status_code == 200 else []
        except Exception:  # noqa: BLE001
            return []
    return caps_cache[model]


def resolve_default(names: List[str]) -> str:
    for want in (PREFERRED_MODEL, f"{PREFERRED_MODEL}:latest"):
        if want in names:
            return want
    gpt = [n for n in names if n.startswith("gpt-oss")]
    return gpt[0] if gpt else (names[0] if names else PREFERRED_MODEL)


async def pick_vision_model(names: List[str]) -> Optional[str]:
    if VISION_MODEL and VISION_MODEL in names:
        return VISION_MODEL
    for n in names:
        if "vision" in await capabilities(n):
            return n
    return None


async def warm_up():
    """Load the default model into memory so the first reply starts quickly."""
    try:
        info = await list_model_info()
        model_sizes.update({m["name"]: m.get("size", 0) for m in info})
        names = [m["name"] for m in info]
        state["model"] = resolve_default(names)
        if state["model"] not in names:
            print(f"[Gurasees's Deep Learning Local LLM Chatbot] '{state['model']}' is not installed yet. Run: ollama pull {PREFERRED_MODEL}")
            return
        await preload(state["model"], DEFAULT_CTX)
        where = "CPU" if on_cpu(state["model"]) else "GPU/auto"
        print(f"[Gurasees's Deep Learning Local LLM Chatbot] model '{state['model']}' loaded and ready ({where})")
    except Exception as e:  # noqa: BLE001
        print(f"[Gurasees's Deep Learning Local LLM Chatbot] warm-up skipped: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None))
    task = asyncio.create_task(warm_up())
    yield
    task.cancel()
    if image_proc and image_proc.poll() is None:
        image_proc.terminate()
    await client.aclose()


app = FastAPI(title="Gurasees's Deep Learning Local LLM Chatbot", lifespan=lifespan)

if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       max_age=3600)


# ---------------------------------------------------------------- login (only when APP_PASSWORD is set)

def session_token() -> str:
    """Derived from the password, so the password itself is never stored in the browser."""
    return hmac.new(APP_PASSWORD.encode(), b"gurasees-chat-session", hashlib.sha256).hexdigest()


def token_ok(token: str) -> bool:
    return bool(APP_PASSWORD) and hmac.compare_digest(token or "", session_token())


PUBLIC_API = {"/api/auth", "/api/login"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if APP_PASSWORD and path.startswith("/api/") and path not in PUBLIC_API and request.method != "OPTIONS":
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else request.query_params.get("token", "")
        if not token_ok(token):
            return JSONResponse({"detail": "Login required."}, status_code=401)
    return await call_next(request)


class LoginRequest(BaseModel):
    password: str


@app.get("/api/auth")
async def auth_status():
    return {"required": bool(APP_PASSWORD)}


@app.post("/api/login")
async def login(req: LoginRequest):
    if not APP_PASSWORD:
        return {"token": ""}
    if not hmac.compare_digest(req.password.encode(), APP_PASSWORD.encode()):
        await asyncio.sleep(1.5)   # slows down password guessing
        raise HTTPException(401, "Wrong password.")
    return {"token": session_token()}


# ---------------------------------------------------------------- document text extraction

def _xml_text(data: bytes, para_tag: str, text_tag: str) -> str:
    """Pull text out of an Office XML part, one line per paragraph/row."""
    root = ET.fromstring(data)
    lines = []
    for para in root.iter():
        if para.tag.endswith(para_tag):
            t = "".join(n.text or "" for n in para.iter() if n.tag.endswith(text_tag))
            if t.strip():
                lines.append(t)
    return "\n".join(lines)


def extract_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        parts = ["word/document.xml"] + sorted(n for n in z.namelist()
                                               if re.match(r"word/(header|footer|footnotes)\d*\.xml", n))
        return "\n\n".join(_xml_text(z.read(p), "}p", "}t") for p in parts if p in z.namelist())


def extract_pptx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return "\n\n".join(f"--- Slide {i} ---\n{_xml_text(z.read(s), '}p', '}t')}"
                           for i, s in enumerate(slides, 1))


def extract_xlsx(raw: bytes) -> str:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{ns}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
        names = []
        if "xl/workbook.xml" in z.namelist():
            names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{ns}sheet")]
        sheets = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        out = []
        for i, s in enumerate(sheets):
            rows = []
            for row in ET.fromstring(z.read(s)).iter(f"{ns}row"):
                cells = []
                for c in row.iter(f"{ns}c"):
                    v = c.find(f"{ns}v")
                    if c.get("t") == "s" and v is not None:
                        cells.append(shared[int(v.text)])
                    elif c.get("t") == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(f"{ns}t")))
                    else:
                        cells.append(v.text if v is not None and v.text else "")
                if any(cells):
                    rows.append(" | ".join(cells))
            title = names[i] if i < len(names) else f"Sheet{i + 1}"
            out.append(f"--- Sheet: {title} ---\n" + "\n".join(rows))
        return "\n\n".join(out)


def extract_pdf(raw: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    pages = [f"--- Page {i} ---\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages, 1)]
    text = "\n\n".join(pages)
    if not re.sub(r"--- Page \d+ ---|\s", "", text):
        raise ValueError("this PDF has no selectable text (it is probably scanned). "
                         "Take a screenshot of the pages and upload them as photos instead.")
    return text


def extract_text(name: str, raw: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(raw)
    if ext in (".docx", ".docm"):
        return extract_docx(raw)
    if ext in (".pptx", ".pptm"):
        return extract_pptx(raw)
    if ext in (".xlsx", ".xlsm"):
        return extract_xlsx(raw)
    if ext in (".doc", ".xls", ".ppt"):
        raise ValueError(f"old {ext} format isn't supported. Save it as {ext}x in Office and upload again.")
    # Everything else: treat as text if it decodes cleanly (code, csv, json, md, html, logs...).
    if b"\x00" in raw[:8000]:
        raise ValueError("this looks like a binary file, so its text can't be read.")
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------- web pages

class _PageText(HTMLParser):
    """Readable text of an HTML page: skips scripts, styles and navigation chrome, keeps headings and list items."""
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "form"}
    BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "pre", "blockquote",
             "header", "aside", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag in self.BLOCK:
            self.out.append("\n" + ("#" * int(tag[1]) + " " if re.fullmatch(r"h[1-6]", tag) else "- " if tag == "li" else ""))

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)

    def text(self) -> str:
        t = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.out))
        t = re.sub(r"(?m)^[ \t]*(?:#+|-)[ \t]*$", "", t)   # headings / list items that held only links or icons
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


async def check_public_url(url: str) -> None:
    """Refuse links that point at this PC or the local network (the server would otherwise open them for visitors)."""
    if ALLOW_PRIVATE_URLS:
        return
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http and https links can be read")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                                                              type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError("the address could not be found")
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise ValueError("links to private or local addresses are blocked on this server")


async def fetch_page(url: str) -> tuple:
    """(text, error) for a URL, cached for half an hour."""
    hit = page_cache.get(url)
    if hit and time.time() - hit[0] < PAGE_CACHE_SECS:
        return hit[1], hit[2]
    text, error = None, None
    try:
        target = url
        for _hop in range(6):   # follow redirects by hand so every hop is checked
            await check_public_url(target)
            r = await client.get(target, timeout=httpx.Timeout(20, read=30), follow_redirects=False,
                                 headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                target = str(httpx.URL(target).join(r.headers["location"]))
                continue
            break
        else:
            raise ValueError("too many redirects")
        if r.status_code >= 400:
            raise ValueError(f"the site answered HTTP {r.status_code}")
        ctype = r.headers.get("content-type", "").lower()
        raw = r.content[:MAX_UPLOAD_BYTES]
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = await asyncio.to_thread(extract_pdf, raw)
        elif "html" in ctype or raw.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
            p = _PageText()
            p.feed(r.text)
            text = (f"Title: {p.title.strip()}\n\n" if p.title.strip() else "") + p.text()
        elif ctype.startswith(("text/", "application/json", "application/xml")) or not ctype:
            text = r.text
        else:
            raise ValueError(f"it is a {ctype.split(';')[0]} file, not a web page")
        if not text.strip():
            raise ValueError("the page has no readable text (it may need JavaScript to show its content)")
        text = text[:MAX_PAGE_CHARS]
    except httpx.TimeoutException:
        error = "the site took too long to answer"
    except httpx.HTTPError as e:
        error = f"couldn't connect ({type(e).__name__})"
    except Exception as e:  # noqa: BLE001
        error = str(e)
    page_cache[url] = (time.time(), text, error)
    return text, error


def urls_in(text: str) -> List[str]:
    found = []
    for u in URL_RE.findall(text or ""):
        u = u.rstrip(".,;:!?")
        if u not in found and urlparse(u).netloc:
            found.append(u)
    return found[:MAX_URLS_PER_MESSAGE]


# ---------------------------------------------------------------- image generation worker

def image_gen_installed() -> bool:
    return IMAGE_PYTHON.exists() and (BASE_DIR / "imagegen_worker.py").exists()


async def ensure_image_worker() -> None:
    """Start imagegen_worker.py in .venv-image if it isn't running, and wait until it answers."""
    global image_proc
    try:
        await client.get(f"{IMAGE_URL}/health", timeout=2)
        return
    except httpx.HTTPError:
        pass
    if not image_gen_installed():
        raise HTTPException(503, "Image generation isn't installed yet. Run setup_image.bat in the app folder, "
                                 "then try again.")
    if image_proc is None or image_proc.poll() is not None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        log = open(BASE_DIR / "data" / "imagegen.log", "ab")
        image_proc = subprocess.Popen([str(IMAGE_PYTHON), str(BASE_DIR / "imagegen_worker.py")], cwd=BASE_DIR,
                                      stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                      env={**os.environ, "IMAGE_PORT": str(IMAGE_PORT)})
    for _ in range(120):   # importing torch can take a while on the first start
        await asyncio.sleep(1)
        if image_proc.poll() is not None:
            raise HTTPException(500, "The image worker stopped while starting. See data/imagegen.log for details.")
        try:
            await client.get(f"{IMAGE_URL}/health", timeout=2)
            return
        except httpx.HTTPError:
            continue
    raise HTTPException(504, "The image worker didn't start within 2 minutes. See data/imagegen.log.")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 2
    seed: Optional[int] = None


@app.post("/api/image")
async def create_image(req: ImageRequest):
    async with image_lock:
        await ensure_image_worker()
        try:
            r = await client.post(f"{IMAGE_URL}/generate", json=req.model_dump(), timeout=httpx.Timeout(10, read=1800))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Image worker error: {type(e).__name__}. See data/imagegen.log.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise HTTPException(r.status_code, f"Image generation failed: {detail}")
    j = r.json()
    iid = uuid.uuid4().hex
    (IMAGES_DIR / f"{iid}.png").write_bytes(base64.b64decode(j["png"]))
    return {"id": iid, "url": f"/api/images/{iid}.png", "seed": j["seed"], "seconds": j["seconds"],
            "device": j["device"], "model": j["model"], "width": j["width"], "height": j["height"]}


@app.get("/api/images/{name}")
async def get_image(name: str):
    if not re.fullmatch(r"[a-f0-9]{32}\.png", name) or not (IMAGES_DIR / name).exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(IMAGES_DIR / name, media_type="image/png")


# ---------------------------------------------------------------- zip download

class ZipEntry(BaseModel):
    path: str
    content: str


class ZipRequest(BaseModel):
    name: str = "code"
    files: List[ZipEntry]


def safe_zip_path(p: str, used: set) -> str:
    p = re.sub(r"^[A-Za-z]:", "", p.strip())   # drop a Windows drive letter
    parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", s).strip() for s in re.split(r"[\\/]+", p)]
    parts = [s for s in parts if s and s not in (".", "..")]
    path = "/".join(parts) or "file.txt"
    stem, dot, ext = path.rpartition(".") if "." in path.rsplit("/", 1)[-1] else (path, "", "")
    n = 2
    while path.lower() in used:
        path = f"{stem}-{n}{dot}{ext}" if dot else f"{stem}-{n}"
        n += 1
    used.add(path.lower())
    return path


@app.post("/api/zip")
async def make_zip(req: ZipRequest):
    if not req.files:
        raise HTTPException(400, "No files to pack.")
    buf, used = io.BytesIO(), set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in req.files:
            z.writestr(safe_zip_path(f.path, used), f.content)
    name = re.sub(r"[^\w.-]+", "-", req.name).strip("-")[:60] or "code"
    return StreamingResponse(iter([buf.getvalue()]), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


# ---------------------------------------------------------------- API

@app.get("/api/health")
async def health():
    try:
        info = await list_model_info()
    except Exception:  # noqa: BLE001
        return {"ollama": False, "models": [], "default_model": state["model"], "vision_model": None,
                "image_gen": image_gen_installed()}
    names = [m["name"] for m in info]
    model_sizes.update({m["name"]: m.get("size", 0) for m in info})
    state["model"] = resolve_default(names)
    models = [{"name": m["name"], "capabilities": await capabilities(m["name"]), "size": m.get("size", 0),
               "params": m.get("details", {}).get("parameter_size", ""),
               "family": m.get("details", {}).get("family", ""),
               "device": "cpu" if on_cpu(m["name"]) else "gpu"} for m in info]
    return {"ollama": True, "models": models, "default_model": state["model"],
            "loaded": await loaded_models(), "vram_gb": round(VRAM_BYTES / 1024 ** 3, 1),
            "default_installed": any(n.startswith("gpt-oss") for n in names),
            "vision_model": await pick_vision_model(names), "image_gen": image_gen_installed()}


async def read_upload(file: UploadFile) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 50 MB.")
    try:
        text = await asyncio.to_thread(extract_text, file.filename or "file", raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Couldn't read {file.filename}: {e}")
    text = text.strip()
    if not text:
        raise HTTPException(422, f"{file.filename} contains no readable text.")
    truncated = len(text) > MAX_DOC_CHARS
    return {"name": file.filename, "text": text[:MAX_DOC_CHARS], "chars": min(len(text), MAX_DOC_CHARS),
            "truncated": truncated}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    return await read_upload(file)


@app.get("/api/loaded")
async def loaded():
    try:
        return {"loaded": await loaded_models()}
    except Exception:  # noqa: BLE001
        return {"loaded": []}


class PreloadRequest(BaseModel):
    model: str
    num_ctx: int = DEFAULT_CTX


@app.post("/api/preload")
async def preload_model(req: PreloadRequest):
    """Start loading a model in the background when the user picks it, so it's ready by the time they send."""
    if req.model not in model_sizes:
        model_sizes.update({m["name"]: m.get("size", 0) for m in await list_model_info()})
    if req.model in await loaded_models():
        return {"loaded": True}
    asyncio.create_task(preload(req.model, req.num_ctx))
    return {"loaded": False}


@app.post("/api/unload")
async def unload_model(req: PreloadRequest):
    """Free a model's memory now (keep_alive 0), e.g. to give gpt-oss all the RAM."""
    await client.post(f"{OLLAMA_URL}/api/generate", timeout=60,
                      json={"model": req.model, "prompt": "", "keep_alive": 0})
    return {"loaded": await loaded_models()}


# ---------------------------------------------------------------- projects (stored in data/projects/<id>.json)

def _project_path(pid: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", pid):
        raise HTTPException(404, "Project not found.")
    return PROJECTS_DIR / f"{pid}.json"


def load_project(pid: str) -> dict:
    path = _project_path(pid)
    if not path.exists():
        raise HTTPException(404, "Project not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_project(p: dict) -> None:
    p["updated"] = int(time.time() * 1000)
    path = _project_path(p["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def project_summary(p: dict) -> dict:
    """Project without file contents, for the browser."""
    files = [{k: f[k] for k in ("id", "name", "chars")} for f in p.get("files", [])]
    return {**p, "files": files}


class ProjectIn(BaseModel):
    name: Optional[str] = None
    instructions: Optional[str] = None
    memory: Optional[List[str]] = None
    auto_learn: Optional[bool] = None


@app.get("/api/projects")
async def projects_list():
    out = []
    for path in PROJECTS_DIR.glob("*.json"):
        try:
            out.append(project_summary(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda p: p.get("created", 0))


@app.post("/api/projects")
async def projects_create(body: ProjectIn):
    now = int(time.time() * 1000)
    p = {"id": uuid.uuid4().hex, "name": (body.name or "").strip() or "New project",
         "instructions": (body.instructions or "").strip(), "memory": [], "files": [],
         "auto_learn": True if body.auto_learn is None else body.auto_learn, "created": now}
    save_project(p)
    return project_summary(p)


@app.patch("/api/projects/{pid}")
async def projects_update(pid: str, body: ProjectIn):
    async with project_locks.setdefault(pid, asyncio.Lock()):
        p = load_project(pid)
        if body.name is not None:
            p["name"] = body.name.strip() or p["name"]
        if body.instructions is not None:
            p["instructions"] = body.instructions.strip()
        if body.memory is not None:
            p["memory"] = [m.strip() for m in body.memory if m.strip()][:MAX_MEMORY_ITEMS]
        if body.auto_learn is not None:
            p["auto_learn"] = body.auto_learn
        save_project(p)
    return project_summary(p)


@app.delete("/api/projects/{pid}")
async def projects_delete(pid: str):
    _project_path(pid).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/projects/{pid}/files")
async def projects_add_file(pid: str, file: UploadFile = File(...)):
    doc = await read_upload(file)
    async with project_locks.setdefault(pid, asyncio.Lock()):
        p = load_project(pid)
        p.setdefault("files", []).append({"id": uuid.uuid4().hex, "name": doc["name"], "text": doc["text"],
                                          "chars": doc["chars"]})
        save_project(p)
    return project_summary(p)


@app.delete("/api/projects/{pid}/files/{fid}")
async def projects_remove_file(pid: str, fid: str):
    async with project_locks.setdefault(pid, asyncio.Lock()):
        p = load_project(pid)
        p["files"] = [f for f in p.get("files", []) if f["id"] != fid]
        save_project(p)
    return project_summary(p)


def project_context(p: dict, budget_chars: int) -> tuple:
    """System-prompt section for a project. Returns (text, names of files that didn't fully fit)."""
    parts = [f'You are working inside the project "{p["name"]}".']
    if p.get("instructions"):
        parts.append("Project instructions (always follow these):\n" + p["instructions"])
    if p.get("memory"):
        parts.append("What you remember about this project from earlier chats (use it naturally, "
                     "and don't list it back unless asked):\n" + "\n".join(f"- {m}" for m in p["memory"]))
    text = "\n\n".join(parts)
    cut = []
    files = p.get("files", [])
    if files:
        blocks = []
        room = budget_chars - len(text)
        for f in files:
            body = f["text"]
            if len(body) > room:
                cut.append(f["name"])
                body = body[:max(room, 0)]
            if body:
                blocks.append(f'<file name="{f["name"]}">\n{body}\n</file>')
            room -= len(body) + 40
        if blocks:
            text += "\n\nProject files (reference material for every chat in this project):\n\n" + "\n\n".join(blocks)
    return text, cut


def parse_bullets(text: str) -> List[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    items = []
    for ln in text.splitlines():
        m = re.match(r"\s*(?:[-*•]|\d+[.)])\s+(.+)", ln)
        if m:
            item = m.group(1).strip().strip("*").strip()
            item = re.sub(r"^[A-Z][\w /'-]{1,24}:\s+(?=\S)", "", item)   # "Tech stack: Flask..." -> "Flask..."
            if item and item not in items and not JUNK_MEMORY.search(item):
                items.append(item)
    return items


class LearnRequest(BaseModel):
    model: Optional[str] = None
    user: str
    assistant: str
    title: str = ""
    num_ctx: int = 8192            # must match the chat's, or Ollama reloads the model


@app.post("/api/projects/{pid}/learn")
async def projects_learn(pid: str, req: LearnRequest):
    """Ask the model to fold the latest exchange into the project's memory."""
    async with project_locks.setdefault(pid, asyncio.Lock()):
        p = load_project(pid)
        old = p.get("memory", [])
        if not p.get("auto_learn", True):
            return {"memory": old, "added": [], "removed": []}
        model = req.model or state["model"]
        prompt = LEARN_PROMPT.format(
            name=p["name"], title=req.title or "untitled", limit=MAX_MEMORY_ITEMS,
            memory="\n".join(f"- {m}" for m in old) or "(empty)",
            user=req.user[:4000], assistant=req.assistant[:4000])
        payload = {"model": model, "stream": False, "keep_alive": KEEP_ALIVE,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": run_options(model, req.num_ctx, temperature=0.2)}
        if "thinking" in await capabilities(model):
            payload["think"] = "low" if is_gpt_oss(model) else False
        try:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Memory update failed: {e}")
        new = parse_bullets(r.json().get("message", {}).get("content", ""))[:MAX_MEMORY_ITEMS]
        if not new and old:
            return {"memory": old, "added": [], "removed": []}   # model returned junk: keep what we had
        p["memory"] = new
        save_project(p)
    return {"memory": new, "added": [m for m in new if m not in old], "removed": [m for m in old if m not in new]}


# ---------------------------------------------------------------- chat

class Attachment(BaseModel):
    id: str
    name: str
    kind: str                      # "doc", "image" or "video"
    text: Optional[str] = None     # document text, or a cached image/video description
    data: Optional[str] = None     # base64 image
    frames: List[str] = []         # base64 frames of a video
    times: List[float] = []        # seconds into the video for each frame
    duration: float = 0


class Message(BaseModel):
    role: str
    content: str
    attachments: List[Attachment] = []


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    system: Optional[str] = None
    strength: Optional[str] = None     # fast / balanced / max
    think: str = "medium"              # legacy: gpt-oss reasoning effort, used when strength is missing
    temperature: Optional[float] = None  # None = the model's recommended default
    top_p: Optional[float] = None
    num_ctx: int = 8192
    project_id: Optional[str] = None


def has_visual(a: Attachment) -> bool:
    return bool(a.data) if a.kind == "image" else bool(a.frames) if a.kind == "video" else False


def build_content(m: Message, model_sees_images: bool, pages: Optional[Dict[str, tuple]] = None,
                  page_limit: int = MAX_PAGE_CHARS) -> str:
    parts = []
    if m.role == "user" and pages:
        for url in urls_in(m.content):
            text, error = pages.get(url, (None, "not fetched"))
            if text:
                cut = ' note="only the beginning of the page fits"' if len(text) > page_limit else ""
                parts.append(f'<webpage url="{url}"{cut}>\n{text[:page_limit]}\n</webpage>')
            else:
                parts.append(f'<webpage url="{url}" error="{error}">[This page could not be read: {error}]</webpage>')
    for a in m.attachments:
        if a.kind == "doc":
            parts.append(f'<file name="{a.name}">\n{a.text or ""}\n</file>')
        elif a.kind == "video" and model_sees_images and a.frames:
            stamps = ", ".join(f"{t:.1f}s" for t in a.times)
            parts.append(f'<video name="{a.name}" duration="{a.duration:.1f}s">The attached images are '
                         f'{len(a.frames)} frames from this video, in order, at {stamps}.</video>')
        elif a.kind in ("image", "video") and not (model_sees_images and has_visual(a)):
            if a.text:
                parts.append(f'<{a.kind} name="{a.name}" note="description by a vision model">\n{a.text}\n</{a.kind}>')
            else:
                parts.append(f'<{a.kind} name="{a.name}">[{a.kind} attached, but no vision model was available '
                             f'to look at it]</{a.kind}>')
    if parts:
        return "Attached files:\n\n" + "\n\n".join(parts) + "\n\n---\n\n" + m.content
    return m.content


def trim_history(messages: List[dict], system: str, num_ctx: int) -> List[dict]:
    """Keep the newest messages that fit in the context window (~3.5 chars per token)."""
    budget = int(max(num_ctx - REPLY_RESERVE, 1024) * 3.5) - len(system)
    kept, used = [], 0
    for m in reversed(messages):
        used += len(m["content"])
        if kept and used > budget:
            break
        kept.append(m)
    return list(reversed(kept))


def line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


async def describe_visual(vision_model: str, a: Attachment) -> str:
    if a.kind == "video":
        prompt = VIDEO_PROMPT.format(n=len(a.frames), dur=round(a.duration, 1),
                                     times=", ".join(f"{t:.1f}s" for t in a.times))
        images = a.frames
    else:
        prompt, images = IMAGE_PROMPT, [a.data]
    r = await client.post(f"{OLLAMA_URL}/api/chat", timeout=900, json={
        "model": vision_model, "stream": False, "keep_alive": "10m",
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "options": {"temperature": 0.2},
    })
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model = req.model or state["model"]
    strength = req.strength or {"low": "fast", "high": "max"}.get(req.think, "balanced")
    if strength not in STRENGTH_EFFORT:
        strength = "balanced"
    system = ((req.system or "").strip() or DEFAULT_SYSTEM) + "\n\n" + APP_ABILITIES

    # Streams NDJSON events:
    #   {"status": "..."}                       progress note (e.g. reading an image)
    #   {"warn": "..."}                         note kept above the answer
    #   {"image_desc": {"id":..,"text":..}}     cache this description on the attachment
    #   {"t": "..."} reasoning chunk           {"c": "..."} answer chunk
    #   {"done": true, ...stats}               {"error": "..."}
    async def event_stream():
        nonlocal system
        try:
            caps = await capabilities(model)
            sees_images = "vision" in caps
            thinks = "thinking" in caps
            names = None

            if req.project_id:
                project = load_project(req.project_id)
                ctx_text, cut = project_context(project, int(req.num_ctx * 3.5 * 0.5))
                system += "\n\n" + ctx_text
                if cut:
                    yield line({"warn": f"Project files too long for the current context, so only part of "
                                        f"{', '.join(cut)} was read. Raise Context length in Settings."})
            if not thinks and STRENGTH_PROMPTS.get(strength):
                system += "\n\n" + STRENGTH_PROMPTS[strength]

            # Images and videos for a text-only model: have a local vision model describe them first.
            for m in req.messages:
                for a in m.attachments:
                    if a.kind not in ("image", "video") or a.text or not has_visual(a) or sees_images:
                        continue
                    if names is None:
                        names = await list_models()
                    vision = await pick_vision_model(names)
                    if not vision:
                        yield line({"warn": f"{model} can't see images or video and no vision model is installed, "
                                            f"so {a.name} was skipped. Run: ollama pull gemma3:4b"})
                        continue
                    what = "video frames of " if a.kind == "video" else ""
                    yield line({"status": f"Looking at {what}{a.name} with {vision}…"})
                    a.text = await describe_visual(vision, a)
                    yield line({"image_desc": {"id": a.id, "text": a.text}})

            # Web links in the user's messages: fetch the pages (cached) and hand their text to the model.
            pages: Dict[str, tuple] = {}
            for m in req.messages:
                if m.role != "user":
                    continue
                for url in urls_in(m.content):
                    if url in pages:
                        continue
                    cached = page_cache.get(url)
                    if not (cached and time.time() - cached[0] < PAGE_CACHE_SECS):
                        yield line({"status": f"Reading {urlparse(url).netloc}…"})
                    pages[url] = await fetch_page(url)
                    if pages[url][1] and m is req.messages[-1]:
                        yield line({"warn": f"Couldn't read {url}: {pages[url][1]}"})
            # Pages of the newest message share half of the context window.
            newest = urls_in(req.messages[-1].content) if req.messages and req.messages[-1].role == "user" else []
            page_limit = min(MAX_PAGE_CHARS, int((req.num_ctx - REPLY_RESERVE) * 3.5 * 0.5) // max(1, len(newest)))
            long_pages = [u for u in newest if pages.get(u, (None,))[0] and len(pages[u][0]) > page_limit]
            if long_pages:
                yield line({"warn": f"Only the first {page_limit:,} characters of {', '.join(long_pages)} fit in the "
                                    f"current context. Raise Context length in Settings to read more of the page."})

            history = []
            for m in req.messages:
                msg = {"role": m.role, "content": build_content(m, sees_images, pages, page_limit)}
                imgs = []
                for a in m.attachments:
                    if a.kind == "image" and a.data:
                        imgs.append(a.data)
                    elif a.kind == "video":
                        imgs.extend(a.frames)
                if sees_images and imgs:
                    msg["images"] = imgs
                history.append(msg)
            history = trim_history(history, system, req.num_ctx)

            temperature = req.temperature if req.temperature is not None else default_temperature(model)
            top_p = req.top_p if req.top_p is not None else (1.0 if is_gpt_oss(model) else 0.9)
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}] + history,
                "stream": True,
                "keep_alive": KEEP_ALIVE,
                "options": run_options(model, req.num_ctx, temperature=temperature, top_p=top_p),
            }
            if thinks:
                # gpt-oss takes an effort level; other thinking models take true/false.
                payload["think"] = STRENGTH_EFFORT[strength] if is_gpt_oss(model) else strength != "fast"

            yield line({"status": ""})
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")
                    try:
                        body = json.loads(body).get("error", body)
                    except Exception:  # noqa: BLE001
                        pass
                    if "not found" in body:
                        body += f"  →  Run: ollama pull {model}"
                    yield line({"error": f"Ollama error ({r.status_code}): {body}"})
                    return
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if data.get("error"):
                        yield line({"error": data["error"]})
                        return
                    msg = data.get("message", {})
                    if msg.get("thinking"):
                        yield line({"t": msg["thinking"]})
                    if msg.get("content"):
                        yield line({"c": msg["content"]})
                    if data.get("done"):
                        tokens = data.get("eval_count", 0)
                        secs = data.get("eval_duration", 0) / 1e9
                        yield line({"done": True, "tokens": tokens,
                                    "prompt_tokens": data.get("prompt_eval_count", 0),
                                    "tps": round(tokens / secs, 1) if secs else 0,
                                    "seconds": round(data.get("total_duration", 0) / 1e9, 2),
                                    "model": model})
        except httpx.ConnectError:
            yield line({"error": f"Can't reach Ollama at {OLLAMA_URL}. Make sure the Ollama app is running."})
        except Exception as e:  # noqa: BLE001
            yield line({"error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/config.js")
async def config_js():
    """Same address the static hosts serve; here the API is on the same origin, so apiBase is empty."""
    return FileResponse(BASE_DIR / "static" / "config.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
