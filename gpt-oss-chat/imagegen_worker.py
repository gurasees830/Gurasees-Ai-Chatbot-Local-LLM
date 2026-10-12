"""Image generation worker for Gurasees's Deep Learning Local LLM Chatbot.

Runs in its own Python environment (.venv-image) because PyTorch needs an older Python than the chat server.
server.py starts it automatically on the first image request; you never need to run it by hand.

Model: SD-Turbo (Stable Diffusion 2.1 distilled to 1-4 steps). Fits a 4 GB GPU; falls back to the CPU.
The weights (~2.5 GB) download from Hugging Face on first use and are cached in ~/.cache/huggingface.
"""
import base64
import io
import os
import threading
import time
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stabilityai/sd-turbo")
PORT = int(os.getenv("IMAGE_PORT", "8002"))

app = FastAPI(title="Image worker")
pipe = None
device = "cpu"
lock = threading.Lock()


def load_pipeline(force_cpu: bool = False, full_precision: bool = False):
    """Load SD-Turbo. Small GPUs load one part at a time (model CPU offload) so Ollama's models can stay loaded."""
    global pipe, device
    from diffusers import AutoPipelineForText2Image

    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda and "GTX 16" in torch.cuda.get_device_name(0):
        full_precision = True   # GTX 16xx cards return NaNs (black images) in half precision
    dtype = torch.float16 if use_cuda and not full_precision else torch.float32
    p = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL, dtype=dtype, variant="fp16")
    p.set_progress_bar_config(disable=True)
    if use_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram_gb < 8:
            p.enable_model_cpu_offload()
        else:
            p.to("cuda")
        p.enable_attention_slicing()
        device = f"GPU ({torch.cuda.get_device_name(0)}{', fp32' if full_precision else ''})"
    else:
        p.to("cpu")
        device = "CPU"
    pipe = p
    print(f"[image worker] {IMAGE_MODEL} loaded on {device}", flush=True)


def is_blank(img) -> bool:
    """GTX 16xx cards can produce NaNs in half precision, which come out as an all-black image."""
    lo, hi = img.convert("L").getextrema()
    return hi < 8


class GenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 2
    seed: Optional[int] = None


def run(req: GenRequest, seed: int):
    gen = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=req.prompt, negative_prompt=req.negative_prompt or None, width=req.width, height=req.height,
                num_inference_steps=req.steps, guidance_scale=0.0, generator=gen).images[0]


@app.get("/health")
def health():
    return {"loaded": pipe is not None, "device": device, "model": IMAGE_MODEL}


@app.post("/generate")
def generate(req: GenRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Describe the image you want.")
    req.width = max(256, min(768, req.width // 64 * 64))
    req.height = max(256, min(768, req.height // 64 * 64))
    req.steps = max(1, min(8, req.steps))
    seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "little")
    with lock:
        t0 = time.time()
        if pipe is None:
            load_pipeline()
        try:
            img = run(req, seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[image worker] GPU out of memory, switching to CPU", flush=True)
            load_pipeline(force_cpu=True)
            img = run(req, seed)
        if is_blank(img) and torch.cuda.is_available() and "fp32" not in device and device != "CPU":
            print("[image worker] blank image in half precision, reloading in full precision", flush=True)
            load_pipeline(full_precision=True)
            img = run(req, seed)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return {"png": base64.b64encode(buf.getvalue()).decode(), "seed": seed, "device": device,
            "seconds": round(time.time() - t0, 1), "model": IMAGE_MODEL, "width": req.width, "height": req.height}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
