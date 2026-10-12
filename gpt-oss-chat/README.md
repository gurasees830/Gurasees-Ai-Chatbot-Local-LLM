# Gurasees's Deep Learning Local LLM Chatbot

Gurasees's Deep Learning Local LLM Chatbot is a private, offline chat app for **gpt-oss:20b**, **qwen2.5-coder** and any other model you've installed with Ollama.

## Start

Double-click **`start.bat`**. On every start it:

- starts the Ollama server if it isn't running,
- updates the app's Python packages,
- downloads gpt-oss:20b (~13 GB) and the gemma3:4b vision model (~3 GB) if they're missing,
- sets up image generation once (`setup_image.bat`, ~3 GB for PyTorch),
- opens http://localhost:8001.

## Share it online

Double-click **`start_public.bat`** for a free, password-protected public link from your PC, or see [HOSTING.md](../HOSTING.md)
for putting the website on Netlify, Vercel or Cloudflare Pages.

## New: web links, image generation, ZIP downloads

- **Web links**: paste one or more URLs (up to 3 per message) into your message. The app downloads each page (HTML, PDF or text) and gives its text to the model, so any model can summarise or answer questions about it. Pages that need JavaScript to show their content may come back empty.
- **Create image**: **+** → **Create image**, then describe the picture. You can also type `/image a red fox in snow` or "draw a picture of …". Images are drawn on your PC by **SD-Turbo** (Stable Diffusion) in a separate worker (`imagegen_worker.py`, own Python environment `.venv-image`, because PyTorch doesn't support the app's newer Python). The very first image downloads the model (~2.5 GB, about 10 minutes). On this PC's GTX 1650 an image then takes ~1 minute after a restart and ~40 seconds when warm (GTX 16xx cards must run it in full precision). Size and steps are in Settings. Images are saved in `data/images/`; worker logs in `data/imagegen.log`.
- **Download ZIP**: every reply with code gets a **Download ZIP** button that packs all its code blocks into a .zip. The models are told about this, so ask e.g. "build a todo app with React and FastAPI and give me a zip". Files are named from the `**File: path**` line the model writes before each block.

## Features

- **Model & strength picker** in the chat bar (the button next to **+**). Pick any installed model, then a strength:
  - **gpt-oss** (has built-in reasoning): Fast / Balanced / Max = low / medium / high reasoning effort.
  - **Qwen coder models** (no built-in reasoning): Fast = short direct answers, Balanced = normal, Max = asked to work step by step and double-check.
  - After you use a model, the menu shows how fast it actually runs on your PC (tok/s).
- **Projects** (sidebar → Projects → **+**): group chats that share
  - **Instructions** applied to every chat in the project,
  - **Files** the model reads in every chat (PDF, Word, Excel, code…),
  - **Memory**: after each reply the model writes down durable facts (your goals, decisions, preferences, tech stack). Every new chat in the project starts with that memory. You can view, add and delete memories on the project page, or turn **Learn from chats** off.
  - Projects are saved on disk in `data/projects/`, so they survive clearing the browser.
- **Attachments** (the **+** button): Photos, Take a screenshot, Video, Documents, Any file. You can also drag & drop, or paste screenshots with Ctrl+V.
- **Documents supported**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), TXT, Markdown, CSV, JSON, HTML, code files, logs.
- **Video**: the browser takes 3–8 frames from the clip (MP4/WEBM; no upload, no ffmpeg) and a vision model describes them. Audio is not used.
- Chat search, rename (click the title at the top), live reasoning view, Markdown with copy buttons, Stop / Regenerate, light / dark theme.
- **Settings** (gear icon): system prompt, context length, temperature (Auto by default), theme, show reasoning.

## Photos, screenshots and video: install a vision model once

gpt-oss and the Qwen coder models **can only read text**. To make images and video work, install a small vision model:

```
ollama pull gemma3:4b
```

The app finds it automatically. It reads each image (including any text in it) and passes a detailed description to your chosen model.

## Performance on this PC (GTX 1650 4 GB, 14 GB RAM)

| Model | Speed | Notes |
|---|---|---|
| gurasees-coder, qwen2.5-coder:3b | ~35 tok/s | Fits in the GPU. Best for everyday use. |
| qwen2.5-coder:7b | ~3 tok/s | Doesn't fully fit in 4 GB VRAM, so it runs partly on the CPU. |
| gpt-oss:20b | ~3 tok/s | Runs mostly on the CPU. First reply after starting takes ~3–4 min to load. |

## Several models loaded at once

Models too big for the graphics card (gpt-oss, qwen2.5-coder:7b) run fully on the CPU, and small ones run on the GPU, so they don't push each other out of memory. Models stay loaded for 2 hours, so switching between them in the model menu is instant.

- **● Ready** in the model menu means the model is already in memory. Picking a model that isn't loaded starts loading it in the background right away.
- **Unload** frees a model's memory. On this PC, keeping all three big-to-small models loaded makes gpt-oss about half as fast (1.5 instead of 3 tok/s) because Windows starts swapping. Unload qwen2.5-coder:7b when you want gpt-oss at full speed.
- To turn this off and let Ollama place models itself, set `BIG_MODELS_ON_CPU=0`.

## Tips

- **Long PDFs or projects with many files**: raise *Context length* in Settings. Changing it makes Ollama reload the model.
- **Scanned PDFs** have no text layer. Screenshot the pages and upload them as photos instead.
- Environment variables: `MODEL`, `VISION_MODEL`, `PORT`, `OLLAMA_URL`, `KEEP_ALIVE`.
