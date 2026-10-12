# Gurasees's Deep Learning Local LLM Chatbot

A private chatbot that runs open models on your own computer with [Ollama](https://ollama.com): no API keys, no cloud.

| Folder | What it is |
|---|---|
| [`gpt-oss-chat/`](gpt-oss-chat) | **The main app.** Chat with gpt-oss, Qwen coder, Gemma and any other Ollama model. Reads documents and web links, understands photos and screenshots, creates images, packs code into a downloadable ZIP, and has Projects with memory. |
| [`local-llm-coder/`](local-llm-coder) | A smaller coding chatbot plus the `gurasees-coder` model definition and fine-tuning scripts. |
| [`HOSTING.md`](HOSTING.md) | How to share the chatbot online (free tunnel from your PC, or the website on Netlify / Vercel / Cloudflare Pages). |

## Quick start (Windows)

1. Install [Ollama](https://ollama.com/download) and [Python 3.10+](https://python.org) (tick "Add python.exe to PATH").
2. Double-click `gpt-oss-chat/start.bat`. It sets everything up, downloads the models, and opens http://localhost:8001.
3. To build the custom coding model: `cd local-llm-coder`, then `ollama create gurasees-coder -f Modelfile.gurasees`.

Details: [gpt-oss-chat/README.md](gpt-oss-chat/README.md). Model downloads and the `.venv` folders are not stored in this repository; `start.bat` recreates them.
