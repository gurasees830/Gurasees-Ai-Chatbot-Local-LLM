# Gurasees Ollama: a private coding chatbot you can train

A coding problem solver that runs 100% on your own computer. No API keys, no data leaves your machine.

- **gurasees-coder model** (`Modelfile.gurasees`): qwen2.5-coder tuned for problem solving. It restates the problem, explains the approach with time/space complexity, writes runnable code, then traces an example and lists edge cases.
- **Modes** in the composer: Solve, Debug, Explain, Optimize, Review, Tests. Each adds focused instructions on the server.
- **Fast:** the server preloads the model at startup and keeps it in GPU memory for 2 hours (`KEEP_ALIVE`), so replies don't wait 10-30s for a reload. Old history is trimmed to fit the context window, and each reply shows its tokens/sec.

```
Browser UI  ->  FastAPI server (server.py)  ->  Ollama (runs the model)
                    |
                    +-> data/feedback.jsonl  ->  training/  ->  your fine-tuned model
```

## Quick start (Windows)

1. **Install Ollama** from https://ollama.com/download and run the installer. It starts in the background automatically (look for the llama icon in the system tray).
2. **Install Python 3.10+** from https://python.org. On the first installer screen, tick **"Add python.exe to PATH"**.
3. **Unzip this project**, open the folder, and **double-click `start.bat`**.

On the first run it creates a Python environment, installs dependencies, downloads the 3B coding model (about 2 GB, once), and opens http://localhost:8000. Later runs start in seconds. Keep the black window open while you chat; close it to stop the app.

Prefer typing commands? Open PowerShell in the project folder:
```powershell
ollama pull qwen2.5-coder:3b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```
Then open http://localhost:8000.

macOS/Linux: install Ollama (`curl -fsSL https://ollama.com/install.sh | sh` on Linux), then use the same commands with `source .venv/bin/activate`.

### Choosing a model
Match the model to your VRAM/RAM. For a GPU with 4 GB of VRAM, `qwen2.5-coder:3b` fits fully on the GPU and is fast.

| Your machine            | Command                            |
|-------------------------|------------------------------------|
| 4 GB GPU / 8 GB RAM     | `ollama pull qwen2.5-coder:3b`     |
| 16 GB RAM, bigger/slower| `ollama pull qwen2.5-coder:7b`     |
| 32 GB RAM or 16 GB GPU  | `ollama pull qwen2.5-coder:14b`    |

Any model installed in Ollama appears in the UI's model dropdown. Check GPU use with `ollama ps` (look for "100% GPU").
To change the default: `MODEL=qwen2.5-coder:7b python server.py` (PowerShell: `$env:MODEL="qwen2.5-coder:7b"; python server.py`).

## 2. How it works
Your browser talks to `server.py`, which streams replies from Ollama at http://localhost:11434. Nothing leaves your computer.

## 3. "Training" your chatbot

There are three levels, from easiest to most powerful.

### Level 1: System prompt (instant)
Gurasees' problem-solving prompt is built in (`BASE_SYSTEM` in `server.py` and `Modelfile.gurasees`). Add your own rules under **Settings → Custom instructions** ("always add type hints"). `start.bat` builds the model automatically; to rebuild it after editing the Modelfile:
```bash
ollama create gurasees-coder -f Modelfile.gurasees
```

### Level 2: Collect your own training data (while you use it)
In the chat, click **👍 Good** on answers you like, or **✎ Fix & save** to rewrite an answer into the ideal version. Each one is appended to `data/feedback.jsonl`. The sidebar counter shows how many you have.

Convert them into a training file:
```bash
cd training
python prepare_data.py      # writes train.jsonl (also includes seed_examples.jsonl)
```
You'll want roughly **100 to 1000** good examples. Quality beats quantity.

`training/seed_examples.jsonl` ships with 15 hand-written examples (Two Sum, sliding window, linked lists, debugging, optimization, code review, tests) in Gurasees' answer format.

### Level 3: Fine-tune the model (LoRA)
This needs an NVIDIA GPU (8 GB+ VRAM for the 3B model, 16 GB for 7B), so a 4 GB card like the GTX 1650 should use Colab. **No GPU?** Use a free Google Colab T4: upload `training/finetune.py` and `train.jsonl`, run `pip install unsloth`, then `python finetune.py`, and download the resulting `.gguf`.

```bash
pip install unsloth
cd training
python finetune.py          # trains, then exports ./gguf_out/*.gguf
```

Then load the result into Ollama:
1. Open `Modelfile.finetuned` and make sure the `FROM` line points at the `.gguf` file that was produced.
2. `ollama create my-coder-ft -f Modelfile.finetuned`
3. Refresh the UI and pick `my-coder-ft`.

If you changed `BASE_MODEL` in `finetune.py`, change the `TEMPLATE` in `Modelfile.finetuned` to match that model's chat format.

## Project layout
```
server.py              FastAPI backend (streams from Ollama, saves feedback)
static/index.html      Chat UI (single file, works offline, built-in syntax highlighting)
Modelfile.gurasees     The gurasees-coder model (problem-solving prompt + tuned parameters)
Modelfile.custom       Minimal example of a custom Ollama model
Modelfile.finetuned    Ollama model from your fine-tuned GGUF
data/feedback.jsonl    Created automatically: answers you rated or fixed
training/prepare_data.py   feedback -> train.jsonl
training/finetune.py       LoRA fine-tune + GGUF export
training/seed_examples.jsonl   15 coding-problem examples in Gurasees' answer format
```

## Troubleshooting
- **"Can't reach Ollama":** start it (`ollama serve`) and refresh. Check http://localhost:11434 shows "Ollama is running".
- **Slow first reply:** the model is loading into GPU memory. The server preloads it at startup; after that, expect ~45-50 tok/s on a GTX 1650.
- **Slow replies:** use a smaller model (3b), or a smaller context window in Settings. Run `ollama ps` and check it says "100% GPU".
- **`ollama` not recognized:** the Windows installer doesn't always add it to PATH. Use `"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"` instead (`start.bat` already does).
- **Out of memory:** smaller model, or close other apps.
- **Fine-tune fails on import/version errors:** Unsloth and TRL change often; follow the install notes at https://docs.unsloth.ai, or run it in their Colab notebooks.
- **Port in use:** `PORT=8080 python server.py`
