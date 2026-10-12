"""Turn the answers you rated in the UI into a fine-tuning dataset.

Reads:  ../data/feedback.jsonl   (saved by the chat UI)
        seed_examples.jsonl      (hand-written examples, optional)
Writes: train.jsonl              (one {"messages": [...]} per line)

Run:  python prepare_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
FEEDBACK = HERE.parent / "data" / "feedback.jsonl"
SEED = HERE / "seed_examples.jsonl"
OUT = HERE / "train.jsonl"

# Keep in sync with BASE_SYSTEM in server.py so the fine-tuned model learns the same persona.
SYSTEM = ("You are Gurasees, an expert software engineer and competitive programmer who solves coding problems. "
          "Restate the goal, explain the approach with its time and space complexity, give complete runnable code "
          "in a fenced code block with the language tag, then trace an example and list edge cases. "
          "When debugging, find the root cause first, then show the minimal fix.")


def load(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def valid(messages):
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return False
    return all(m.get("content", "").strip() for m in messages)


def main():
    rows, seen = [], set()
    for rec in load(SEED) + load(FEEDBACK):
        if rec.get("rating", "up") != "up":
            continue
        msgs = [m for m in rec["messages"] if m["role"] in ("user", "assistant")]
        if not valid(msgs):
            continue
        key = json.dumps(msgs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"messages": [{"role": "system", "content": SYSTEM}] + msgs})

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {OUT}")
    if len(rows) < 50:
        print("Tip: aim for 100-1000 high-quality examples. With fewer than ~50 the model "
              "barely changes; use the system prompt in Modelfile.custom for light customization instead.")


if __name__ == "__main__":
    main()
