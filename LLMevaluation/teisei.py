import json
from pathlib import Path

input_path = Path("data/answers_your_new_10000_5.json")
output_path = Path("data/re_answers_your_new_10000_5.json")

MARKER = "#docs: 609\n"  # json.loadすると \n は「実際の改行」になります

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)  # ← ここが重要（行ごとじゃない）

for qid, text in data.items():
    if isinstance(text, str) and MARKER in text:
        data[qid] = text.split(MARKER, 1)[1]  # MARKER以前（MARKER含む）を削除

with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"wrote: {output_path}")


