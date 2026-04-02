import json
import re

RAW_PATH = "answers_rag_raw.txt"
OUT_PATH = "answers_rag.json"

with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

# 1. 外側の { } を除去
content = raw.strip()
if content.startswith("{"):
    content = content[1:]
if content.endswith("}"):
    content = content[:-1]

# 2. qXXX ごとに分割（"q001": から始まる想定）
pattern = r'"(q\d+)"\s*:\s*"'
parts = re.split(pattern, content)

data = {}

# parts = ["", "q001", "text...", "q002", "text...", ...]
for i in range(1, len(parts), 2):
    qid = parts[i]
    text = parts[i + 1]

    # 末尾の ", を除去
    text = text.rstrip().rstrip(",")

    # エスケープ処理（超重要）
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")

    data[qid] = text

# 3. 正しい JSON として保存
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUT_PATH}")

