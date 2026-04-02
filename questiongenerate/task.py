from openai import OpenAI
import json
import re
from corpus import CORPUS_DESCRIPTION

# =========================
# 設定
# =========================
MODEL_NAME = "gpt-4o-mini"      # 厳密再現のため固定
TEMPERATURE = 0.7
MAX_TOKENS = 600

client = OpenAI()

# =========================
# Persona の読み込み
# =========================
with open("output/personas.json", "r", encoding="utf-8") as f:
    personas = json.load(f)

all_tasks = []

# =========================
# Persona ごとに Task 生成
# =========================
for persona in personas:
    prompt = f"""
You are designing evaluation tasks for a system that answers global, synthesis-oriented questions.

Corpus description:
{CORPUS_DESCRIPTION}

Persona:
{persona['persona_text']}

Task:
Generate exactly 5 distinct information-seeking tasks that this persona might want to
accomplish using the corpus.

Guidelines:
- Tasks should require synthesizing information across many documents.
- Tasks should be high-level (analysis, comparison, synthesis).
- Avoid tasks answerable by retrieving a single fact.
- Avoid yes/no tasks.

Output format:
Return a numbered list from 1 to 5.
Each task should be exactly one sentence.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in evaluation task design for information retrieval systems."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    content = response.choices[0].message.content.strip()

    # =========================
    # 出力パース（厳密）
    # =========================
    # 「1. xxx」「2) xxx」などを想定
    tasks = re.findall(
        r'^\s*\d+[\.\)]\s*(.+)',
        content,
        re.MULTILINE
    )

    # 件数チェック（論文再現で最重要）
    if len(tasks) != 5:
        raise ValueError(
            f"[ERROR] Expected exactly 5 tasks for persona {persona['persona_id']}, "
            f"but got {len(tasks)}.\n\nRaw output:\n{content}"
        )

    # 保存
    for i, task_text in enumerate(tasks, start=1):
        all_tasks.append({
            "task_id": f"{persona['persona_id']}_t{i}",
            "persona_id": persona["persona_id"],
            "task_text": task_text.strip(),
            "generation_model": MODEL_NAME,
            "temperature": TEMPERATURE
        })

# =========================
# 出力保存
# =========================
with open("output/tasks.json", "w", encoding="utf-8") as f:
    json.dump(all_tasks, f, indent=2, ensure_ascii=False)

print(f"Saved tasks.json ({len(all_tasks)} tasks total)")
