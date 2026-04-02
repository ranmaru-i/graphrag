from openai import OpenAI
import json
import re
from corpus import CORPUS_DESCRIPTION

# =========================
# 設定（論文再現用に固定）
# =========================
MODEL_NAME = "gpt-4o-mini"      # 厳密再現のため固定
TEMPERATURE = 0.7
MAX_TOKENS = 700

client = OpenAI()

# =========================
# 入力データ読み込み
# =========================
with open("output/personas.json", "r", encoding="utf-8") as f:
    personas = {p["persona_id"]: p["persona_text"] for p in json.load(f)}

with open("output/tasks.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

questions = []
question_counter = 1

# =========================
# Task ごとに Question 生成
# =========================
for task in tasks:
    persona_id = task["persona_id"]
    persona_text = personas[persona_id]
    task_text = task["task_text"]

    prompt = f"""
You are generating evaluation questions for a system designed to answer
global sensemaking queries over a large text corpus.

Corpus description:
{CORPUS_DESCRIPTION}

Persona:
{persona_text}

Task:
{task_text}

Task:
Generate exactly 5 distinct questions that this persona might ask in order to
accomplish the task using the corpus.

Guidelines:
- Questions must require synthesizing information across the corpus as a whole.
- Questions should not be answerable by retrieving a single passage or document.
- Avoid factoid questions (e.g., asking for a specific name, date, or quote).
- Avoid yes/no questions.
- Questions should invite analysis, comparison, or synthesis.

Output format:
Return a numbered list from 1 to 5.
Each item should be a single, well-formed question.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at designing evaluation questions for information retrieval systems."
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
    parsed_questions = re.findall(
        r'^\s*\d+[\.\)]\s*(.+)',
        content,
        re.MULTILINE
    )

    # 件数チェック（論文再現で最重要）
    if len(parsed_questions) != 5:
        raise ValueError(
            f"[ERROR] Expected exactly 5 questions for task {task['task_id']}, "
            f"but got {len(parsed_questions)}.\n\nRaw output:\n{content}"
        )

    # 保存
    for q_text in parsed_questions:
        questions.append({
            "question_id": f"q{question_counter:03}",
            "persona_id": persona_id,
            "task_id": task["task_id"],
            "question_text": q_text.strip(),
            "generation_model": MODEL_NAME,
            "temperature": TEMPERATURE
        })
        question_counter += 1

# =========================
# 件数チェック（最終）
# =========================
EXPECTED_TOTAL = len(tasks) * 5  # 25 tasks × 5 questions = 125

if len(questions) != EXPECTED_TOTAL:
    raise ValueError(
        f"[ERROR] Expected {EXPECTED_TOTAL} questions in total, "
        f"but got {len(questions)}."
    )

# =========================
# 出力保存
# =========================
with open("output/questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"Saved questions.json ({len(questions)} questions total)")

