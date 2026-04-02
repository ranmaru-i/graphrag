from openai import OpenAI
import json
import re
from corpus import CORPUS_DESCRIPTION

# =========================
# 設定（論文再現用に固定）
# =========================
MODEL_NAME = "gpt-4o-mini"      # 厳密再現のため固定
TEMPERATURE = 0.7
MAX_TOKENS = 800

client = OpenAI()

# =========================
# Persona 生成プロンプト
# =========================
persona_prompt = f"""
You are helping design an evaluation for a system that answers high-level questions
requiring global understanding of a large text corpus.

The corpus is described as follows:
{CORPUS_DESCRIPTION}

Task:
Generate exactly 5 distinct user personas who would reasonably use this corpus
to gain high-level understanding, insights, or synthesis.

Guidelines:
- Each persona should represent a different perspective, role, or goal.
- Personas should be generic (not real individuals).
- Personas should plausibly require synthesizing information across the entire corpus.

Output format:
Return a numbered list from 1 to 5.
Each persona should be described in 2–3 sentences.
"""

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are an expert in evaluation design for information retrieval systems."
        },
        {
            "role": "user",
            "content": persona_prompt
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
parsed_personas = re.findall(
    r'^\s*\d+[\.\)]\s*(.+)',
    content,
    re.MULTILINE
)

# 件数チェック（論文再現で最重要）
if len(parsed_personas) != 5:
    raise ValueError(
        f"[ERROR] Expected exactly 5 personas, but got {len(parsed_personas)}.\n\n"
        f"Raw output:\n{content}"
    )

# =========================
# 保存
# =========================
personas = []
for i, persona_text in enumerate(parsed_personas, start=1):
    personas.append({
        "persona_id": f"p{i}",
        "persona_text": persona_text.strip(),
        "generation_model": MODEL_NAME,
        "temperature": TEMPERATURE
    })

with open("output/personas.json", "w", encoding="utf-8") as f:
    json.dump(personas, f, indent=2, ensure_ascii=False)

print("Saved personas.json (5 personas)")
