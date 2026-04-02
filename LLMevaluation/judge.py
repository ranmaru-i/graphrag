import json
import random
from openai import OpenAI
import re

client = OpenAI()

CRITERIA = {
    "comprehensiveness": (
        "Comprehensiveness: How thoroughly the answer covers the important aspects "
        "of the topic, including major themes and relevant factors across the corpus."
    ),
    "diversity": (
        "Diversity: How well the answer reflects a range of perspectives, viewpoints, "
        "or themes, rather than focusing narrowly on a single angle."
    ),
    "empowerment": (
        "Empowerment: How well the answer helps the reader build an understanding "
        "of the topic and feel informed and capable of reasoning further about it."
    ),
    "directness": (
        "Directness: How directly and clearly the answer addresses the question "
        "without unnecessary digressions or irrelevant content."
    ),
}

def build_prompt(question, a1, a2, criterion):
    return f"""
Question:
{question}

Answer 1:
{a1}

Answer 2:
{a2}

Evaluation criterion:
{criterion}

Which answer is better? Answer 1 / Answer 2 / Tie
Explain briefly.
"""

def parse_winner(output, ans1, ans2):
    text = output.lower()

    if re.search(r"answer\s*1", text):
        return ans1["method"]
    elif re.search(r"answer\s*2", text):
        return ans2["method"]
    else:
        return "Tie"


def judge_once(question, ansA, ansB, criterion):
    if random.random() < 0.5:
        a1, a2 = ansA, ansB
        flipped = False
    else:
        a1, a2 = ansB, ansA
        flipped = True

    prompt = build_prompt(question, a1["text"], a2["text"], criterion)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fair evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=300
    )

    out = resp.choices[0].message.content.strip()

    winner = parse_winner(out, a1, a2)

    return winner, out, flipped

def main():
    with open("data/questions.json") as f:
        questions = json.load(f)

    with open("data/re_answers_your_new_10000_5.json") as f:
        your_answers = json.load(f)

    with open("data/answers_rag.json") as f:
        base_answers = json.load(f)

    results = []

    for q in questions:
        qid = q.get("question_id")
        question_text = q["question_text"]
        
        for cname, ctext in CRITERIA.items():
            for rep in range(5):
                winner, raw, flipped = judge_once(
                    question_text,
                    {"method": "YourMethod", "text": your_answers[qid]},
                    {"method": "BasicRAG", "text": base_answers[qid]},
                    ctext
                )
                results.append({
                    "question_id": qid,
                    "criterion": cname,
                    "replicate": rep,
                    "winner": winner,
                    "raw_output": raw,
                    "flipped": flipped
                })

    with open("results/re_judge_results_basicRAG_new_10000_5.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
