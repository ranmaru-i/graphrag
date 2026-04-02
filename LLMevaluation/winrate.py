import json
from collections import defaultdict
from statistics import mean

# =========================
# 設定
# =========================

JSONL_PATH = "results/re_judge_results_basicRAG_new_10000.jsonl"  # ← あなたの jsonl ファイル
TARGET_METHOD = "YourMethod"
OPPONENT_METHOD = "BasicRAG"

SCORE_MAP = {
    TARGET_METHOD: 100,
    OPPONENT_METHOD: 0,
    "Tie": 50
}

# =========================
# 1. jsonl 読み込み
# =========================

records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# =========================
# 2. (criterion, question_id) ごとに replicate を集約
# =========================

# structure:
# scores[criterion][question_id] = [100, 50, 0, ...]
scores = defaultdict(lambda: defaultdict(list))

for r in records:
    criterion = r["criterion"]
    question_id = r["question_id"]
    winner = r["winner"]

    if winner not in SCORE_MAP:
        raise ValueError(f"Unknown winner value: {winner}")

    score = SCORE_MAP[winner]
    scores[criterion][question_id].append(score)

# =========================
# 3. replicate 平均 → 質問スコア
# =========================

# question_scores[criterion][question_id] = mean score
question_scores = defaultdict(dict)

for criterion, questions in scores.items():
    for qid, replicate_scores in questions.items():
        question_scores[criterion][qid] = mean(replicate_scores)

# =========================
# 4. 質問平均 → criterion 勝率
# =========================

criterion_results = {}

for criterion, qscores in question_scores.items():
    avg_score = mean(qscores.values())          # 0–100
    win_rate = avg_score / 100.0                # 0–1
    criterion_results[criterion] = {
        "average_score": avg_score,
        "win_rate": win_rate,
        "num_questions": len(qscores)
    }

# =========================
# 5. 結果表示
# =========================

print("=== YourMethod Win Rates (Appendix F compliant) ===\n")

for criterion, result in sorted(criterion_results.items()):
    print(f"[{criterion}]")
    print(f"  Questions     : {result['num_questions']}")
    print(f"  Avg score     : {result['average_score']:.2f}")
    print(f"  Win rate (%)  : {result['win_rate'] * 100:.2f}")
    print()
