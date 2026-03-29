# graphrag/query/context_builder/node_only.py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from typing import Any
import pandas as pd

import json
from datetime import datetime
from pathlib import Path

import tiktoken

import re

from graphrag.query.context_builder.builders import LocalContextBuilder, ContextBuilderResult


def _log_llm_input(query: str, context: str, records_df: pd.DataFrame):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / "llm_inputs.jsonl"

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "context": context,
        "documents": records_df.to_dict(orient="records"),
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_document_text(json_text: str) -> str:
    """
    JSON文字列を LLM 向けの自然文テキストに変換
    """
    obj = json.loads(json_text)

    title = obj.get("title", "").strip()
    body = obj.get("body", "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts)


def truncate_to_tokens(text: str, enc, max_tokens: int) -> str:
    """
    encでトークン化したとき max_tokens に収まるように末尾を切る
    """
    if max_tokens <= 0:
        return ""
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def extract_front_middle_back(
    text: str,
    enc,
    max_tokens: int,
    ratios=(0.25, 0.5, 0.25),
) -> str:
    """
    文書を前・中・後から抜粋して max_tokens に収める
    """
    if max_tokens <= 0:
        return ""

    tokens = enc.encode(text)
    n = len(tokens)

    if n <= max_tokens:
        return text

    f_ratio, m_ratio, b_ratio = ratios
    f_n = int(max_tokens * f_ratio)
    m_n = int(max_tokens * m_ratio)
    b_n = max_tokens - f_n - m_n

    front = tokens[:f_n]

    mid_start = max(0, (n // 2) - (m_n // 2))
    middle = tokens[mid_start : mid_start + m_n]

    back = tokens[-b_n:] if b_n > 0 else []

    selected = front + middle + back
    return enc.decode(selected)


def extract_head(
    text: str,
    enc,
    max_tokens: int,
) -> str:
    """
    文書冒頭のみを max_tokens まで抜粋（preview 用）
    """
    return truncate_to_tokens(text, enc, max_tokens)

def split_sentences(text: str) -> list[str]:
    """
    雑に文章分割（英語ニュース想定）。
    - . ? ! の後の空白で分割
    - 略語や小数点での誤分割はある程度許容（preview用途なので割り切り）
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def build_preview_snippet(rendered: str, enc, max_tokens: int) -> str:
    """
    Previewを「タイトル＋冒頭2文＋転換/因果/出来事っぽい文1文」にする。
    収まらなければ最後にトークンで切る。
    rendered は render_document_text の出力（Title/Body入りの自然文）を想定。
    """
    if max_tokens <= 0:
        return ""

    # Title/Bodyの抽出（render_document_textの形式に依存）
    title = ""
    body = rendered

    m = re.search(r"^Title:\s*(.*?)(?:\n\n|$)", rendered, flags=re.MULTILINE)
    if m:
        title = m.group(1).strip()

    m2 = re.search(r"\n\nBody:\n(.*)$", rendered, flags=re.DOTALL)
    if m2:
        body = m2.group(1).strip()

    # 本文を文分割
    sents = split_sentences(body)
    lead = sents[:2]

    # “転換/因果/出来事”っぽいシグナル語を含む文を探す
    signal_patterns = [
        r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"\byet\b",
        r"\btherefore\b", r"\bthus\b", r"\bas a result\b", r"\bbecause\b",
        r"\bdue to\b", r"\bleading to\b", r"\bprompted\b", r"\bsparked\b",
        r"\bafter\b", r"\bwhen\b", r"\bonce\b",
        r"\bannounced\b", r"\bapproved\b", r"\bruled\b", r"\bpassed\b",
        r"\bcharged\b", r"\barrested\b", r"\bsued\b",
        r"\bsaid\b", r"\bstated\b", r"\baccording to\b",
    ]
    signal_re = re.compile("|".join(signal_patterns), flags=re.IGNORECASE)

    signal = ""
    for s in sents[2:]:
        if signal_re.search(s):
            signal = s
            break
    if not signal and len(sents) >= 3:
        signal = sents[2]

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if lead:
        parts.append("Lead: " + " ".join(lead))
    if signal:
        parts.append("Signal: " + signal)

    preview = "\n".join(parts).strip()

    # 予算内に収める（最後に安全にトークン切り）
    return truncate_to_tokens(preview, enc, max_tokens)


class NodeOnlyContextBuilder(LocalContextBuilder):
    """
    ContextBuilder that only includes top-N document nodes (no communities,
    no relationships, no covariates). Expects to receive via kwargs:
      - top_nodes_df: pd.DataFrame with at least columns ['node','score'] (ordered desc)
      - document_text_dict: dict mapping node_id -> document text (str)
      - max_context_tokens (optional): int
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        default_max_context_tokens: int = 10000,
    ):
        self.enc = tiktoken.get_encoding("o200k_base")
        self.default_max_context_tokens = default_max_context_tokens

    def build_context(
        self,
        query: str,
        conversation_history: Any | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        assert "top_nodes_df" in kwargs, "top_nodes_df NOT passed"

        top_nodes_df: pd.DataFrame | None = kwargs.get("top_nodes_df")
        document_text_dict: dict | None = kwargs.get("document_text_dict", {})
        max_context_tokens: int = kwargs.get("max_context_tokens", self.default_max_context_tokens)

        if top_nodes_df is None or top_nodes_df.empty:
            return ContextBuilderResult(
                context_chunks="",
                context_records={},
            )

        # 並び順だけ保持（スコアは使わない）
        if "score" in top_nodes_df.columns:
            top = top_nodes_df.sort_values("score", ascending=False)
        else:
            top = top_nodes_df

        # 上位Nのうち、実際に本文が存在するものだけを対象にする
        node_ids = []
        for _, row in top.iterrows():
            node_id = str(row.get("node") if "node" in row else row.iloc[0])
            if document_text_dict.get(node_id) is not None:
                node_ids.append(node_id)

        if not node_ids:
            return ContextBuilderResult(
                context_chunks="",
                context_records={},
            )

        n_docs = len(node_ids)

        # ----------------------------
        # ここが重要：本当に均等割する
        #  - まず固定ヘッダを入れる
        #  - 残りを「文書ごとの予算」に均等分割
        #  - Preview/Mainは各文書予算の中で分割
        # ----------------------------
        #ここからはpreviewあり
        PREVIEW_RATIO = 0.20
        PREVIEW_MIN_TOKENS = 260  # ここが効く（少なくともTitle+Lead+Signalが見える）

        header = "-----Documents-----\n"
        preview_header = "-----Document Previews-----\n"
        main_header = "-----Document Details-----\n"

        context_chunks = header + preview_header
        base_tokens = len(self.enc.encode(context_chunks))

        # main_headerも必ず入るので、先に確保しておく
        main_header_tokens = len(self.enc.encode(main_header))

        available = max_context_tokens - base_tokens - main_header_tokens
        if available <= 0:
            # そもそもヘッダだけで上限を超えるような設定は破綻なので最小返し
            context_chunks = truncate_to_tokens(header, self.enc, max_context_tokens)
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records={"entities": pd.DataFrame(columns=["node", "text", "score"])},
                llm_calls=0,
                prompt_tokens=len(self.enc.encode(context_chunks)),
                output_tokens=0,
            )

        # 文書ごとの総予算（preview+main+各ブロックヘッダ込み）
        per_doc_budget = max(1, available // n_docs)

        # Preview/Mainの比率割り当て（各文書ごとに同じ）
        preview_budget = max(PREVIEW_MIN_TOKENS, int(per_doc_budget * PREVIEW_RATIO))
        main_budget = max(1, per_doc_budget - preview_budget)

        # ----------------------------
        # 第1段：Preview（各文書の予算内で必ず入れる）
        # ----------------------------
        current_tokens = len(self.enc.encode(context_chunks))

        for node_id in node_ids:
            raw = document_text_dict.get(node_id)
            rendered = render_document_text(raw)

            # ブロックヘッダ分を先に引く（これも均等割の一部）
            block_header = f"[Preview] {node_id}\n"
            header_t = len(self.enc.encode(block_header))
            body_budget = max(0, preview_budget - header_t)

            preview_text = build_preview_snippet(rendered, enc=self.enc, max_tokens=body_budget)

            SEPARATOR = "\n\n"

            block = f"{block_header}{preview_text}"
            # 末尾の区切り(SEPARATOR)分を先に確保しておき、本文だけtruncateする
            sep_t = len(self.enc.encode(SEPARATOR))
            block = truncate_to_tokens(block, self.enc, max(0, preview_budget - sep_t))
            block = block + SEPARATOR

            context_chunks += block
            current_tokens += len(self.enc.encode(block))

        # ----------------------------
        # 第2段：Main（各文書の予算内で必ず入れる）
        # ----------------------------
        context_chunks += main_header
        current_tokens = len(self.enc.encode(context_chunks))

        records = []

        DOC_SEP = "\n\n"  # ← ここが「絶対に入る区切り」
        doc_sep_t = len(self.enc.encode(DOC_SEP))

        for node_id in node_ids:
            raw = document_text_dict.get(node_id)
            rendered = render_document_text(raw)

            block_header = f"-----Document: {node_id}-----\n"
            header_t = len(self.enc.encode(block_header))

            # 重要：ヘッダ + セパレータは必ず入れる前提で、本文予算を決める
            # main_budget が小さすぎる場合でも「区切り」は入るようにする
            body_budget = max(0, main_budget - header_t - doc_sep_t)

            excerpt = extract_front_middle_back(rendered, enc=self.enc, max_tokens=body_budget)

            # 重要：truncate は excerpt 側だけ（ヘッダ/区切りは壊さない）
            excerpt = truncate_to_tokens(excerpt, self.enc, body_budget)

            block = f"{block_header}{excerpt}{DOC_SEP}"

            # 念のため：最終保険（ここで壊れても最低限区切りが残りやすい）
            block = truncate_to_tokens(block, self.enc, main_budget)

            context_chunks += block
            current_tokens += len(self.enc.encode(block))

            records.append({
                "node": node_id,
                "text": excerpt,
                "score": float(top.loc[top["node"] == node_id, "score"].iloc[0]) if "score" in top.columns and (top["node"] == node_id).any() else None,
            })
        #ここまで

        """ここからはPreviewなし
        # ヘッダ（Previewは無し）
        header = "-----Documents-----\n"
        main_header = "-----Document Details-----\n"

        # 各doc区切り（連結防止）
        DOC_SEP = "\n\n"

        # まずヘッダを確定
        context_chunks = header + main_header
        base_tokens = len(self.enc.encode(context_chunks))

        available = max_context_tokens - base_tokens
        if available <= 0:
            context_chunks = truncate_to_tokens(header, self.enc, max_context_tokens)
            return ContextBuilderResult(
                context_chunks=context_chunks,
                context_records={"entities": pd.DataFrame(columns=["node", "text", "score"])},
                llm_calls=0,
                prompt_tokens=len(self.enc.encode(context_chunks)),
                output_tokens=0,
            )

        # docごとの予算（均等割）
        per_doc_budget = max(1, available // n_docs)

        records = []

        # Mainのみ詰める
        for node_id in node_ids:
            raw = document_text_dict.get(node_id)
            rendered = render_document_text(raw)

            block_header = f"-----Document: {node_id}-----\n"
            header_t = len(self.enc.encode(block_header))
            sep_t = len(self.enc.encode(DOC_SEP))

            # 「ヘッダ + 区切り」は必ず残す。本文だけをこの予算に収める
            max_excerpt_tokens = max(0, per_doc_budget - header_t - sep_t)

            excerpt = extract_front_middle_back(rendered, enc=self.enc, max_tokens=max_excerpt_tokens)
            excerpt = truncate_to_tokens(excerpt, self.enc, max_excerpt_tokens)

            block = f"{block_header}{excerpt}{DOC_SEP}"

            # 念のため（基本ここでは超えない設計だが保険）
            block = truncate_to_tokens(block, self.enc, per_doc_budget)

            context_chunks += block

            records.append(
                {
                    "node": node_id,
                    "text": excerpt,
                    "score": float(top.loc[top["node"] == node_id, "score"].iloc[0])
                    if "score" in top.columns and (top["node"] == node_id).any()
                    else None,
                }
            )
        ここまで"""

        # 最後に、絶対に max_context_tokens を超えないように保険
        context_chunks = truncate_to_tokens(context_chunks, self.enc, max_context_tokens)
        current_tokens = len(self.enc.encode(context_chunks))



        df = pd.DataFrame(records) if records else pd.DataFrame(columns=["node", "text", "score"])

        _log_llm_input(
            query=query,
            context=context_chunks,
            records_df=df,
        )

        return ContextBuilderResult(
            context_chunks=context_chunks,
            context_records={"entities": df},
            llm_calls=0,
            prompt_tokens=current_tokens,
            output_tokens=0,
        )
