# graphrag/index/operations/extract_graph/tfidf_graph.py
import asyncio
import pandas as pd
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import partial

async def extract_graph_tfidf(
    text_units: pd.DataFrame,
    text_column: str = "text",
    id_column: str = "id",
    threshold: float = 0.03,
    min_df: int | float = 1,
    max_df: float = 0.5,
    max_features: int | None = 1000000,
):
    """
    文書丸ごと1ノードでTF-IDFハイパーエッジグラフを作成。
    entities は作らず、relationships のみ生成。
    """

    def _sync_extract(docs, ids):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        X_csc = X.tocsc()

        relationships = []

        for term_idx, term in enumerate(vocab):
            col = X_csc.getcol(term_idx)
            if col.nnz == 0:
                continue
            indices = col.indices
            data = col.data
            sel_mask = data >= threshold
            if sel_mask.sum() > 0:
                source_node = term
                for doc_idx, score in zip(indices[sel_mask], data[sel_mask]):
                    doc_id = ids[int(doc_idx)]
                    relationships.append({
                        "source": source_node,        # ハイパーエッジID
                        "target": doc_id,         # 文書ID
                        "weight": float(score),
                        "description": f"TF-IDF >= {threshold} ({term})",
                        "text_unit_ids": [doc_id],
                    })
        
        relationships_df = pd.DataFrame(relationships)

        return pd.DataFrame(), relationships_df, vectorizer

    docs = text_units[text_column].astype(str).tolist()
    ids = text_units[id_column].astype(str).tolist()

    return await asyncio.to_thread(partial(_sync_extract, docs, ids))