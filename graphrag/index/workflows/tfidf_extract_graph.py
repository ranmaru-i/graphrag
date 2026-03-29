# graphrag/index/workflows/tfidf_extract_graph.py

import logging
import pandas as pd
import pickle

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.extract_graph.tfidf_graph import extract_graph_tfidf as tfidf_graph
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)

async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Run TF-IDF based graph extraction."""
    logger.info("Workflow started: tfidf_extract_graph")
    text_units = await load_table_from_storage("text_units", context.output_storage)

    # --- (2) TF-IDFグラフ抽出 ---
    # --- tfidf_extract_graph を呼び出す ---
    relationships_df, vectorizer_df = await tfidf_extract_graph(
        text_units=text_units,
        config=config
    )

    df_dict = build_df_dict_from_relationships(relationships_df)
    N_docs = get_N_docs(text_units)

    await write_table_to_storage(
        pd.DataFrame(
            [{"term": k, "df": v} for k, v in df_dict.items()]
        ),
        "df_stats",
        context.output_storage,
    )

    await write_table_to_storage(
        pd.DataFrame([{"N_docs": N_docs}]),
        "corpus_stats",
        context.output_storage,
    )

    await write_table_to_storage(
        vectorizer_df,
        "tfidf_vectorizer",
        context.output_storage,
    )

    # --- (3) entities_df を relationships_df から自動生成 ---
    unique_nodes = pd.unique(
        relationships_df[["source", "target"]].values.ravel("K")
    )

    entities_df = pd.DataFrame({
        "title": unique_nodes,
        "type": "entity",  # TF-IDF単語 or 文書をノードとして扱う
        "description": "",
        "document_ids": [[] for _ in unique_nodes],
        "text_unit_ids": [[] for _ in unique_nodes],
        "frequency": 0.0,
    })

    # 保存
    await write_table_to_storage(entities_df, "entities", context.output_storage)
    await write_table_to_storage(relationships_df, "relationships", context.output_storage)

    ensure_empty_community_files()

    logger.info("Workflow completed: tfidf_extract_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities_df,
            "relationships": relationships_df,
        }
    )


async def tfidf_extract_graph(
    text_units: pd.DataFrame,
    config: GraphRagConfig,
) -> pd.DataFrame:
    """TF-IDF で文書群間のハイパーエッジ（relationships）を生成する部分。"""
    logger.info(">>> Running TF-IDF graph extraction <<<")

    _, relationships_df, vectorizer = await tfidf_graph(
        text_units=text_units,
        threshold=getattr(config.extract_graph_tfidf, "threshold", 0.1),
        min_df=getattr(config.extract_graph_tfidf, "min_df", 1),
        max_df=getattr(config.extract_graph_tfidf, "max_df", 0.8),
        max_features=getattr(config.extract_graph_tfidf, "max_features", None),
    )

    vectorizer_bytes = pickle.dumps(vectorizer)

    vectorizer_df = pd.DataFrame([
        {"name": "tfidf_vectorizer", "blob": vectorizer_bytes}
    ])

    return relationships_df, vectorizer_df

def build_df_dict_from_relationships(relationships_df: pd.DataFrame) -> dict[str, int]:
    """
    df_dict[term] = その term が出現する document 数
    """
    df = (
        relationships_df
        .drop_duplicates(subset=["source", "target"])
        .groupby("source")["target"]
        .nunique()
    )
    return df.to_dict()

def get_N_docs(text_units: pd.DataFrame) -> int:
    return text_units["id"].nunique()

import os
def ensure_empty_community_files(output_dir="./output"):

    os.makedirs(output_dir, exist_ok=True)

    # --- (1) communities.parquet の列構造 ---
    communities_columns = [
        "id",
        "human_readable_id",
        "community",
        "level",
        "parent",
        "children",
        "title",
        "entity_ids",
        "relationship_ids",
        "text_unit_ids",
        "period",
        "size",
    ]

    # --- (2) community_reports.parquet の列構造 ---
    community_reports_columns = [
        "id",
        "human_readable_id",
        "community",
        "level",
        "parent",
        "children",
        "title",
        "summary",
        "full_content",
        "rank",
        "rating_explanation",
        "findings",
        "full_content_json",
        "period",
        "size",
    ]

    os.makedirs(output_dir, exist_ok=True)
    # --- (3) ファイル作成関数 ---
    def create_if_missing(filename, columns):
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"🟡 {filename} が存在しないため、新規作成します。")
            pd.DataFrame(columns=columns).to_parquet(filepath, index=False)
        else:
            try:
                df = pd.read_parquet(filepath)
                print(f"✅ {filename} は既に存在しています。行数: {len(df)}")
            except Exception as e:
                print(f"⚠️ {filename} の読み込みに失敗しました: {e}")
                print(f"→ 空のファイルを再作成します。")
                pd.DataFrame(columns=columns).to_parquet(filepath, index=False)

    # --- (4) チェック＆作成 ---
    create_if_missing("communities.parquet", communities_columns)
    create_if_missing("community_reports.parquet", community_reports_columns)

    print("🎉 チェック完了！")


