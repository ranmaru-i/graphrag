# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""TF-IDF ワークフロー用のコミュニティレポート生成モジュール."""

import logging

import pandas as pd

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.extract_graph.tfidf_graph import extract_graph_tfidf as tfidf_graph
from graphrag.index.operations.summarize_communities.explode_communities import explode_communities
from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
    build_level_context,
    build_local_context,
)
from graphrag.index.operations.summarize_communities.summarize_communities import summarize_communities
from graphrag.index.operations.finalize_community_reports import finalize_community_reports
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """TF-IDF に基づいたコミュニティレポート生成ワークフロー."""
    logger.info("Workflow started: create_community_reports_tfidf")

    # 前段の text_units を取得
    text_units = await load_table_from_storage("text_units", context.output_storage)
    communities = await load_table_from_storage("communities", context.output_storage)

    if text_units is None or text_units.empty:
        raise ValueError("No text_units found for TF-IDF community reports")

    # --- document_ids / n_tokens / title を補完 ---
    if "document_ids" not in text_units.columns:
        text_units["document_ids"] = text_units["id"].apply(lambda x: [x])
    if "n_tokens" not in text_units.columns:
        text_units["n_tokens"] = text_units["text"].astype(str).apply(lambda t: len(t.split()))
    if "title" not in text_units.columns:
        text_units["title"] = text_units["id"]

    await write_table_to_storage(text_units, "text_units", context.output_storage)

    # --- 内部で create_community_reports_tfidf を呼ぶ ---
    output_reports = await create_community_reports_tfidf(
        text_units=text_units,
        communities=communities,
        config=config,
        callbacks=context.callbacks,
        cache=context.cache,
    )

    await write_table_to_storage(output_reports, "community_reports", context.output_storage)

    logger.info("Workflow completed: create_community_reports_tfidf")
    return WorkflowFunctionOutput(result=output_reports)


async def create_community_reports_tfidf(
    text_units: pd.DataFrame,
    communities: pd.DataFrame,
    config: GraphRagConfig,
    callbacks,
    cache,
) -> pd.DataFrame:
    """TF-IDF に基づくコミュニティレポート作成処理."""

    # --- relationships を生成 ---
    _, relationships_df = await tfidf_graph(
        text_units=text_units,
        threshold=getattr(config.extract_graph_tfidf, "threshold", 0.1),
        min_df=getattr(config.extract_graph_tfidf, "min_df", 1),
        max_df=getattr(config.extract_graph_tfidf, "max_df", 0.8),
        max_features=getattr(config.extract_graph_tfidf, "max_features", None),
    )

    # --- 文書ノードのみの entities ---
    entities_df = text_units[["id", "text", "document_ids", "n_tokens", "title"]].copy()
    entities_df["type"] = "document"
    entities_df["description"] = ""
    entities_df["text_unit_ids"] = entities_df["document_ids"]
    entities_df["frequency"] = 0.0

    logger.info(f"Text units (entities_df) shape: {entities_df.shape}")
    logger.info(f"Text units columns: {list(entities_df.columns)}")
    logger.info(f"Text units sample:\n{entities_df.head()}")

    logger.info(f"Communities shape: {communities.shape}")
    logger.info(f"Communities columns: {list(communities.columns)}")
    logger.info(f"Communities sample:\n{communities.head()}")


    # --- communities を explode ---
    nodes = explode_communities(communities, entities_df)
    nodes = _prep_nodes(nodes)
    edges = _prep_edges(relationships_df)

    # --- local_context を作成 ---
    max_input_length = getattr(config.community_reports, "max_input_length", 1024)
    local_contexts = build_local_context(
        nodes,
        edges,
        claims=None,
        callbacks=callbacks,
    )

    # --- summarize_communities を呼ぶ ---
    summarization_strategy = config.community_reports.resolved_strategy(config.root_dir, config.get_language_model_config(config.community_reports.model_id))
    summarization_strategy["extraction_prompt"] = summarization_strategy.get("graph_prompt", "")

    community_reports = await summarize_communities(
        nodes,
        communities,
        local_contexts,
        build_level_context,
        callbacks,
        cache,
        summarization_strategy,
        max_input_length=max_input_length,
        async_mode=summarization_strategy.get("async_mode"),
        num_threads=summarization_strategy.get("concurrent_requests", 4),
    )

    # --- finalize ---
    return finalize_community_reports(community_reports, communities)


def _prep_nodes(input: pd.DataFrame) -> pd.DataFrame:
    # node_degree列がなければ0で作成
    if "node_degree" not in input.columns:
        input["node_degree"] = 0

    # descriptionの欠損値を埋める
    input["description"] = input.get("description", pd.Series([""] * len(input))).fillna("No Description")

    # node_details列を作成
    input["node_details"] = input[["id", "title", "description", "node_degree"]].to_dict(orient="records")

    return input

def _prep_edges(input: pd.DataFrame) -> pd.DataFrame:
    input.loc[:, "description"] = input.get("description", "").fillna("No Description")
    if "edge_degree" not in input.columns:
        input["edge_degree"] = 0
    input.loc[:, "edge_details"] = input.loc[:, ["id", "source", "target", "description", "edge_degree"]].to_dict(orient="records")
    return input
