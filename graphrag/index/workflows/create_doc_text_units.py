# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
import logging
from typing import Any, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    logger.info("Workflow started: create_doc_text_units")
    documents = await load_table_from_storage("documents", context.output_storage)

    chunks = config.chunks

    output = create_doc_text_units(
        documents,
        context.callbacks,
        chunks.group_by_columns,
        prepend_metadata=chunks.prepend_metadata,
    )

    await write_table_to_storage(output, "text_units", context.output_storage)

    logger.info("Workflow completed: create_doc_text_units")
    return WorkflowFunctionOutput(result=output)


def create_doc_text_units(
    documents: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    group_by_columns: list[str],
    prepend_metadata: bool = False,
) -> pd.DataFrame:
    """文書1つをそのまま1チャンクとして text_units を作成"""

    # ID順でソート
    sort = documents.sort_values(by=["id"], ascending=[True])

    # 各文書をリスト化 [(id, text)]
    sort["text_with_ids"] = sort.apply(lambda row: [row["id"], row["text"]], axis=1)

    # group_by_columns に基づいて集約
    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore

    aggregated = (
        (sort.groupby(group_by_columns, sort=False) if len(group_by_columns) > 0 else sort.groupby(lambda _x: True))
        .agg(agg_dict)
        .reset_index()
    )
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    # 文書1つをそのまま chunk として扱う
    def chunker(row: pd.Series) -> pd.Series:
        text, metadata_str = row["texts"][0][1], ""

        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(metadata, dict):
                metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items()) + "\n"
            text = metadata_str + text

        document_id = row["texts"][0][0]
        n_tokens = len(text.split())  # 簡易トークン数

        row["chunk"] = (document_id, text, n_tokens)
        return row

    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)

    # explode は不要
    aggregated.rename(columns={"chunk": "text"}, inplace=True)

    # SHA512でユニークIDを作成
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["text"]), axis=1
    )

    # text を document_ids, text, n_tokens に展開
    aggregated[["document_ids", "text", "n_tokens"]] = pd.DataFrame(
        aggregated["text"].tolist(), index=aggregated.index
    )

    # downstream用に text_units の形式で返す
    return cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
