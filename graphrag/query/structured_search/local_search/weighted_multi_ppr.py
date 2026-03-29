import networkx as nx
import pandas as pd

def build_bipartite_graph_from_relationships(relationships_df: pd.DataFrame):
    """
    relationships_df を受け取り NetworkX グラフを構築。
    - relationships_df は少なくとも columns: ['source','target','weight','description'] を想定
    - source が単語文字列の場合はそのまま単語ノードになる
    - source がハイパーエッジ UUID で description に単語が書かれている場合は
      description から単語を抽出して mapping を作る（後で personalization に使えるように）
    Returns:
      G: NetworkX DiGraph (directed)
      meta: dict with helpful helpers:
         - 'term_to_source_nodes': dict(term -> [source_node_id,...])
         - 'document_nodes': set(...)   # all targets (files)
    """
    G = nx.DiGraph()
    document_nodes = set()
    doc_to_terms = {}  # doc -> list of (term, w_forward)

    for _, row in relationships_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        w = float(row.get("weight", 1.0))
        #desc = row.get("description", "")

        # add nodes with type metadata (optional)
        if not G.has_node(src):
            # decide if source looks like a plain word or uuid-like
            # we don't enforce; just store raw
            G.add_node(src, kind="source")
        if not G.has_node(tgt):
            G.add_node(tgt, kind="document")

        # add directed edge source -> target with weight
        G.add_edge(src, tgt, weight=w)

        # collect document nodes
        document_nodes.add(tgt)

        # collect for reverse weight computation
        doc_to_terms.setdefault(tgt, []).append((src, w))
    
    for tgt, src_weight_list in doc_to_terms.items():
        total = sum(w for _, w in src_weight_list)
        if total <= 0:
            # fallback: uniform
            n = len(src_weight_list)
            for src, _ in src_weight_list:
                rev_w = 1.0 / n if n > 0 else 0.0
                G.add_edge(tgt, src, weight=rev_w)
        else:
            for src, w in src_weight_list:
                rev_w = float(w) / float(total)
                G.add_edge(tgt, src, weight=rev_w)

    return G, document_nodes

def weighted_multisource_ppr(
    relationships_df: pd.DataFrame,
    query_term_weights: dict,
    teleport_prob: float = 0.15,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    top_k: int | None = 10,                                         #変更検討箇所
):
    """
    重み付きマルチソース PPR を実行して文書ノードのランキングを返す。

    Args:
      relationships_df: TF-IDF の relationships DataFrame (source,target,weight,description)
      query_term_weights: dict mapping term (str) -> weight (float). 例: {"graph":0.7, "tfidf":0.3}
      teleport_prob: teleport (personalization) probability
      max_iter, tol: pagerank 性能パラメータ
      top_k: 上位 k 件だけ返す（None で全件）

    Returns:
      pd.DataFrame: columns = ['node','score','rank'] (document nodes only, 降順)
      full_ppr: dict 全ノードに対する PPR スコア（必要なら利用）
    """
    # 1) グラフ構築
    G, doc_nodes = build_bipartite_graph_from_relationships(relationships_df)

    # 2) personalization ベクトル作成（term が source として直接使える）
    # filter out terms not present in graph
    filtered = {t: float(w) for t, w in query_term_weights.items() if t in G.nodes}
    if not filtered:
        raise ValueError("No query terms found among graph source nodes. Check token normalization.")

    # normalize to sum=1 (NetworkX 要件)
    total = sum(filtered.values())
    personalization = {node: weight / total for node, weight in filtered.items()}

    # 3) run personalized PageRank (uses edge attribute 'weight')
    ppr = nx.pagerank(
        G,
        alpha=1.0 - teleport_prob,  # networkx pagerank's teleport_proba argument is damping factor; careful!
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        weight="weight",
    )
    # NOTE: networkx.pagerank uses 'teleport_prob' as damping factor (probability of following a link).
    # In PPR notation often teleport_prob=teleport_prob, so we convert: damping = 1 - teleport.

    # 4) 文書ノードの抽出と DataFrame 化
    rows = [{"node": node, "score": float(ppr.get(node, 0.0))} for node in doc_nodes]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    if not df.empty:
        df["rank"] = df["score"].rank(method="min", ascending=False).astype(int)
    else:
        df["rank"] = pd.Series(dtype=int)

    if top_k is not None:
        df = df.head(top_k)

    return df, ppr
