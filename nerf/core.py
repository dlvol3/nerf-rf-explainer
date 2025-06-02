import pandas as pd
import numpy as np
import networkx as nx
import time
import math
import os
from typing import List, Tuple, Optional, Dict


# Decorator for timing function execution
# Yue Zhang <yue.zhang@lih.lu>
def timing(func):
    def wrap(*args, **kw):
        print(f"<function name: {func.__name__}>")
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        print(f"[timecosts: {end - start:.2f} s]")
        return result

    return wrap


# Flatten the structure of random forest classifiers
# Extract all decision node info from all trees
@timing
def flatforest(rf, testdf: pd.DataFrame) -> pd.DataFrame:
    tree_infotable = pd.DataFrame()
    for t, tree in enumerate(rf.estimators_):
        node_count = tree.tree_.node_count
        node_index = np.arange(node_count)
        lc = tree.tree_.children_left
        rc = tree.tree_.children_right
        feature_index = tree.tree_.feature
        threshold = tree.tree_.threshold
        gini = tree.tree_.impurity
        value = tree.tree_.value
        node_in_forest = node_index + rf.decision_path(testdf)[1].item(t)

        df_tree = pd.DataFrame(
            {
                "node_index": node_index,
                "left_c": lc,
                "right_c": rc,
                "feature_index": feature_index,
                "feature_threshold": threshold,
                "gini": gini,
                "tree_index": t + 1,
                "nodeInForest": node_in_forest,
            }
        )

        # Calculation of the default gini gain
        gs_list, node_type = [], []
        for i in range(node_count):
            if feature_index[i] == -2:
                gs_list.append(-1)
                node_type.append("leaf_node")
                continue
            li, ri = lc[i], rc[i]
            total = np.sum(value[i])
            gs = (
                gini[i]
                - (np.sum(value[li]) / total) * gini[li]
                - (np.sum(value[ri]) / total) * gini[ri]
            )
            gs_list.append(gs)
            node_type.append("decision_node")

        df_tree["GS"] = gs_list
        df_tree["node_type"] = node_type
        tree_infotable = pd.concat([tree_infotable, df_tree])

    print(f"Forest flattened with {tree_infotable.shape[0]} rows.")
    return tree_infotable.reset_index(drop=True)


# Extract paths that each sample travels through during prediction
# Match with corresponding node decisions and prediction outcomes
@timing
def extarget(
    rf, testdf: pd.DataFrame, flatdf: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_hits = []
    pred_records = []
    dp_indptr = rf.decision_path(testdf)[0].indptr
    indices = rf.decision_path(testdf)[0].indices

    for i in range(len(dp_indptr) - 1):
        floor, ceil = dp_indptr[i], dp_indptr[i + 1]
        preds, trees = [], []
        for t, tree in enumerate(rf.estimators_):
            preds.append(tree.predict(testdf)[i])
            trees.append(t)

        match_array = np.array(preds) == rf.predict(testdf)[i]
        pred_df = pd.DataFrame(
            {
                "prediction": preds,
                "tree index": trees,
                "sample": i,
                "matching": ["match" if m else "not_matching" for m in match_array],
            }
        )
        pred_records.append(pred_df)

        # Extract the node path hit for each sample in the forest
        hits = flatdf[flatdf["nodeInForest"].isin(indices[floor:ceil])][
            ["feature_index", "GS", "tree_index", "feature_threshold"]
        ].copy()
        hits["sample_index"] = i
        raw_hits.append(hits)

    return pd.concat(raw_hits).reset_index(drop=True), pd.concat(
        pred_records
    ).reset_index(drop=True)


# Generate feature interaction table from matching decision paths
# For each sample and tree, generate all valid feature pairs that contribute to correct predictions
@timing
def nerftab(
    raw_hits: pd.DataFrame, pred_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    singles, pairs = [], []
    for i in pred_df["sample"].unique():
        matched_trees = (
            pred_df[(pred_df["sample"] == i) & (pred_df["matching"] == "match")][
                "tree index"
            ]
            + 1
        )
        nodes = raw_hits[
            (raw_hits["sample_index"] == i)
            & (raw_hits["tree_index"].isin(matched_trees))
            & (raw_hits["feature_index"] != -2)
        ]
        singles.append(nodes)
        for tree_id in matched_trees:
            subset = nodes[nodes["tree_index"] == tree_id]
            for i1 in range(len(subset)):
                for i2 in range(i1 + 1, len(subset)):
                    fi, fj = sorted(
                        [
                            subset.iloc[i1]["feature_index"],
                            subset.iloc[i2]["feature_index"],
                        ]
                    )
                    gs_i, gs_j = subset.iloc[i1]["GS"], subset.iloc[i2]["GS"]
                    threshold_i, threshold_j = (
                        subset.iloc[i1]["feature_threshold"],
                        subset.iloc[i2]["feature_threshold"],
                    )
                    pairs.append(
                        [fi, fj, gs_i, gs_j, threshold_i, threshold_j, tree_id, i]
                    )
    df_singles = pd.concat(singles).reset_index(drop=True)
    df_pairs = pd.DataFrame(
        pairs,
        columns=[
            "feature_i",
            "feature_j",
            "GS_i",
            "GS_j",
            "threshold_i",
            "threshold_j",
            "tree_index",
            "sample_index",
        ],
    )
    return df_singles, df_pairs


# Local explanation: edge intensity between feature pairs from one sample
@timing
def localnerf(df_pairs: pd.DataFrame, local_index: int) -> pd.DataFrame:
    subset = df_pairs[df_pairs["sample_index"] == local_index].copy()
    subset["GSP"] = subset["GS_i"] + subset["GS_j"]
    grouped = (
        subset.groupby(["feature_i", "feature_j"], as_index=False)["GSP"]
        .agg(["size", "sum"])
        .reset_index()
    )
    grouped.columns = ["feature_i", "feature_j", "count", "total_gsp"]
    grouped["EI"] = grouped["count"] * grouped["total_gsp"]
    return grouped[["feature_i", "feature_j", "EI"]]


# Generate whole network and subnetwork from one sample
@timing
def twonets(
    ei_df: pd.DataFrame,
    filename: str,
    feature_index: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    top_k_degree_factor: int = 4,
    top_k_ei_factor: int = 10,
) -> Dict:
    if feature_index and feature_names:
        replace_dict = dict(zip(feature_index, feature_names))
        ei_df = ei_df.replace(replace_dict)

    if not os.path.exists("output"):
        os.makedirs("output")

    # export the 'everything' network
    ei_df.to_csv(f"output/{filename}_everything.txt", sep="\t", index=False)

    G = nx.from_pandas_edgelist(ei_df, "feature_i", "feature_j", edge_attr="EI")
    dc = nx.degree_centrality(G)
    dc_df = pd.DataFrame.from_dict(dc, orient="index", columns=["degree_centrality"])
    dc_df.to_csv(f"output/{filename}_DC.txt", sep="\t")

    # take the top sub of the DC
    top_nodes = sorted(dc, key=dc.get, reverse=True)[
        : int(top_k_degree_factor * math.sqrt(len(dc)))
    ]
    # top edge intensity
    top_edges = ei_df.sort_values("EI", ascending=False)[
        : int(top_k_ei_factor * math.sqrt(len(ei_df)))
    ]
    # sub network with gene names
    sub_net = top_edges[
        top_edges["feature_i"].isin(top_nodes) & top_edges["feature_j"].isin(top_nodes)
    ]
    sub_net.to_csv(f"output/{filename}_sub.txt", sep="\t", index=False)

    return {
        "network": ei_df,
        "centrality": dc,
        "top_nodes": top_nodes,
        "top_edges": top_edges,
        "subnetwork": sub_net,
    }


# One-click pipeline: from RF model and test data to feature interaction network of one sample
@timing
def run_nerf_pipeline(
    rf,
    testdf: pd.DataFrame,
    sample_index: int,
    feature_index: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    flat = flatforest(rf, testdf)
    raw_hits, pred_info = extarget(rf, testdf, flat)
    singles, pairs = nerftab(raw_hits, pred_info)
    ei_df = localnerf(pairs, sample_index)
    results = twonets(ei_df, f"sample_{sample_index}", feature_index, feature_names)
    return results
