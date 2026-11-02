"""Subprocess entry point for the project-local exact graph and Leiden stages."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from methods.graphs.resources import GIB, available_memory_gib
from methods.graphs.preparation import build_graph
from methods.clustering.partition import cluster_graph

MAX_COMMUNITY_EDGES = 500_000_000

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("graph", "cluster"), required=True)
    parser.add_argument("--graph-source", type=Path)
    parser.add_argument("--threshold", type=float, default=.70)
    parser.add_argument("--threshold-step", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--community-workers", type=int, default=1)
    parser.add_argument("--memory-gb", type=float, default=125.)
    parser.add_argument("--working-memory-gb", type=float, default=24.)
    parser.add_argument("--community-working-memory-gb", type=float, default=125.)
    parser.add_argument("--output-gb", type=float, default=40.)
    parser.add_argument("--free-disk-gb", type=float, default=50.)
    parser.add_argument("--memory-reserve-gb", type=float, default=16.)
    parser.add_argument("--community-max-edges", type=int, default=MAX_COMMUNITY_EDGES)
    parser.add_argument("--resolution", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=17729)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--fixed-threshold", action="store_true")
    args = parser.parse_args()
    args.join_method = "bernoulli"
    if (not 0 < args.threshold <= 1 or not 0 < args.threshold_step <= 1 or args.workers < 1
            or args.community_workers < 1 or args.community_max_edges < 1
            or args.resolution <= 0 or args.iterations == 0 or args.iterations < -1):
        raise ValueError("Invalid graph/clustering settings")
    required = args.memory_gb + args.memory_reserve_gb
    if available_memory_gib() < required:
        raise RuntimeError(f"Need at least {required:.1f} GiB available before starting")
    if shutil.disk_usage(args.output.parent).free / GIB < args.free_disk_gb:
        raise RuntimeError("Graph/partition would violate the free-disk reserve")
    with np.load(args.split) as split:
        train_index, feature_ids = split["train_index"], split["feature_ids"]
    if not len(feature_ids):
        raise ValueError("No prevalence-retained features")
    if args.mode == "graph":
        manifest, _ = build_graph(args, train_index, feature_ids)
        print(json.dumps(manifest), flush=True)
    else:
        if args.graph_source is None:
            raise ValueError("Clustering requires a completed source graph")
        cluster_graph(args, train_index, feature_ids)


if __name__ == "__main__":
    main()
