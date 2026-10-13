#!/usr/bin/env python3
"""Cluster stored graphs for the Bernoulli-join side experiments."""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np

JOIN_PROJECT = Path("/home/kapachy/Microbium/bernoulli-correlation-join")
sys.path.insert(0, str(JOIN_PROJECT))
from bernoulli_join import GIB, iter_edge_shards, write_json_atomic  # noqa: E402
from community_detection import run_community_detection, run_kaminpar  # noqa: E402


def signed_graph(source, destination, sign, threshold, disk_reserve_gib=500):
    """Keep exact edges of one sign; no similarity is recomputed."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    source_manifest = json.loads((source / "run.json").read_text())
    if source_manifest["status"] != "complete" or source_manifest["threshold"] > threshold:
        raise ValueError("source graph is incomplete or its threshold is too high")
    marker = destination / "run.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved.get("source_graph") == str(source.resolve())
                and saved.get("source_mtime_ns") == (source / "run.json").stat().st_mtime_ns
                and saved.get("threshold") == threshold and saved.get("sign") == sign
                and (saved.get("edge_count", 0) == 0 or (destination / "edges").exists())):
            return saved
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "edges").mkdir(parents=True)
    os.link(source / "nodes.npy", destination / "nodes.npy")
    edges = 0
    parts = 0
    for shard in iter_edge_shards(source):
        keep = np.asarray(shard)[np.asarray(shard["phi"]) * sign >= np.float32(threshold)]
        if keep.size:
            if shutil.disk_usage(destination).free - keep.nbytes < disk_reserve_gib * GIB:
                raise RuntimeError("signed graph would violate free-disk reserve")
            np.save(destination / "edges" / f"part-{parts:06d}.npy", keep, allow_pickle=False)
            edges += int(keep.size)
            parts += 1
    manifest = {
        **source_manifest, "status": "complete", "method": "signed-exact-filter",
        "source_graph": str(source.resolve()),
        "source_mtime_ns": (source / "run.json").stat().st_mtime_ns,
        "threshold": float(threshold), "sign": sign,
        "edge_count": edges, "edge_shards": parts, "edge_bytes": edges * 12,
    }
    write_json_atomic(marker, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variation", choices=("signed_families", "kaminpar", "coprojection"))
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--free-disk-gb", type=float, default=500)
    parser.add_argument("--memory-gb", type=float, default=200)
    args = parser.parse_args()
    available = next(int(line.split()[1]) * 1024 / GIB for line in
                     Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    print(f"available-memory={available:.1f}GiB; process ceiling={args.memory_gb:.1f}GiB",
          flush=True)
    if shutil.disk_usage(args.output.parent).free / GIB < args.free_disk_gb:
        raise RuntimeError("insufficient free disk for the configured reserve")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.variation == "coprojection":
        run_community_detection(
            args.graph, output_dir=args.output, workers=args.workers,
            memory_limit_gib=args.memory_gb, working_memory_gib=args.memory_gb,
            output_limit_gib=40, free_disk_reserve_gib=args.free_disk_gb,
        )
    elif args.variation == "kaminpar":
        if args.k is None:
            raise ValueError("KaMinPar requires the baseline edge-bearing community count")
        run_kaminpar(
            args.graph, args.output / "community", args.k, workers=args.workers,
            memory_limit_gib=args.memory_gb, working_memory_gib=100,
            output_limit_gib=40, free_disk_reserve_gib=args.free_disk_gb,
            imbalance_eps=0.03, weight_scale=4, split_components=True,
        )
    else:
        negative_graph = args.output / "negative_graph"
        negative = signed_graph(args.graph, negative_graph, -1, args.threshold, args.free_disk_gb)
        print(f"negative: {negative['edge_count']:,} exact edges", flush=True)
        source = json.loads((args.graph / "run.json").read_text())
        baseline_community = all((args.graph / name).exists() for name in
                                 ("clusters.csv.gz", "cluster_summary.csv", "community.json"))
        reuse_positive = negative["edge_count"] == 0 and baseline_community
        if reuse_positive:
            positive_dir = args.output / "positive_community"
            positive_dir.mkdir(exist_ok=True)
            for name in ("clusters.csv.gz", "cluster_summary.csv", "community.json"):
                target = positive_dir / name
                if not target.exists():
                    os.link(args.graph / name, target)
            print(f"positive: reusing {source['edge_count']:,}-edge baseline communities", flush=True)
        else:
            print("positive: streaming signed edges from the stored graph", flush=True)
            completion = args.output / "positive_community/community.json"
            if not (completion.exists() and json.loads(completion.read_text()).get("status") == "complete"):
                run_community_detection(
                    args.graph, output_dir=args.output / "positive_community",
                    workers=args.workers, memory_limit_gib=args.memory_gb,
                    working_memory_gib=100, output_limit_gib=40,
                    free_disk_reserve_gib=args.free_disk_gb,
                    edge_sign=1, leiden_iterations=2,
                )
            shutil.rmtree(args.output / "positive_graph", ignore_errors=True)
        for name, graph in (("negative", negative_graph),):
            completion = args.output / f"{name}_community/community.json"
            if completion.exists() and json.loads(completion.read_text()).get("status") == "complete":
                print(f"{name}: reusing completed communities", flush=True)
                continue
            run_community_detection(
                graph, output_dir=args.output / f"{name}_community",
                workers=args.workers, memory_limit_gib=args.memory_gb,
                working_memory_gib=100, output_limit_gib=40,
                free_disk_reserve_gib=args.free_disk_gb,
                leiden_iterations=2,
            )


if __name__ == "__main__":
    main()
