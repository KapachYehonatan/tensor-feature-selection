"""Fit a partition using saved exact edges, without rebuilding the source graph."""
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import numpy as np


@dataclass(frozen=True)
class Leiden:
    resolution: float = 1.0
    seed: int = 17729
    iterations: int = -1
    workers: int = 1
    working_memory_gib: float = 125

    def __post_init__(self):
        if (not np.isfinite(self.resolution) or self.resolution <= 0 or self.workers < 1
                or self.working_memory_gib <= 0 or self.iterations == 0 or self.iterations < -1):
            raise ValueError("Invalid Leiden settings")

    def fit(self, graph, matrix, selection, output, graph_settings):
        command = graph_settings.command(matrix, selection, output) + [
            "--mode", "cluster", "--graph-source", str(graph),
            "--resolution", str(self.resolution), "--seed", str(self.seed),
            "--iterations", str(self.iterations), "--community-workers", str(self.workers),
            "--community-working-memory-gb", str(self.working_memory_gib)]
        with (Path(output).parent / "clustering.log").open("a") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=Path(__file__).resolve().parents[2])
        state = json.loads((Path(output) / "integration.json").read_text())
        if state["status"] != "complete" or state["edges_pruned"]:
            raise ValueError("Partition is incomplete or its source edges were pruned")
        return state

    def dependencies(self):
        from methods.clustering import partition
        from methods.graphs import preparation, worker
        from execution.cache import write_json
        return (type(self).fit, partition.cluster_graph, partition.fallback_thresholds,
                partition.choose_effective_threshold, partition.stable_cluster_labels,
                preparation.count_threshold_edges, preparation.filter_exact_graph, preparation.edge_strength, worker.main, write_json)

    def source_files(self):
        return (Path(__file__).with_name("community.py"), Path(__file__).with_name("partition.py"),
                Path(__file__).parents[1] / "graphs/resources.py")
