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
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads((Path(output) / "integration.json").read_text())
        if state["status"] != "complete" or state["edges_pruned"]:
            raise ValueError("Partition is incomplete or its source edges were pruned")
        return state

    def dependencies(self):
        from methods.graphs import worker
        return (type(self).fit, worker.cluster_graph, worker.fallback_thresholds,
                worker.choose_effective_threshold, worker.count_threshold_edges,
                worker.filter_exact_graph, worker.edge_strength, worker.stable_cluster_labels)
