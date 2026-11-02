"""Exact graph construction, independent of partition settings."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BernoulliGraph:
    threshold: float = .70
    step: float = .05
    edge_limit: int = 500_000_000
    workers: int = 4
    memory_gib: float = 125
    working_memory_gib: float = 24
    reserve_gib: float = 16
    free_disk_gib: float = 50
    output_gib: float = 40

    def __post_init__(self):
        if (not 0 < self.threshold <= 1 or not 0 < self.step <= 1 or self.workers < 1
                or self.edge_limit < 1 or min(self.memory_gib, self.working_memory_gib, self.output_gib) <= 0
                or min(self.reserve_gib, self.free_disk_gib) < 0):
            raise ValueError("Invalid Bernoulli graph/resource settings")

    def command(self, matrix, selection, output):
        return [sys.executable, "-u", "-m", "methods.graphs.worker",
                str(matrix), str(selection), str(output), "--threshold", str(self.threshold),
                "--threshold-step", str(self.step), "--workers", str(self.workers),
                "--memory-gb", str(self.memory_gib), "--working-memory-gb", str(self.working_memory_gib),
                "--memory-reserve-gb", str(self.reserve_gib), "--free-disk-gb", str(self.free_disk_gib),
                "--output-gb", str(self.output_gib), "--community-max-edges", str(self.edge_limit)]

    def build(self, matrix, selection, output):
        with (Path(output).parent / "graph.log").open("a") as log:
            subprocess.run(self.command(matrix, selection, output) + ["--mode", "graph"],
                           check=True, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)

    def dependencies(self):
        from methods.graphs.preparation import build_graph, pack_sparse_detection
        from methods.graphs.worker import main
        from execution.cache import write_json
        return (type(self).build, type(self).command, build_graph, pack_sparse_detection, main, write_json)

    def source_files(self):
        return tuple(Path(__file__).with_name(name) for name in ("exact.py", "kernels.py", "resources.py"))
