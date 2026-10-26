"""Experiment scope and evaluation settings, separate from method definitions."""
from dataclasses import dataclass, field
from methods.preprocessing import Preprocessing
from methods.pipelines import Method


@dataclass(frozen=True)
class Evaluation:
    folds: int = 5
    seed: int = 42
    fine_k: tuple = tuple(range(1, 11))
    coarse_k: tuple = (10, 20, 50, 75)
    shuffles: int = 999

    def __post_init__(self):
        if self.folds < 2 or self.shuffles < 1 or not self.k or min(self.k) < 1:
            raise ValueError("Invalid evaluation settings")

    @property
    def k(self):
        return tuple(sorted(set(self.fine_k + self.coarse_k)))

    @staticmethod
    def tag(values):
        return (f"{values[0]}-{values[-1]}" if len(values) > 1 and tuple(values) == tuple(range(values[0], values[-1] + 1))
                else "_".join(map(str, values)))


@dataclass(frozen=True)
class Experiment:
    name: str
    datasets: tuple[str, ...]
    methods: tuple[Method, ...]
    metrics: tuple[str, ...]
    outputs: tuple[str, ...]
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    evaluation: Evaluation = field(default_factory=Evaluation)

    def __post_init__(self):
        if not self.name or self.name.startswith(".") or "/" in self.name or "\\" in self.name:
            raise ValueError("Experiment name must be one plain directory name")
        if not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Datasets must be nonempty and unique")
        if not self.methods or len({m.name for m in self.methods}) != len(self.methods):
            raise ValueError("Method names must be nonempty and unique")
        if set(self.metrics) - {"prediction", "discoveries", "correlation", "shuffled_fdp"}:
            raise ValueError("Unknown metric")
        if any(not p.endswith(".png") or "/" in p or "\\" in p or p.startswith(".") for p in self.outputs):
            raise ValueError("Outputs must be plain PNG filenames")
