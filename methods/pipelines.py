"""Readable compositions; selectors own evidence and discovery behavior."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from datasets.loading import Dataset
from methods.graphs.bernoulli_join import BernoulliGraph
from methods.clustering.leiden import Leiden
from methods.selectors.mann_whitney import MannWhitney
from methods.selectors.bins import Bins
from methods.correction import EBH
from methods.preprocessing import retained_pair_values
from methods.representations import collapsed_average, aggregate_pair_clusters

Selector = MannWhitney | Bins


@dataclass(frozen=True)
class Method:
    name: str
    representation: str
    selector: Selector
    graph: BernoulliGraph | None = None
    clustering: Leiden | None = None

    def __post_init__(self):
        if self.representation not in {"raw", "bacteria", "gene", "join"}:
            raise ValueError("Unknown representation")
        if self.representation == "join" and (self.graph is None or self.clustering is None):
            raise ValueError("Join requires separate graph and clustering components")


def Bacteria(selector: Selector | None = None, name: str = "Bacteria-collapsed") -> Method:
    return Method(name, "bacteria", MannWhitney() if selector is None else selector)


def Gene(selector: Selector | None = None, name: str = "Gene-collapsed") -> Method:
    return Method(name, "gene", MannWhitney() if selector is None else selector)


def BernoulliJoin(selector: Selector | None = None, graph: BernoulliGraph | None = None,
                  clustering: Leiden | None = None, name: str = "Bernoulli-join") -> Method:
    return Method(name, "join", MannWhitney() if selector is None else selector,
                  BernoulliGraph() if graph is None else graph, Leiden() if clustering is None else clustering)


def BinsEValue(alpha: float = .10, name: str = "Bins e-value") -> Method:
    return Method(name, "raw", Bins(EBH(alpha)))


def construct(data: Dataset, training: np.ndarray, retained: np.ndarray, method: Method,
              labels: np.ndarray | None = None) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    """Return all-sample predictors, feature IDs, and the complete family size.

    Group memberships and constant exclusion use training rows only. Constant
    raw pairs still participate in means; remove constants after aggregation.
    The raw family's size is the supplied catalog, before prevalence filtering.
    IDs refer to original pairs/entities, or to learned clusters for join.
    """
    values = retained_pair_values(data.values, retained)
    if method.representation == "raw":
        features, ids, full_family = values, retained, data.values.shape[1]
    elif method.representation == "join":
        features, _ = aggregate_pair_clusters(values, labels)
        ids = np.arange(features.shape[1])
        full_family = len(ids)
    else:
        entity = data.pair_bacteria if method.representation == "bacteria" else data.pair_gene
        features, ids = collapsed_average(values, entity[retained], int(entity.max()) + 1)
        full_family = int(entity.max()) + 1
    fitted = features[training]
    eligible = (np.flatnonzero(fitted.max(axis=0).toarray().ravel() > fitted.min(axis=0).toarray().ravel())
                if fitted.shape[1] else np.array([], int))
    return features[:, eligible].tocsr(), np.asarray(ids)[eligible], full_family
