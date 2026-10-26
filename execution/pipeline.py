"""Shared fold-local and all-sample execution of compositional methods."""
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import traceback
import subprocess
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from datasets.loading import load
from execution import cache
from methods import preprocessing as prep, representations as reps
from methods.pipelines import construct
from methods.graphs.bernoulli_join import JOIN_PROJECT
from methods.selectors.mann_whitney import MannWhitney
from metrics.prediction import predict_prefixes, predict_model, point_metrics, new_model, summarize
from metrics.correlation import average_pairwise_absolute_correlation
from metrics.shuffled_fdp import label_permutations, discovery_statistics


@dataclass
class Context:
    scope: str
    training: np.ndarray
    validation: np.ndarray


@dataclass
class Family:
    key: str
    directory: Path
    values: sparse.csr_matrix
    ids: np.ndarray
    original_size: int
    references: dict


@lru_cache(None)
def external_versions():
    text = subprocess.check_output([str(JOIN_PROJECT / ".venv/bin/python"), "-c",
        "import json,sys,numpy,numba,igraph;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,numba=numba.__version__,igraph=igraph.__version__)))"], text=True)
    return json.loads(text)


def contexts(y, evaluation):
    if set(np.unique(y)) != {0, 1} or np.bincount(np.asarray(y, dtype=int)).min() < evaluation.folds:
        raise ValueError("Stratified CV requires both classes with at least one per fold")
    folds = StratifiedKFold(evaluation.folds, shuffle=True, random_state=evaluation.seed)
    result, covered = [], np.zeros(len(y), int)
    for fold, (a, b) in enumerate(folds.split(np.zeros(len(y)), y)):
        if np.intersect1d(a, b).size or len(a) + len(b) != len(y):
            raise ValueError("Invalid fold isolation")
        covered[b] += 1
        result.append(Context(f"fold_{fold}", a, b))
    if not np.all(covered == 1):
        raise ValueError("Every participant must be validated exactly once")
    return result


def prepare_family(data, context, preprocessing, method, notify=None):
    notify = notify or (lambda stage: None)
    fitting = cache.array_hash(context.training)
    key, record = cache.identity("preprocessing", dict(detection=cache.sparse_hash(data.values[context.training], values=False), fit=fitting),
        preprocessing, (type(preprocessing).fit, prep.prevalence_mask, prep.sample_support), software=("numpy", "scipy"))
    directory = cache.stage_path(data.name, context.scope, "preprocessing", key, record)
    selection = directory / "selection.npz"
    if not cache.complete(directory, key):
        retained, cutoff = preprocessing.fit(data.values, context.training)
        if not len(retained):
            raise ValueError("Prevalence filtering retained no pairs")
        np.savez_compressed(selection, train_index=context.training, test_index=context.validation,
                            feature_ids=retained, prevalence_threshold=cutoff)
        cache.seal(directory, key, [selection])
    with np.load(selection) as saved:
        retained = saved["feature_ids"]
        np.testing.assert_array_equal(saved["train_index"], context.training)
    partition, partition_key, graph_key = None, None, None
    if method.representation == "join":
        detected = data.values[context.training][:, retained].tocsr()
        detection = dict(indices=cache.array_hash(detected.indices), indptr=cache.array_hash(detected.indptr),
                         shape=detected.shape, ids=cache.array_hash(retained), fit=fitting)
        graph_key, graph_record = cache.identity("graph", detection, dict(graph=asdict(method.graph),
            external={k: v for k, v in external_versions().items() if k != "igraph"}),
            method.graph.dependencies(),
            (JOIN_PROJECT / "bernoulli_join.py",), software=("numpy", "scipy"))
        graph = cache.stage_path(data.name, context.scope, "graph", graph_key, graph_record)
        cached = cache.complete(graph, graph_key)
        notify("graph")
        print(f"{data.name}/{context.scope}: graph {'cached' if cached else 'building'}; log={graph / 'graph.log'}", flush=True)
        if not cached:
            method.graph.build(data.path / "X_valid.npz", selection, graph / "work")
            cache.seal(graph, graph_key, [p for p in (graph / "work").rglob("*") if p.is_file()])
        partition_key, partition_record = cache.identity("clustering", dict(graph=graph_key),
            dict(graph_resources=asdict(method.graph), clustering=asdict(method.clustering), external=external_versions()),
            method.clustering.dependencies(),
            (JOIN_PROJECT / "community_detection.py",), software=("numpy", "scipy"))
        partition_dir = cache.stage_path(data.name, context.scope, "clustering", partition_key, partition_record)
        cached = cache.complete(partition_dir, partition_key)
        notify("clustering")
        print(f"{data.name}/{context.scope}: clustering {'cached' if cached else 'running'}; log={partition_dir / 'clustering.log'}", flush=True)
        if not cached:
            method.clustering.fit(graph / "work", data.path / "X_valid.npz", selection,
                                  partition_dir / "work", method.graph)
            cache.seal(partition_dir, partition_key, [p for p in (partition_dir / "work").rglob("*") if p.is_file()])
        with np.load(partition_dir / "work/assignments.npz") as saved:
            np.testing.assert_array_equal(saved["feature_ids"], retained)
            partition = saved["clusters"]
    family_key, family_record = cache.identity("representation", dict(values=data.fingerprint["X_valid.npz"],
        mappings=data.fingerprint["edges.npz"], preprocessing=key, partition=partition_key),
        dict(representation=method.representation),
        (construct, prep.retained_pair_values, reps.collapsed_average, reps.marginal_matrix, reps.aggregate_pair_clusters),
        software=("numpy", "scipy"))
    family_dir = cache.stage_path(data.name, context.scope, "representation", family_key, family_record)
    if not cache.complete(family_dir, family_key):
        values, ids, size = construct(data, context.training, retained, method, partition)
        sparse.save_npz(family_dir / "values.npz", values)
        np.savez_compressed(family_dir / "features.npz", ids=ids, original_size=size)
        cache.seal(family_dir, family_key, [family_dir / "values.npz", family_dir / "features.npz"])
    with np.load(family_dir / "features.npz") as saved:
        ids, size = saved["ids"], int(saved["original_size"])
    return Family(family_key, family_dir, sparse.load_npz(family_dir / "values.npz").tocsr(), ids, size,
                  dict(preprocessing=key, graph=graph_key, clustering=partition_key, representation=family_key))


def test_dependencies(selector):
    return selector.dependencies()


def selection(data, context, family, method, rank_limit):
    selector = method.selector
    key, record = cache.identity("testing", dict(values=cache.sparse_hash(family.values[context.training]), ids=cache.array_hash(family.ids),
        labels=cache.array_hash(data.samples.y.to_numpy()[context.training])), dict(kind=selector.kind, settings=selector.test_settings()),
        test_dependencies(selector), software=("numpy", "scipy"))
    directory = cache.stage_path(data.name, context.scope, "testing", key, record)
    if not cache.complete(directory, key):
        result = selector.test(family.values[context.training], data.samples.y.to_numpy()[context.training])
        np.savez_compressed(directory / "evidence.npz", **result)
        cache.seal(directory, key, [directory / "evidence.npz"])
    with np.load(directory / "evidence.npz") as saved:
        evidence = {k: saved[k] for k in saved.files}
    selected_key, selected_record = cache.identity("selection", dict(testing=key),
        dict(selector=asdict(selector), rank_limit=rank_limit, original_family=family.original_size),
        selector.selection_dependencies(), software=("numpy", "scipy"))
    selected_dir = cache.stage_path(data.name, context.scope, "selection", selected_key, selected_record)
    if not cache.complete(selected_dir, selected_key):
        result = selector.select(evidence, family.values[context.training], family.ids, family.original_size, rank_limit)
        np.savez_compressed(selected_dir / "selection.npz", **result)
        table = pd.DataFrame(dict(feature_id=family.ids, **evidence))
        table["native"] = np.isin(np.arange(len(table)), result["native"])
        if "q" in result:
            table["q_by"] = result["q"]
        table.to_csv(selected_dir / "features.csv.gz", index=False)
        cache.seal(selected_dir, selected_key, [selected_dir / "selection.npz", selected_dir / "features.csv.gz"])
    with np.load(selected_dir / "selection.npz") as saved:
        result = {k: saved[k] for k in saved.files}
    return selected_key, result, key


def prediction(data, context, family, selected_key, selected, evaluation):
    key, record = cache.identity("prediction", dict(selection=selected_key,
        validation=cache.array_hash(context.validation), labels=cache.array_hash(data.samples.y.to_numpy()),
        validation_values=cache.sparse_hash(family.values[context.validation][:, selected["order"]]), samples=data.fingerprint["samples.csv"]),
        dict(k=evaluation.k, C=1, balanced=True, seed=42), (predict_prefixes, predict_model, new_model), software=("numpy", "scipy", "scikit-learn"))
    directory = cache.stage_path(data.name, context.scope, "prediction", key, record)
    if not cache.complete(directory, key):
        values = family.values[:, selected["order"]]
        a, b = context.training, context.validation
        predictions = predict_prefixes(values[a], values[b], data.samples.y.to_numpy()[a], data.samples.y.to_numpy()[b], evaluation.k)
        predictions["sample_index"] = b[predictions.test_position]
        predictions["sample_id"] = data.samples.sample_id.to_numpy()[predictions.sample_index]
        predictions.to_csv(directory / "predictions.csv.gz", index=False)
        cache.seal(directory, key, [directory / "predictions.csv.gz"])
    predictions = pd.read_csv(directory / "predictions.csv.gz", dtype={"sample_id": str})
    metrics = pd.DataFrame([dict(requested_k=k, effective_k=min(k, len(selected["order"])),
        **point_metrics(group.y.to_numpy(), group.probability.to_numpy()))
        for k, group in predictions.groupby("requested_k")])
    return metrics, predictions, key


def shuffled_counts(data, family, methods, evaluation):
    """One transient evidence batch per family; all cutoffs reuse each MW vector."""
    y = data.samples.y.to_numpy()
    targets = label_permutations(y, data.name + "/full_data", evaluation.shuffles, evaluation.seed)
    states, completed = [], set()
    for method in methods:
        selector = method.selector
        key, record = cache.identity("shuffles", dict(family=family.key, labels=cache.array_hash(y)),
            dict(selector=asdict(selector), budget=evaluation.shuffles, seed=evaluation.seed),
            (*test_dependencies(selector), *selector.selection_dependencies(), label_permutations),
            software=("numpy", "scipy"))
        directory = cache.stage_path(data.name, "full", "shuffles", key, record)
        values = np.full(evaluation.shuffles, -1, dtype=int)
        if cache.complete(directory, key):
            values = np.load(directory / "counts.npy")
            completed.add(directory)
        elif cache.complete(directory / "checkpoint", key):
            values = np.load(directory / "checkpoint/counts.npy")
        size = selector.family_size(len(family.ids), family.original_size)
        if (values.shape != (evaluation.shuffles,) or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < -1) or np.any(values > size)):
            raise ValueError("Invalid shuffle checkpoint")
        states.append((method, directory, key, values))
    if len(completed) == len(states):
        return ({method.name: values for method, _, _, values in states},
                {method.name: key for method, _, key, _ in states})
    selector = methods[0].selector
    prepared = selector.prepare(family.values)
    baselines = {s[0].selector.correction: s for s in states if isinstance(s[0].selector, MannWhitney)
                 and s[0].selector.correlation_threshold is None}
    for left in range(0, evaluation.shuffles, 8):
        indices = []
        for i in range(left, min(left + 8, evaluation.shuffles)):
            pending = [s for s in states if s[3][i] < 0]
            for state in pending:
                baseline = baselines.get(state[0].selector.correction)
                if baseline is not None and baseline[3][i] == 0:
                    state[3][i] = 0
            if any(s[3][i] < 0 for s in pending):
                indices.append(i)
        if indices:
            vectors = selector.batch_test(family.values, targets[indices], prepared)
            for i, vector in zip(indices, vectors):
                evidence = {selector.kind: vector}
                original_counts = {}
                for method, directory, key, values in states:
                    if values[i] >= 0:
                        continue
                    if original_counts.get(method.selector.correction) == 0 and isinstance(method.selector, MannWhitney):
                        values[i] = 0
                    else:
                        values[i] = len(method.selector.select(evidence, family.values, family.ids, family.original_size, 0)["native"])
                    if isinstance(method.selector, MannWhitney) and method.selector.correlation_threshold is None:
                        original_counts[method.selector.correction] = values[i]
        for _, directory, key, values in states:
            if directory in completed:
                continue
            checkpoint = directory / "checkpoint"
            checkpoint.mkdir(exist_ok=True)
            temporary = checkpoint / "counts.tmp.npy"
            np.save(temporary, values)
            temporary.replace(checkpoint / "counts.npy")
            cache.seal(checkpoint, key, [checkpoint / "counts.npy"])
        print(f"{data.name}: shuffled {min(left + 8, evaluation.shuffles)}/{evaluation.shuffles}", flush=True)
    result = {}
    for method, directory, key, values in states:
        size = method.selector.family_size(len(family.ids), family.original_size)
        if np.any(values < 0) or np.any(values > size):
            raise ValueError("Incomplete or invalid shuffled counts")
        if directory not in completed:
            np.save(directory / "counts.npy", values)
            cache.seal(directory, key, [directory / "counts.npy"])
        result[method.name] = values
    return result, {method.name: key for method, _, key, _ in states}


def run(experiment):
    from reporting.comparison import validate
    validate(experiment)
    output = cache.PROJECT / "outputs" / experiment.name
    tables = output / ".cache"
    tables.mkdir(parents=True, exist_ok=True)
    for filename in experiment.outputs:
        (output / filename).unlink(missing_ok=True)
    for name in ("fold_metrics.csv", "predictions.csv.gz", "prediction_summary.csv", "full_metrics.csv",
                 "topk_correlations.csv", "p_value_comparison.csv", "summary.csv", "failures.csv", "artifacts.csv"):
        path = tables / name
        if path.exists():
            path.unlink()
    cache.write_json(tables / "experiment.json", asdict(experiment))
    cache.write_json(tables / "metric_protocol.json", cache.metric_protocol(experiment)[1])
    def status(stage, dataset=None, scope=None, method=None, state="running"):
        cache.write_json(tables / "status.json", dict(state=state, dataset=dataset, scope=scope, method=method, stage=stage))
    fold_rows, oof_rows, full_rows, correlation_rows, failures, references = [], [], [], [], [], []
    for dataset in experiment.datasets:
        try:
            data = load(dataset)
            print(f"{dataset}: samples={len(data.samples)}, classes={data.samples.y.value_counts().to_dict()}, "
                  f"pairs={data.values.shape[1]}, bacteria={len(np.unique(data.pair_bacteria))}, genes={len(np.unique(data.pair_gene))}", flush=True)
            scopes = contexts(data.samples.y.to_numpy(), experiment.evaluation) if "prediction" in experiment.metrics else []
            if "prediction" in experiment.metrics:
                for fold, context in enumerate(scopes):
                    for method in experiment.methods:
                        print(f"{dataset}/{context.scope}: {method.name}", flush=True)
                        update = lambda stage: status(stage, dataset, context.scope, method.name)
                        update("preprocessing")
                        family = prepare_family(data, context, experiment.preprocessing, method, update)
                        update("selection")
                        selected_key, chosen, test_key = selection(data, context, family, method, max(experiment.evaluation.k))
                        update("prediction")
                        metrics, predictions, prediction_key = prediction(data, context, family, selected_key, chosen, experiment.evaluation)
                        references.append(dict(dataset=dataset, scope=context.scope, method=method.name,
                            **family.references, testing=test_key, selection=selected_key, prediction=prediction_key))
                        fold_rows.append(metrics.assign(dataset=dataset, method=method.name, fold=fold, participants=len(data.samples)))
                        oof_rows.append(predictions.assign(dataset=dataset, method=method.name, fold=fold))
            if set(experiment.metrics) - {"prediction"}:
                context = Context("full", np.arange(len(data.samples)), np.array([], int))
                groups = defaultdict(list)
                families, choices = {}, {}
                for method in experiment.methods:
                    print(f"{dataset}/full: {method.name}", flush=True)
                    update = lambda stage: status(stage, dataset, "full", method.name)
                    update("preprocessing")
                    family = prepare_family(data, context, experiment.preprocessing, method, update)
                    update("selection")
                    selected_key, chosen, test_key = selection(data, context, family, method, max(experiment.evaluation.k))
                    references.append(dict(dataset=dataset, scope="full", method=method.name,
                        **family.references, testing=test_key, selection=selected_key))
                    families[family.key] = family
                    groups[(family.key, test_key)].append(method)
                    choices[method.name] = chosen
                for (family_key, _), methods in groups.items():
                    family = families[family_key]
                    status("shuffles" if "shuffled_fdp" in experiment.metrics else "metrics", dataset, "full", ", ".join(m.name for m in methods))
                    counts, count_keys = shuffled_counts(data, family, methods, experiment.evaluation) if "shuffled_fdp" in experiment.metrics else ({}, {})
                    for method in methods:
                        chosen = choices[method.name]
                        size = method.selector.family_size(len(family.ids), family.original_size)
                        row = dict(dataset=dataset, method=method.name, tested_features=size, real_discoveries=len(chosen["native"]))
                        if method.name in counts:
                            row.update(discovery_statistics(counts[method.name], len(chosen["native"]), size))
                            next(r for r in reversed(references) if r["scope"] == "full" and r["method"] == method.name)["shuffles"] = count_keys[method.name]
                        if "correlation" in experiment.metrics:
                            row.update(average_pairwise_absolute_correlation(family.values, chosen["native"]))
                            for k in experiment.evaluation.coarse_k:
                                correlation_rows.append(dict(dataset=dataset, method=method.name, requested_k=k,
                                    **average_pairwise_absolute_correlation(family.values, chosen["order"][:k])))
                        full_rows.append(row)
            del data
        except Exception as error:
            failures.append(dict(dataset=dataset, error=f"{type(error).__name__}: {error}"))
            print(traceback.format_exc(), flush=True)
            status("dataset", dataset, state="failed")
        for filename, rows in (("fold_metrics.csv", fold_rows), ("predictions.csv.gz", oof_rows)):
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(tables / filename, index=False)
        for filename, rows in (("full_metrics.csv", full_rows), ("topk_correlations.csv", correlation_rows), ("failures.csv", failures)):
            if rows:
                pd.DataFrame(rows).to_csv(tables / filename, index=False)
        if references:
            pd.DataFrame(references).to_csv(tables / "artifacts.csv", index=False)
    if fold_rows:
        summarize(pd.concat(fold_rows, ignore_index=True), pd.concat(oof_rows, ignore_index=True), experiment.evaluation).to_csv(
            tables / "prediction_summary.csv", index=False)
    from reporting.comparison import render
    status("reporting")
    render(experiment)
    status("finished", state="incomplete" if failures else "complete")
    return failures
