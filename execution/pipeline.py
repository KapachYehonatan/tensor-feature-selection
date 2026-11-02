"""Shared fold-local and all-sample execution of compositional methods."""
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
import fcntl
import json
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from datasets.loading import Dataset, load
from execution import cache
from execution.spec import Experiment, Evaluation
from methods import preprocessing as prep, representations as reps
from methods.pipelines import Method, construct
from methods.selectors.mann_whitney import MannWhitney
from metrics.prediction import predict_prefixes, predict_model, point_metrics, new_model, summarize
from metrics.correlation import average_pairwise_absolute_correlation
from metrics.shuffled_fdp import label_permutations, discovery_statistics


@dataclass
class Context:
    """Named analysis scope and disjoint integer training/validation row indices."""
    scope: str
    training: np.ndarray
    validation: np.ndarray


@dataclass
class Family:
    """All-sample, training-defined representation and its stage provenance.

    values has samples×eligible-features shape. ids maps its columns to pair,
    bacterium, gene or learned-cluster IDs; original_size is the complete family
    size before exclusions, not the number of eligible matrix columns.
    """
    key: str
    directory: Path
    values: sparse.csr_matrix
    ids: np.ndarray
    original_size: int
    references: dict


def contexts(y: np.ndarray, evaluation: Evaluation) -> list[Context]:
    """Create identical stratified splits for all methods, checking once-only coverage."""
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


def prepare_pairs(data: Dataset, context: Context, preprocessing: prep.Preprocessing) -> tuple[str, Path, np.ndarray]:
    """Cache one shared training-local prevalence mask for this sample context."""
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
    return key, selection, retained


def prepare_families(data: Dataset, context: Context, experiment: Experiment,
                     notify: Callable) -> Iterator[tuple[Method, Family]]:
    """Yield method/family pairs; reuse identical representations within a context.

    Retain matrices only while another selector still needs the same family.
    """
    prepared = prepare_pairs(data, context, experiment.preprocessing)
    keys = [(m.representation, m.graph, m.clustering) for m in experiment.methods]
    remaining, families = Counter(keys), {}
    for method, key in zip(experiment.methods, keys):
        print(f"{data.name}/{context.scope}: {method.name}", flush=True)
        update = lambda stage: notify(stage, method.name)
        update("preprocessing")
        family = families.get(key)
        if family is None:
            family = prepare_family(data, context, experiment.preprocessing, method, update, prepared)
        if remaining[key] > 1:
            families[key] = family
        yield method, family
        remaining[key] -= 1
        if not remaining[key]:
            families.pop(key, None)


def prepare_family(data: Dataset, context: Context, preprocessing: prep.Preprocessing, method: Method,
                   notify: Callable | None = None, prepared: tuple | None = None) -> Family:
    """Cache preprocessing → optional graph/partition → training-defined representation.

    The returned values include validation rows, using groups and eligibility
    learned exclusively from training. prepared reuses the shared pair mask.
    """
    notify = notify or (lambda stage: None)
    fitting = cache.array_hash(context.training)
    key, selection, retained = prepare_pairs(data, context, preprocessing) if prepared is None else prepared
    partition, partition_key, graph_key = None, None, None
    if method.representation == "join":
        detected = data.values[context.training][:, retained].tocsr()
        detection = dict(indices=cache.array_hash(detected.indices), indptr=cache.array_hash(detected.indptr),
                         shape=detected.shape, ids=cache.array_hash(retained), fit=fitting)
        graph_settings = {k: v for k, v in asdict(method.graph).items() if k not in {"step", "edge_limit"}}
        graph_key, graph_record = cache.identity("graph", detection, graph_settings,
            method.graph.dependencies(), method.graph.source_files(), software=("numpy", "scipy", "numba"))
        graph = cache.stage_path(data.name, context.scope, "graph", graph_key, graph_record)
        cached = cache.complete(graph, graph_key)
        notify("graph")
        print(f"{data.name}/{context.scope}: graph {'cached' if cached else 'building'}; log={graph / 'graph.log'}", flush=True)
        if not cached:
            work = graph / "work"
            manifest = work / "graph/run.json"
            if (work.is_symlink() or (work / "graph").is_symlink() or manifest.is_symlink()
                    or work.resolve().parent != graph.resolve()):
                raise ValueError(f"Refusing to reset graph work outside its stage: {work}")
            state = json.loads(manifest.read_text()) if manifest.exists() else {}
            if state.get("status") == "aborted_resource_limit":
                # Retain the failure reason without keeping a partial graph that consumes its retry's disk reserve.
                cache.write_json(graph / "aborted_run.json", state)
                shutil.rmtree(work)
            method.graph.build(data.path / "X_valid.npz", selection, graph / "work")
            cache.seal(graph, graph_key, [p for p in (graph / "work").rglob("*") if p.is_file()])
        partition_key, partition_record = cache.identity("clustering", dict(graph=graph_key),
            dict(graph_resources=asdict(method.graph), clustering=asdict(method.clustering)),
            method.clustering.dependencies(), method.clustering.source_files(), software=("numpy", "scipy", "numba", "igraph"))
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


def selection(data: Dataset, context: Context, family: Family, method: Method,
              rank_limit: int) -> tuple[str, dict, str]:
    """Cache training evidence/selection; return selection key, column positions and test key."""
    selector = method.selector
    key, record = cache.identity("testing", dict(values=cache.sparse_hash(family.values[context.training]), ids=cache.array_hash(family.ids),
        labels=cache.array_hash(data.samples.y.to_numpy()[context.training])), dict(kind=selector.kind, settings=selector.test_settings()),
        selector.dependencies(), software=("numpy", "scipy"))
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


def prediction(data: Dataset, context: Context, family: Family, selected_key: str,
               selected: dict, evaluation: Evaluation) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate training-ranked prefixes on validation rows; return metrics, OOF rows and key."""
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


@dataclass
class ShuffleState:
    """One method's fixed-budget counts; -1 marks an unfinished label shuffle."""
    method: Method
    directory: Path
    key: str
    counts: np.ndarray
    complete: bool

    def save(self, *, final: bool = False) -> None:
        if self.complete:
            return
        destination = self.directory if final else self.directory / "checkpoint"
        destination.mkdir(exist_ok=True)
        temporary = destination / "counts.tmp.npy"
        np.save(temporary, self.counts)
        temporary.replace(destination / "counts.npy")
        cache.seal(destination, self.key, [destination / "counts.npy"])
        self.complete = final


def load_shuffle_state(data: Dataset, family: Family, method: Method, evaluation: Evaluation) -> ShuffleState:
    """Validate and resume this method's full-data discovery counts."""
    selector = method.selector
    key, record = cache.identity("shuffles", dict(family=family.key, labels=cache.array_hash(data.samples.y.to_numpy())),
        dict(selector=asdict(selector), budget=evaluation.shuffles, seed=evaluation.seed),
        (*selector.dependencies(), *selector.selection_dependencies(), label_permutations), software=("numpy", "scipy"))
    directory = cache.stage_path(data.name, "full", "shuffles", key, record)
    complete = cache.complete(directory, key)
    checkpoint = directory if complete else directory / "checkpoint"
    counts = (np.load(checkpoint / "counts.npy") if complete or cache.complete(checkpoint, key)
              else np.full(evaluation.shuffles, -1, dtype=int))
    family_size = selector.family_size(len(family.ids), family.original_size)
    if (counts.shape != (evaluation.shuffles,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < (-1 if not complete else 0)) or np.any(counts > family_size)):
        raise ValueError("Invalid shuffle checkpoint")
    return ShuffleState(method, directory, key, counts, complete)


def shuffled_counts(data: Dataset, family: Family, methods: list[Method], evaluation: Evaluation) -> tuple[dict, dict]:
    """Return counts and stage keys by method, using one shared label/evidence batch.

    Every method receives the same permutations of the full-data participant
    labels. A completed baseline count of zero certifies zero for a masked-BY
    filter only when its correction, including alpha, is identical.
    """
    targets = label_permutations(data.samples.y.to_numpy(), data.name + "/full_data", evaluation.shuffles, evaluation.seed)
    states = [load_shuffle_state(data, family, method, evaluation) for method in methods]
    selector = methods[0].selector
    if not all(state.complete for state in states):
        prepared = selector.prepare(family.values)
        baselines = {state.method.selector.correction: state for state in states
                     if isinstance(state.method.selector, MannWhitney)
                     and state.method.selector.correlation_threshold is None}
        for start in range(0, evaluation.shuffles, 8):
            indices = []
            for index in range(start, min(start + 8, evaluation.shuffles)):
                pending = [state for state in states if state.counts[index] < 0]
                for state in pending:
                    baseline = baselines.get(state.method.selector.correction)
                    if baseline is not None and baseline.counts[index] == 0:
                        state.counts[index] = 0
                if any(state.counts[index] < 0 for state in pending):
                    indices.append(index)
            if indices:
                vectors = selector.batch_test(family.values, targets[indices], prepared)
                for index, vector in zip(indices, vectors):
                    evidence = {selector.kind: vector}
                    for state in states:
                        if state.counts[index] >= 0:
                            continue
                        baseline = baselines.get(state.method.selector.correction)
                        if baseline is not None and baseline.counts[index] == 0:
                            state.counts[index] = 0
                        else:
                            selected = state.method.selector.select(evidence, family.values, family.ids, family.original_size, 0)
                            state.counts[index] = len(selected["native"])
            for state in states:
                state.save()
            print(f"{data.name}: shuffled {min(start + 8, evaluation.shuffles)}/{evaluation.shuffles}", flush=True)
        for state in states:
            size = state.method.selector.family_size(len(family.ids), family.original_size)
            if np.any(state.counts < 0) or np.any(state.counts > size):
                raise ValueError("Incomplete or invalid shuffled counts")
            state.save(final=True)
    return ({state.method.name: state.counts for state in states},
            {state.method.name: state.key for state in states})


def run(experiment: Experiment) -> list[dict]:
    """Serialize scientific writers: different experiments share expensive stage files."""
    cache.CACHE.mkdir(parents=True, exist_ok=True)
    with (cache.CACHE / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another scientific experiment is running; retry after it finishes") from error
        return _run(experiment)


def _run(experiment: Experiment) -> list[dict]:
    """Execute an Experiment, publish complete exports, and return dataset failure records."""
    from reporting.comparison import render, validate
    validate(experiment)
    output = cache.PROJECT / "outputs" / experiment.name
    saved = output / ".cache"
    saved.mkdir(parents=True, exist_ok=True)
    retained = (saved / "experiment.json").exists()
    # Failed resumptions must not replace earlier validated results with partial tables.
    with tempfile.TemporaryDirectory(prefix=".tables_", dir=output) as temporary:
        staged = Path(temporary)
        tables = staged / ".cache"
        phase = "tables"
        try:
            tables.mkdir()
            failures = run_tables(experiment, tables)
            if failures:
                cache.write_json(saved / "status.json", dict(state="incomplete", stage="finished", failures=failures,
                                                            retained_previous_results=retained))
                return failures
            phase = "reporting"
            cache.write_json(saved / "status.json", dict(state="running", stage=phase))
            render(experiment, output=staged)
        except Exception as error:
            cache.write_json(saved / "status.json", dict(state="failed", stage=phase,
                error=f"{type(error).__name__}: {error}", retained_previous_results=retained))
            raise
        for name in ("fold_metrics.csv", "predictions.csv.gz", "prediction_summary.csv", "full_metrics.csv",
                     "topk_correlations.csv", "p_value_comparison.csv", "summary.csv", "failures.csv", "artifacts.csv"):
            if (tables / name).exists():
                (tables / name).replace(saved / name)
            elif (saved / name).exists():
                (saved / name).unlink()
        for name in experiment.outputs:
            (staged / name).replace(output / name)
        for name in ("experiment.json", "metric_protocol.json"):
            (tables / name).replace(saved / name)
    cache.write_json(saved / "status.json", dict(state="complete", stage="finished"))
    return []


def run_cv(data: Dataset, experiment: Experiment, status: Callable) -> tuple[list, list, list]:
    """Run training-local selection/prediction; return fold frames, OOF frames and stage references."""
    fold_rows, predictions, references = [], [], []
    for fold, context in enumerate(contexts(data.samples.y.to_numpy(), experiment.evaluation)):
        notify = lambda stage, name: status(stage, data.name, context.scope, name)
        for method, family in prepare_families(data, context, experiment, notify):
            notify("selection", method.name)
            selection_key, selected, testing_key = selection(data, context, family, method, max(experiment.evaluation.k))
            notify("prediction", method.name)
            metrics, probabilities, prediction_key = prediction(data, context, family, selection_key, selected, experiment.evaluation)
            references.append(dict(dataset=data.name, scope=context.scope, method=method.name,
                **family.references, testing=testing_key, selection=selection_key, prediction=prediction_key))
            fold_rows.append(metrics.assign(dataset=data.name, method=method.name, fold=fold, participants=len(data.samples)))
            predictions.append(probabilities.assign(dataset=data.name, method=method.name, fold=fold))
    return fold_rows, predictions, references


def run_full_statistics(data: Dataset, experiment: Experiment, status: Callable) -> tuple[list, list, list]:
    """Select on all samples; return native-set metrics, score-ranked correlations and references.

    These representations never enter CV. Share shuffled evidence only between
    methods with the same feature family and testing-stage identity.
    """
    context = Context("full", np.arange(len(data.samples)), np.array([], int))
    groups, families, choices, references = defaultdict(list), {}, {}, {}
    notify = lambda stage, name: status(stage, data.name, "full", name)
    for method, family in prepare_families(data, context, experiment, notify):
        notify("selection", method.name)
        selection_key, selected, testing_key = selection(data, context, family, method, max(experiment.evaluation.k))
        references[method.name] = dict(dataset=data.name, scope="full", method=method.name,
            **family.references, testing=testing_key, selection=selection_key)
        families[family.key] = family
        groups[(family.key, testing_key)].append(method)
        choices[method.name] = selected
    full_rows, correlations = [], []
    for (family_key, _), methods in groups.items():
        family = families[family_key]
        stage = "shuffles" if "shuffled_fdp" in experiment.metrics else "metrics"
        notify(stage, ", ".join(method.name for method in methods))
        counts, count_keys = (shuffled_counts(data, family, methods, experiment.evaluation)
                              if "shuffled_fdp" in experiment.metrics else ({}, {}))
        for method in methods:
            selected = choices[method.name]
            size = method.selector.family_size(len(family.ids), family.original_size)
            row = dict(dataset=data.name, method=method.name, tested_features=size, real_discoveries=len(selected["native"]))
            if method.name in counts:
                row.update(discovery_statistics(counts[method.name], len(selected["native"]), size))
                references[method.name]["shuffles"] = count_keys[method.name]
            if "correlation" in experiment.metrics:
                row.update(average_pairwise_absolute_correlation(family.values, selected["native"]))
                for k in experiment.evaluation.coarse_k:
                    correlations.append(dict(dataset=data.name, method=method.name, requested_k=k,
                        **average_pairwise_absolute_correlation(family.values, selected["order"][:k])))
            full_rows.append(row)
    return full_rows, correlations, list(references.values())


def write_tables(directory: Path, frames: dict, records: dict) -> None:
    """Checkpoint private export buffers; frame lists concatenate rows, record lists become tables."""
    for filename, rows in frames.items():
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(directory / filename, index=False)
    for filename, rows in records.items():
        if rows:
            pd.DataFrame(rows).to_csv(directory / filename, index=False)


def run_tables(experiment: Experiment, tables: Path) -> list[dict]:
    """Build private replacement tables; publish nothing and return dataset failures."""
    output = cache.PROJECT / "outputs" / experiment.name
    cache.write_json(tables / "experiment.json", asdict(experiment))
    cache.write_json(tables / "metric_protocol.json", cache.metric_protocol(experiment)[1])

    def status(stage, dataset=None, scope=None, method=None, state="running"):
        cache.write_json(output / ".cache/status.json",
                         dict(state=state, dataset=dataset, scope=scope, method=method, stage=stage))

    fold_rows, predictions, full_rows, correlations, failures, references = [], [], [], [], [], []
    for dataset in experiment.datasets:
        try:
            data = load(dataset)
            print(f"{dataset}: samples={len(data.samples)}, classes={data.samples.y.value_counts().to_dict()}, "
                  f"pairs={data.values.shape[1]}, bacteria={len(np.unique(data.pair_bacteria))}, genes={len(np.unique(data.pair_gene))}", flush=True)
            if "prediction" in experiment.metrics:
                folds, oof, stages = run_cv(data, experiment, status)
                fold_rows.extend(folds)
                predictions.extend(oof)
                references.extend(stages)
            if set(experiment.metrics) - {"prediction"}:
                metrics, redundancy, stages = run_full_statistics(data, experiment, status)
                full_rows.extend(metrics)
                correlations.extend(redundancy)
                references.extend(stages)
            del data
        except Exception as error:
            failures.append(dict(dataset=dataset, error=f"{type(error).__name__}: {error}"))
            print(traceback.format_exc(), flush=True)
            status("dataset", dataset, state="failed")
        write_tables(tables, {"fold_metrics.csv": fold_rows, "predictions.csv.gz": predictions},
            {"full_metrics.csv": full_rows, "topk_correlations.csv": correlations,
             "failures.csv": failures, "artifacts.csv": references})
    if fold_rows:
        summarize(pd.concat(fold_rows, ignore_index=True), pd.concat(predictions, ignore_index=True),
                  experiment.evaluation).to_csv(tables / "prediction_summary.csv", index=False)
    return failures
