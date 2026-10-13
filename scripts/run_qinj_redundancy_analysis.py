#!/usr/bin/env python3
"""Isolated QinJ redundancy study; all tuning precedes outer-test evaluation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from research_utils import prevalence_mask, normalize_rows, sqrt_sparse
from scripts.run_method_comparison import bh_family, feature_table, collapsed_average
from scripts.run_bernoulli_correlation_join import aggregate_pair_clusters, JOIN_PYTHON
from feature_selection.run_feature_selection import new_model

OUTPUT = PROJECT / "outputs/qinj_redundancy_analysis"
INPUT = PROJECT / "outputs/preprocessing/QinJ_2012"
BASE = PROJECT / "outputs/method_comparison/real/QinJ_2012"
SEED = 42
THRESHOLD = .80
WORKERS = 8
COMMUNITY_MAX_EDGES = 500_000_000
# Increment when this script changes cached representation construction, not selection.
REPRESENTATION_VERSION = 1
K_VALUES = sorted(set(range(1, 11)) | {20, 35, 50, 60, 75, 85, 100})
GRID = {"filter": [.60, .70, .80, .90], "penalty": [0, .10, .25, .50],
        "merge": [.70, .80, .90], "ridge": [.01, .10, 1, 10],
        "hybrid": [.60, .70, .80, .90], "residual": [0],
        "penalty_mean": [0, .10, .25, .50],
        "penalty_r2": [0, .10, .25, .50],
        "penalty_residual_target": [0, .10, .25, .50]}


def standardized(x):
    """Only a screened sample-by-feature matrix is densified, never raw pairs."""
    a = x.toarray().astype(float)
    a -= a.mean(axis=0)
    norms = np.linalg.norm(a, axis=0)
    return np.divide(a, norms, out=np.zeros_like(a), where=norms > 1e-14), norms > 1e-14


def select(x, effect, mode, parameter, maximum=100, target=None):
    if mode not in ("filter", "penalty", "penalty_mean", "penalty_r2", "penalty_residual_target"):
        raise ValueError(f"Unknown selector: {mode}")
    z, valid = standardized(x)
    selected, events = [], []
    redundancy = np.zeros(x.shape[1])
    available = valid.copy()
    regression = mode in ("penalty_r2", "penalty_residual_target")
    if regression:
        residuals, basis = z.copy(), []
    if mode == "penalty_residual_target":
        target = np.asarray(target, dtype=float)
        if target.shape != (x.shape[0],) or not np.isfinite(target).all():
            raise ValueError("Residual-target penalty requires finite training labels matching rows")
        target = target - target.mean()
        norm = np.linalg.norm(target)
        if norm <= 1e-14:
            raise ValueError("Residual-target penalty requires a nonconstant training target")
        target = target / norm
    if mode == "filter":
        for i in range(x.shape[1]):
            if not valid[i]:
                events.append((i, "constant", None, None))
                continue
            correlations = np.abs(z[:, selected].T @ z[:, i]) if selected else np.array([])
            maximum_r = float(correlations.max()) if len(correlations) else 0.
            if maximum_r >= parameter:
                events.append((i, "correlated", selected[int(correlations.argmax())], maximum_r))
                continue
            selected.append(i)
            events.append((i, "accepted", None, maximum_r))
            if len(selected) == maximum:
                break
    else:
        for _ in range(min(maximum, int(valid.sum()))):
            score = np.where(available, np.abs(effect) - parameter * redundancy, -np.inf)
            i = int(np.argmax(score))  # Stable BH-order tie break.
            selected.append(i)
            events.append((i, "accepted", None, float(redundancy[i])))
            available[i] = False
            if mode == "penalty":
                redundancy = np.maximum(redundancy, np.abs(z.T @ z[:, i]))
            elif mode == "penalty_mean":
                redundancy = (redundancy * (len(selected)-1) + np.abs(z.T @ z[:, i])) / len(selected)
            else:
                direction = z[:, i].copy()
                if basis:
                    q = np.column_stack(basis)
                    for _ in range(2):
                        direction -= q @ (q.T @ direction)
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    direction /= norm
                    basis.append(direction)
                    residuals -= np.outer(direction, direction @ residuals)
                energy = np.clip(np.sum(residuals**2, axis=0), 0, 1)
                if mode == "penalty_r2":
                    redundancy = 1 - energy
                else:
                    # Reward remaining target association, not its absence.
                    conditional = np.divide(np.abs(target @ residuals), np.sqrt(energy),
                                            out=np.zeros_like(energy), where=energy > 1e-12)
                    redundancy = 1 - np.clip(conditional, 0, 1)
    return selected, events


def residual_select(x, pvalues, maximum=100):
    """Lowest p first, then maximize corr(feature, its OLS residual).

    Centering includes the intercept. For unit-norm columns the score is
    sqrt(1-R²); incremental orthogonal projection equals joint OLS regression.
    Exact ties follow candidate order; zero residuals score zero, not NaN.
    """
    z, available = standardized(x)
    energy = available.astype(float)
    selected, events, basis = [], [], []
    for _ in range(min(maximum, int(available.sum()))):
        if not selected:
            i = int(np.argmin(np.where(available, pvalues, np.inf)))
        else:
            i = int(np.argmax(np.where(available, energy, -np.inf)))
        selected.append(i)
        events.append((i, "accepted", None, float(np.sqrt(energy[i]))))
        available[i] = False
        residual = z[:, i].copy()
        if basis:
            q = np.column_stack(basis)
            # Reorthogonalize to keep nearly collinear selections stable.
            for _ in range(2):
                residual -= q @ (q.T @ residual)
        norm = np.linalg.norm(residual)
        if norm > 1e-12:
            direction = residual / norm
            basis.append(direction)
            energy = np.clip(energy - (direction @ z)**2, 0, 1)
            energy[energy < 1e-12] = 0
    return selected, events


def merge_features(train, test, sizes, threshold):
    n = min(100, train.shape[1])
    if n < 2:
        groups = [[i] for i in range(n)]
    else:
        z, _ = standardized(train[:, :n])
        distance = np.clip(1 - z.T @ z, 0, 2)
        np.fill_diagonal(distance, 0)
        labels = fcluster(linkage(squareform(distance, checks=False), method="complete"),
                          1 - threshold, criterion="distance")
        groups = sorted([np.flatnonzero(labels == label).tolist() for label in np.unique(labels)], key=min)
    weights = sparse.lil_matrix((n, len(groups)), dtype=float)
    for j, group in enumerate(groups):
        weights[group, j] = np.asarray(sizes)[group] / np.asarray(sizes)[group].sum()
    weights = weights.tocsr()
    return (sqrt_sparse(train[:, :n].power(2) @ weights),
            sqrt_sparse(test[:, :n].power(2) @ weights), groups)


def diagnostics(x):
    x = x[:, :10]
    z, valid = standardized(x)
    c = z.T @ z
    c[~valid, :] = np.nan
    c[:, ~valid] = np.nan
    n = x.shape[1]
    result = {"top_n": n, "nearest_abs_median": np.nan, "nearest_abs_max": np.nan,
              "nearest_signed_median": np.nan, "high_correlation_pairs": 0,
              "effective_rank": np.nan}
    if valid.sum():
        eig = np.maximum(np.linalg.eigvalsh(c[np.ix_(valid, valid)]), 0)
        prob = eig[eig > 1e-12] / eig.sum()
        result["effective_rank"] = float(np.exp(-np.sum(prob * np.log(prob))))
    if valid.sum() > 1:
        off = c.copy()
        np.fill_diagonal(off, np.nan)
        neighbors = np.nanmax(np.abs(off[valid]), axis=1)
        result.update(nearest_abs_median=float(np.median(neighbors)),
                      nearest_abs_max=float(neighbors.max()),
                      nearest_signed_median=float(np.median(np.nanmax(off[valid], axis=1))),
                      high_correlation_pairs=int(np.sum(np.abs(off[np.triu_indices(n, 1)]) >= .8)))
    return result, c


def screen(train, test, y, ids, level, sizes):
    scores = bh_family(train, [y])
    order = scores["order"][scores["q"][scores["order"]] <= .10]
    table = feature_table(order, np.full(len(ids), level), ids, scores)
    table["pair_count"] = np.asarray(sizes)[order]
    return train[:, order].tocsr(), test[:, order].tocsr(), table


def validate_outer_split(base, train_idx, test_idx, ids):
    found = []
    for name in ("bernoulli_join_graph_split.npz", "bernoulli_join_split.npz"):
        path = base / name
        if not path.exists():
            continue
        with np.load(path) as saved:
            for key, value in (("train_index", train_idx), ("test_index", test_idx), ("feature_ids", ids)):
                if key not in saved or not np.array_equal(saved[key], value):
                    raise ValueError(f"Outer graph cached {key} mismatch: {path}")
        found.append(path)
    if not found:
        raise FileNotFoundError(f"No original Bernoulli split metadata in {base}")
    return found[0]


def representations(matrix, y, pb, train_idx, test_idx, directory, outer=False):
    directory.mkdir(parents=True, exist_ok=True)
    split_file = directory / "split.npz"
    mask, _, cutoff = prevalence_mask(matrix[train_idx], .05, 5)
    ids = np.flatnonzero(mask)
    if split_file.exists():
        with np.load(split_file) as s:
            for key, value in (("train_index", train_idx), ("test_index", test_idx), ("feature_ids", ids)):
                if not np.array_equal(s[key], value):
                    raise ValueError(f"Incompatible cached split: {directory}")
    else:
        np.savez_compressed(split_file, train_index=train_idx, test_index=test_idx,
                            feature_ids=ids, prevalence_threshold=cutoff)
    graph = BASE / "bernoulli_join_graph" if outer else directory / "join"
    if outer:
        validate_outer_split(BASE, train_idx, test_idx, ids)
    integration_path = graph / "integration.json"
    integration = json.loads(integration_path.read_text()) if integration_path.exists() else None
    if integration and integration["effective_threshold"] != THRESHOLD and (outer or (directory / "representations_complete.json").exists()):
        raise ValueError(f"Cached representation threshold differs from fixed {THRESHOLD:.2f}: {directory}")
    if not integration or integration["effective_threshold"] != THRESHOLD:
        failed = graph / "graph/run.json"
        if not outer and failed.exists() and json.loads(failed.read_text()).get("status") == "failed":
            number = 1
            archive = graph.with_name(f"{graph.name}.failed-{number}")
            while archive.exists():
                number += 1
                archive = graph.with_name(f"{graph.name}.failed-{number}")
            graph.rename(archive)
            print(f"Preserved failed join as {archive}; rebuilding this fold only", flush=True)
        command = [str(JOIN_PYTHON), str(PROJECT / "scripts/bernoulli_join_worker.py"),
                   str(INPUT / "X_valid.npz"), str(split_file), str(graph),
                   "--threshold", str(THRESHOLD), "--workers", str(WORKERS),
                   "--memory-gb", "100", "--memory-reserve-gb", "32",
                   "--working-memory-gb", "24", "--community-working-memory-gb", "100",
                   "--community-max-edges", str(COMMUNITY_MAX_EDGES), "--fixed-threshold"]
        print("Exact fold graph:", " ".join(command), flush=True)
        with (directory / "clustering.log").open("a") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    integration = json.loads((graph / "integration.json").read_text())
    if integration["effective_threshold"] != THRESHOLD:
        raise ValueError(f"Fold graph effective threshold {integration['effective_threshold']:.2f} differs from fixed {THRESHOLD:.2f}")
    with np.load(graph / "assignments.npz") as saved:
        if not np.array_equal(saved["feature_ids"], ids):
            raise ValueError("Graph feature IDs do not match fold prevalence")
        labels = saved["clusters"]
    spaces = {}
    if (directory / "representations_complete.json").exists():
        for family in ("bacteria", "join"):
            spaces[family] = (sparse.load_npz(directory / f"{family}_train.npz"),
                              sparse.load_npz(directory / f"{family}_test.npz"),
                              pd.read_csv(directory / f"{family}_features.csv.gz"))
        return spaces, labels, ids
    train = normalize_rows(matrix[train_idx][:, mask])
    test = normalize_rows(matrix[test_idx][:, mask])
    jt, jv, sizes = aggregate_pair_clusters(train, test, labels)
    spaces["join"] = screen(jt, jv, y[train_idx], np.arange(len(sizes)), "pair_cluster", sizes)
    bt, bids = collapsed_average(train, pb[mask], int(pb.max()) + 1)
    bv, _ = collapsed_average(test, pb[mask], int(pb.max()) + 1)
    spaces["bacteria"] = screen(bt, bv, y[train_idx], bids, "bacterium", np.bincount(pb[mask])[bids])
    print(f"{directory.name}: train={len(train_idx)}, validation={len(test_idx)}, "
          f"classes={np.bincount(y[train_idx]).tolist()}, retained_pairs={len(ids)}, "
          f"bacteria={len(bids)}, clusters={len(sizes)}", flush=True)
    for family, (a, b, table) in spaces.items():
        sparse.save_npz(directory / f"{family}_train.npz", a)
        sparse.save_npz(directory / f"{family}_test.npz", b)
        table.to_csv(directory / f"{family}_features.csv.gz", index=False)
    (directory / "representations_complete.json").write_text('{"complete": true}\n')
    return spaces, labels, ids


def construct(spaces, method, parameter, target=None):
    train, test, table = spaces["bacteria" if method == "bacteria" else "join"]
    table = table.copy()
    if method == "hybrid":
        bt, bv, btable = spaces["bacteria"]
        train = sparse.hstack([train, bt], format="csr")
        test = sparse.hstack([test, bv], format="csr")
        table = pd.concat([table, btable], ignore_index=True)
        order = table.sort_values(["q", "p", "level", "feature_id"], kind="stable").index.to_numpy()
        train, test, table = train[:, order], test[:, order], table.iloc[order].reset_index(drop=True)
    groups = [[i] for i in range(min(100, len(table)))]
    events = []
    if method == "residual":
        chosen, events = residual_select(train, table.p.to_numpy())
        groups = [[i] for i in chosen]
        a, b = train[:, chosen], test[:, chosen]
    elif method in ("filter", "hybrid", "penalty", "penalty_mean", "penalty_r2", "penalty_residual_target"):
        chosen, events = select(train, table.effect.to_numpy(),
                                method if method.startswith("penalty") else "filter", parameter, target=target)
        groups = [[i] for i in chosen]
        a, b = train[:, chosen], test[:, chosen]
    elif method == "merge":
        a, b, groups = merge_features(train, test, table.pair_count.to_numpy(), parameter)
    else:
        a, b = train[:, :100], test[:, :100]
    return a, b, table, groups, events


def evaluate(train, test, ytrain, ytest, ks, c=1):
    rows, predictions, cache = [], [], {}
    for k in ks:
        effective = min(k, train.shape[1])
        if effective not in cache:
            probability = np.full(len(ytest), ytrain.mean())
            if effective:
                model = new_model(SEED)
                model.set_params(logisticregression__C=c)
                model.fit(train[:, :effective], ytrain)
                probability = model.predict_proba(test[:, :effective])[:, 1]
            cache[effective] = probability
        probability = cache[effective]
        predicted = probability >= .5
        rows.append(dict(requested_k=k, effective_k=effective,
                         auroc=roc_auc_score(ytest, probability),
                         precision=precision_score(ytest, predicted, zero_division=0),
                         recall=recall_score(ytest, predicted, zero_division=0),
                         f1=f1_score(ytest, predicted, zero_division=0)))
        predictions.extend(dict(requested_k=k, test_position=i, y=int(ytest[i]),
                                probability=float(value)) for i, value in enumerate(probability))
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def cache_signatures():
    # Separate representation compatibility from selector/run provenance.
    sources = [Path(__file__), INPUT / "X_valid.npz", INPUT / "edges.npz", INPUT / "samples.csv",
               PROJECT / "scripts/run_method_comparison.py", PROJECT / "scripts/bernoulli_join_worker.py",
               PROJECT / "scripts/run_bernoulli_correlation_join.py",
               PROJECT / "research_utils.py", PROJECT / "feature_selection/run_feature_selection.py",
               PROJECT.parent / "bernoulli-correlation-join/bernoulli_join.py",
               PROJECT.parent / "bernoulli-correlation-join/community_detection.py"]
    signature = hashlib.sha256(repr([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]).encode()).hexdigest()
    representation_signature = hashlib.sha256(repr((REPRESENTATION_VERSION,
        [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources[1:]])).encode()).hexdigest()
    return signature, representation_signature


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signature, representation_signature = cache_signatures()
    manifest = OUTPUT / "protocol.json"
    migrations = []
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        migrations = saved.get("cache_compatibility_migrations", [])
        compatible = saved.get("representation_signature") == representation_signature
        if (not compatible and saved["signature"] != signature) or any(
                saved[k] != v for k, v in (("seed", SEED), ("threshold", THRESHOLD), ("inner_folds", 3))):
            raise ValueError("Side-analysis representation signature changed; use a new output directory")
    manifest.write_text(json.dumps(dict(signature=signature, representation_signature=representation_signature,
                                        grid=GRID, seed=SEED,
                                        threshold=THRESHOLD, inner_folds=3,
                                        community_max_edges=COMMUNITY_MAX_EDGES, fixed_threshold=True,
                                        cache_compatibility_migrations=migrations,
                                        penalty_definitions={"penalty": "max absolute Pearson correlation",
                                            "penalty_mean": "mean absolute Pearson correlation",
                                            "penalty_r2": "joint OLS R2 with intercept",
                                            "penalty_residual_target": "1 - abs Pearson corr(OLS residual, training target)"}), indent=2))
    matrix = sparse.load_npz(INPUT / "X_valid.npz").tocsr()
    y = pd.read_csv(INPUT / "samples.csv").y.to_numpy(int)
    with np.load(INPUT / "edges.npz") as saved:
        pb, pg = saved["pair_bacteria"], saved["pair_gene"]
    if matrix.shape != (len(y), len(pb)) or len(pg) != len(pb) or set(y) != {0, 1}:
        raise ValueError("Invalid orientation, labels, or pair mapping")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Invalid abundance values")
    outer_train, outer_test = next(StratifiedShuffleSplit(1, test_size=.2, random_state=SEED).split(matrix, y))
    print(f"{INPUT.name}: samples={len(y)}, classes={np.bincount(y).tolist()}, pairs={len(pb)}, "
          f"bacteria={len(np.unique(pb))}, genes={len(np.unique(pg))}", flush=True)
    cv_rows = []
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(outer_train, y[outer_train])):
        train_idx, validation_idx = outer_train[a], outer_train[b]
        assert not np.intersect1d(train_idx, validation_idx).size
        assert not np.intersect1d(np.r_[train_idx, validation_idx], outer_test).size
        directory = OUTPUT / f"fold_{fold}"
        spaces, _, _ = representations(matrix, y, pb, train_idx, validation_idx, directory)
        for method, parameters in {"baseline": [1], "bacteria": [1], **GRID}.items():
            for position, parameter in enumerate(parameters):
                train, test, _, _, _ = construct(spaces, method, parameter, target=y[train_idx])
                scores, _ = evaluate(train, test, y[train_idx], y[validation_idx], range(1, 11), parameter if method == "ridge" else 1)
                geometry, _ = diagnostics(train)
                cv_rows.append(dict(fold=fold, method=method, parameter=parameter, position=position,
                                    mean_auroc=float(scores.auroc.mean()), **geometry))
                scores.to_csv(directory / f"{method}_{parameter}_metrics.csv", index=False)
        pd.DataFrame(cv_rows).to_csv(OUTPUT / "inner_cv.csv", index=False)
    summary = pd.DataFrame(cv_rows).groupby(["method", "parameter", "position"], as_index=False).agg(
        mean_auroc=("mean_auroc", "mean"), nearest_abs_median=("nearest_abs_median", "mean"))
    locked = summary.sort_values(["method", "mean_auroc", "nearest_abs_median", "position"],
                                 ascending=[True, False, True, True], na_position="last").groupby("method", sort=False).head(1)
    locked.to_csv(OUTPUT / "locked_parameters.csv", index=False)
    # Only now access outer-test predictions. Cached outer graph is training-only.
    spaces, labels, ids = representations(matrix, y, pb, outer_train, outer_test, OUTPUT / "outer", outer=True)
    taxa = pd.read_csv(INPUT / "bacteria.txt.gz", header=None)[0].to_numpy()
    metrics, geometries, all_predictions = [], [], []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for row in locked.itertuples():
        train, test, table, groups, events = construct(spaces, row.method, row.parameter, target=y[outer_train])
        scores, predictions = evaluate(train, test, y[outer_train], y[outer_test], K_VALUES,
                                       row.parameter if row.method == "ridge" else 1)
        if row.method in ("baseline", "bacteria"):
            cached = BASE / ("bh_bacteria" if row.method == "bacteria" else "bernoulli_correlation_join")
            reference = pd.read_csv(cached / "predictions.csv.gz")
            check = predictions.merge(reference, on=["requested_k", "test_position"], suffixes=("", "_reference"))
            if check.empty or not np.allclose(check.probability, check.probability_reference, atol=1e-7):
                raise ValueError(f"Baseline reproduction failed: {row.method}")
        metrics.append(scores.assign(method=row.method, parameter=row.parameter))
        all_predictions.append(predictions.assign(method=row.method, sample_index=outer_test[predictions.test_position]))
        geometry, corr = diagnostics(train)
        geometries.append(dict(method=row.method, **geometry))
        destination = OUTPUT / row.method
        destination.mkdir(exist_ok=True)
        sparse.save_npz(destination / "ranked_train.npz", train)
        sparse.save_npz(destination / "ranked_test.npz", test)
        table.to_csv(destination / "candidate_features.csv.gz", index=False)
        event_table = pd.DataFrame(events, columns=["candidate_index", "decision", "blocking_candidate_index", "correlation"])
        if row.method in ("penalty_mean", "penalty_r2", "penalty_residual_target"):
            event_table = event_table.rename(columns={"correlation": "penalty"})
            event_table["abs_mw_effect"] = np.abs(table.effect.to_numpy()[event_table.candidate_index.to_numpy(int)])
            event_table["selection_score"] = event_table.abs_mw_effect - row.parameter * event_table.penalty
            if row.method == "penalty_residual_target":
                event_table["residual_target_abs_correlation"] = np.where(
                    np.arange(len(event_table)) == 0, np.nan, 1 - event_table.penalty)
        event_table.to_csv(destination / "selection_events.csv", index=False)
        records = []
        for rank, group in enumerate(groups, 1):
            members = table.iloc[group]
            bacteria = set()
            for feature in members.itertuples():
                bacteria.update([int(feature.feature_id)] if feature.level == "bacterium" else
                                np.unique(pb[ids[labels == int(feature.feature_id)]]).tolist())
            records.append(dict(rank=rank, member_candidate_indices=json.dumps(group),
                                original_ranks=json.dumps(members["rank"].astype(int).tolist()),
                                member_ids=json.dumps(members.feature_id.astype(int).tolist()),
                                levels=";".join(members.level), pair_count=int(members.pair_count.sum()),
                                prevalence=float(train[:, rank-1].count_nonzero()/train.shape[0]),
                                bacteria=";".join(taxa[sorted(bacteria)])))
        pd.DataFrame(records).to_csv(destination / "selected_features.csv", index=False)
        geometries[-1]["singleton_fraction_top10"] = float(np.mean([r["pair_count"] == 1 for r in records[:10]])) if records else np.nan
        neighbor_rows = []
        for i in range(len(corr)):
            values = corr[i].copy()
            values[i] = np.nan
            finite = np.isfinite(values)
            if finite.any():
                j = int(np.nanargmax(np.abs(values)))
                neighbor_rows.append(dict(rank=i+1, neighbor_rank=j+1, signed_r=float(values[j]), abs_r=float(abs(values[j]))))
            else:
                neighbor_rows.append(dict(rank=i+1, neighbor_rank=None, signed_r=np.nan, abs_r=np.nan))
        pd.DataFrame(neighbor_rows).to_csv(destination / "top10_nearest_neighbors.csv", index=False)
        pd.DataFrame(corr).to_csv(destination / "top10_correlations.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        if corr.size:
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            fig.colorbar(im, ax=ax)
        ax.set_title(f"{INPUT.name}: {row.method}, training Top 10")
        fig.tight_layout(); fig.savefig(destination / "top10_correlations.png", dpi=200); plt.close(fig)
    metrics = pd.concat(metrics, ignore_index=True)
    for reference in ("baseline", "bacteria"):
        reference_auc = metrics[metrics.method == reference].set_index("requested_k").auroc
        metrics[f"auroc_minus_{reference}"] = metrics.auroc - metrics.requested_k.map(reference_auc)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    pd.concat(all_predictions).to_csv(OUTPUT / "predictions.csv.gz", index=False)
    geometry = pd.DataFrame(geometries)
    geometry["nearest_abs_change_vs_baseline"] = geometry.nearest_abs_median - geometry.set_index("method").loc["baseline", "nearest_abs_median"]
    geometry.to_csv(OUTPUT / "geometry.csv", index=False)
    for name, ks in (("fine", list(range(1, 11))), ("coarse", [5,10,20,35,50,60,75,85,100])):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flat, ("auroc", "precision", "recall", "f1")):
            for method, group in metrics[metrics.requested_k.isin(ks)].groupby("method"):
                ax.plot(group.requested_k, group[metric], marker="o", label=method)
            ax.set(xlabel="Requested K", ylabel=metric); ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUTPUT / f"{name}_metrics.png", dpi=200); plt.close(fig)
    (OUTPUT / "complete.json").write_text(json.dumps(dict(signature=signature, status="complete")))
    print(metrics[metrics.requested_k <= 10].groupby("method").auroc.mean().to_string(float_format=lambda x: f"{x:.3f}"), flush=True)


if __name__ == "__main__":
    main()
