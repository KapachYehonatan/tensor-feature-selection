#!/usr/bin/env python3
"""Training-selected penalties on cached join means only; never builds graphs."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_qinj_redundancy_analysis import evaluate, diagnostics, standardized, new_model, SEED, K_VALUES
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t as student_t
from scipy.special import ndtri
from sklearn.model_selection import StratifiedKFold

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/batch_1"
PRIMARY_K = tuple(range(5, 11))
GRID = {"effect_lcb": [0, 1, 2], "support": [0, .1, .25, .5],
        "concentration": [0, .1, .25, .5], "cluster_size": [0, .05, .1, .25],
        "partition_instability": [0, .25, .5, 1], "within_class_corr": [0, .1, .25, .5]}
COMPOSITION_FAMILIES = {"gene_diverse_size", "homogeneous_size"}
RANK_FAMILIES = {"bh_size_rank_blend", "bh_size_rank_cap", "bh_coverage_rank_blend"}
SOURCE_FAMILIES = {"bh_source_overlap", "size_plus_source"}
CATALOGUE_FAMILIES = {"catalogue_remaining_power", "catalogue_completion_power"}
COVERAGE_FAMILIES = {"taxon_coverage", "size_taxon_coverage", "bh_log_coverage", "effect_log_coverage", "coverage_power", "coverage_support_power", "bh_coverage_rank_blend", "coverage_logistic_gain", "coverage_uncertainty_power", "coverage_alignment_power", "coverage_snr_power", "bh_coverage_power", "coverage_partition_power"}
STABILITY_FAMILIES = {"l1_stability", "size_l1_stability"}
STABILITY_REPEATS = 40
STABILITY_FRACTION = .5
UNCERTAINTY_Z = 1.96


def mw_effect(values, target):
    n1, n0 = int(np.sum(target == 1)), int(np.sum(target == 0))
    ranks = rankdata(values, axis=0, method="average")
    return 2*(ranks[target == 1].sum(axis=0)-n1*(n1+1)/2)/(n1*n0)-1


def resampled_l1_frequency(train, target, c, repeats=STABILITY_REPEATS, seed=SEED):
    """Training-only half-sample selection frequency, not an error guarantee."""
    target = np.asarray(target)
    if target.shape != (train.shape[0],) or set(np.unique(target)) != {0,1}:
        raise ValueError("Resampling requires aligned binary training labels")
    if not np.isfinite(c) or c <= 0 or repeats < 1 or not np.isfinite(train.data).all():
        raise ValueError("Invalid stability-scoring inputs")
    frequency = np.zeros(train.shape[1])
    if not train.shape[1]:return frequency
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(target == label) for label in (0,1)]
    for repeat in range(repeats):
        subset = np.sort(np.concatenate([rng.choice(ids, max(1,int(len(ids)*STABILITY_FRACTION)),
                                                    replace=False) for ids in classes]))
        scorer = new_model(SEED)
        scorer[-1].set_params(l1_ratio=1.0, C=c)
        scaled = scorer[0].fit_transform(train[subset])
        valid = scorer[0].var_ > 1e-28
        active = np.zeros(train.shape[1],dtype=bool)
        if valid.any():
            scorer[-1].fit(scaled[:,valid], target[subset])
            active[valid] = np.abs(scorer[-1].coef_[0]) > 1e-8
        frequency += active
        if (repeat+1) % 10 == 0:
            print(f"Stability C={c:.3f}: resamples={repeat+1}/{repeats}, active={int(active.sum())}",flush=True)
    return frequency/repeats


def effective_sample_support(train):
    """Unit-invariant participation fraction from sparse training predictors."""
    if not sparse.issparse(train) or train.shape[0] < 1:
        raise ValueError("Effective support requires nonempty sparse training rows")
    values = train.astype(float, copy=False)
    if not np.isfinite(values.data).all() or np.any(values.data < 0):
        raise ValueError("Invalid training abundances for effective support")
    totals = np.asarray(values.sum(axis=0)).ravel()
    energy = np.asarray(values.multiply(values).sum(axis=0)).ravel()
    fraction = np.divide(totals**2, values.shape[0]*energy, out=np.zeros(values.shape[1]), where=energy > 0)
    return np.clip(fraction, 0, 1)


def penalty_statistics(train, target, table):
    values = train.toarray().astype(float)
    n, p = values.shape
    if len(target) != n or set(np.unique(target)) != {0,1}:
        raise ValueError("Statistics require aligned binary training labels")
    if not np.isfinite(values).all() or np.any(values < 0) or len(table) != p or np.any(table.pair_count.to_numpy() <= 0):
        raise ValueError("Invalid cached abundance representation")
    columns = ["effect_lcb", "support", "concentration", "cluster_size", "partition_instability",
               "linear_alignment", "bh_linear_alignment"]
    if not p:
        return pd.DataFrame({column: [] for column in columns})
    n1, n0 = int(np.sum(target == 1)), int(np.sum(target == 0))
    ranks = rankdata(values, axis=0, method="average")
    placements1 = (ranks[target == 1]-rankdata(values[target == 1], axis=0))/n0
    placements0 = (ranks[target == 0]-rankdata(values[target == 0], axis=0))/n1
    se = 2*np.sqrt(placements1.var(axis=0, ddof=1)/n1 + placements0.var(axis=0, ddof=1)/n0)
    effect = table.effect.to_numpy()
    np.testing.assert_allclose(mw_effect(values, target), effect, atol=1e-10)
    effective_support = effective_sample_support(train)
    partitions = []
    for _, chunk in StratifiedKFold(3, shuffle=True, random_state=SEED).split(values, target):
        partitions.append(np.sign(effect)*mw_effect(values[chunk], target[chunk]))
    centered=values-values.mean(axis=0)
    yc=target-target.mean()
    denominator=np.linalg.norm(centered,axis=0)*np.linalg.norm(yc)
    pearson=np.divide(centered.T@yc,denominator,out=np.zeros(p),where=denominator>1e-14)
    aligned=np.maximum(0,np.sign(effect)*pearson)
    tstat=aligned*np.sqrt((n-2)/np.maximum(1-aligned**2,1e-12))
    continuous_p=np.maximum(2*student_t.sf(tstat,n-2),np.finfo(float).tiny)
    rank_evidence=-np.log10(np.maximum(table.p.to_numpy(float),np.finfo(float).tiny))
    return pd.DataFrame({"effect_lcb": se,
        "support": 1-np.sqrt(np.mean(values > 0, axis=0)),
        "concentration": 1-effective_support,
        "cluster_size": 1/np.sqrt(table.pair_count.to_numpy()),
        "partition_instability": np.maximum(0, np.abs(effect)-np.min(partitions, axis=0)),
        "linear_alignment": np.maximum(0,np.abs(effect)-aligned),
        "bh_linear_alignment": np.maximum(0,rank_evidence+np.log10(continuous_p))})


def logistic_conditional_gains(train, target, selected):
    """Ridge-adjusted one-step logistic gains conditional on selected columns."""
    values=train.toarray().astype(float)
    scale=values.std(axis=0)
    values=np.divide(values,scale,out=np.zeros_like(values),where=scale>1e-14)
    if selected:
        model=new_model(SEED)
        model.fit(train[:,selected],target)
        probability=model.predict_proba(train[:,selected])[:,1]
        coefficient=np.r_[model[-1].intercept_,model[-1].coef_.ravel()]
    else:
        # Balanced intercept-only null has probability .5 and coefficient zero.
        probability=np.full(len(target),.5)
        coefficient=np.zeros(1)
    class_weights=len(target)/(2*np.bincount(target))
    gradient=class_weights[target]*(target-probability)
    curvature=class_weights[target]*probability*(1-probability)
    design=np.column_stack([np.ones(len(target)),values[:,selected]])
    hessian=design.T@(curvature[:,None]*design)+np.eye(design.shape[1])
    cross=design.T@(curvature[:,None]*values)
    nuisance_gradient=design.T@gradient-coefficient
    efficient_gradient=values.T@gradient-cross.T@np.linalg.solve(hessian,nuisance_gradient)
    information=(np.sum(curvature[:,None]*values**2,axis=0)+1
                 -np.sum(cross*np.linalg.solve(hessian,cross),axis=0))
    return efficient_gradient**2/np.maximum(information,1e-12)


def source_weights_from_assignments(pair_bacteria,ids,labels,table):
    """Pair-count fractions by source taxon; metadata only, never predictors."""
    if not len(table):return sparse.csr_matrix((0,0))
    ids,labels=np.asarray(ids),np.asarray(labels)
    features=table.feature_id.to_numpy(int)
    if ids.shape!=labels.shape or np.any(ids<0) or np.any(ids>=len(pair_bacteria)) or np.any(labels<0):
        raise ValueError("Invalid cached source assignments")
    if len(np.unique(features))!=len(features) or np.any(features<0) or np.any(features>labels.max()):
        raise ValueError("Invalid screened cluster IDs")
    lookup=np.full(int(labels.max())+1,-1,dtype=int);lookup[features]=np.arange(len(features))
    row=lookup[labels];keep=row>=0
    source=np.asarray(pair_bacteria)[ids[keep]]
    if np.any(source<0):raise ValueError("Invalid taxon IDs")
    counts=sparse.coo_matrix((np.ones(keep.sum()),(row[keep],source)),
        shape=(len(table),int(np.max(pair_bacteria))+1)).tocsr()
    size=np.asarray(counts.sum(axis=1)).ravel()
    if not np.array_equal(size,table.pair_count.to_numpy(float)):
        raise ValueError("Source pair counts do not match screened predictors")
    return sparse.diags(1/size)@counts


def taxon_coverage_from_assignments(pair_bacteria, ids, labels, table):
    """Member-weighted fraction of each source's retained pair catalogue."""
    if not len(table):return np.array([],dtype=float)
    weights=source_weights_from_assignments(pair_bacteria,ids,labels,table)
    totals=np.bincount(np.asarray(pair_bacteria)[ids],minlength=weights.shape[1])
    inverse=np.divide(1.,totals,out=np.zeros(len(totals)),where=totals>0)
    coverage=table.pair_count.to_numpy(float)*np.asarray(weights.power(2)@inverse).ravel()
    if not np.isfinite(coverage).all() or np.any((coverage<=0)|(coverage>1+1e-12)):
        raise ValueError("Invalid cached source coverage")
    return np.clip(coverage,0,1)


def catalogue_fractions_from_assignments(pair_bacteria, ids, labels, table):
    """Taxon fractions within each cluster and catalogue fractions it covers."""
    weights = source_weights_from_assignments(pair_bacteria, ids, labels, table)
    if not len(table):return weights, weights.copy()
    totals = np.bincount(np.asarray(pair_bacteria)[ids], minlength=weights.shape[1])
    inverse = np.divide(1., totals, out=np.zeros(len(totals)), where=totals > 0)
    catalogue = sparse.diags(table.pair_count.to_numpy(float)) @ weights @ sparse.diags(inverse)
    return weights, catalogue.tocsr()


def catalogue_metadata(dataset, name, table):
    if not len(table):return sparse.csr_matrix((0, 0)), sparse.csr_matrix((0, 0))
    root = cached_root(dataset)
    graph = PROJECT / "outputs/method_comparison/real" / dataset / "bernoulli_join_graph" if name == "outer" else root / name / "join"
    with np.load(graph / "assignments.npz") as data:ids, labels = data["feature_ids"], data["clusters"]
    with np.load(root / name / "split.npz") as data:
        if not np.array_equal(ids, data["feature_ids"]):raise ValueError("Catalogue graph differs from training prevalence IDs")
    with np.load(PROJECT / "outputs/preprocessing" / dataset / "edges.npz") as data:
        return catalogue_fractions_from_assignments(data["pair_bacteria"], ids, labels, table)


def select_remaining_catalogue(effect, weights, catalogue, power, available=None, maximum=100, mode="remaining"):
    """Greedy count-only marginal coverage, independent of benchmark history."""
    if mode not in ("remaining", "completion"):raise ValueError("Unknown catalogue selection mode")
    effect = np.asarray(effect, dtype=float)
    weights, catalogue = sparse.csr_matrix(weights), sparse.csr_matrix(catalogue)
    if effect.ndim != 1 or weights.shape != catalogue.shape or weights.shape[0] != len(effect):
        raise ValueError("Unaligned catalogue inputs")
    if not np.isfinite(power) or power < 0 or not np.isfinite(effect).all():
        raise ValueError("Invalid catalogue effects/power")
    for matrix in (weights, catalogue):
        if not np.isfinite(matrix.data).all() or np.any((matrix.data < 0)|(matrix.data > 1+1e-12)):
            raise ValueError("Invalid catalogue fractions")
    if len(effect) and (not np.allclose(np.asarray(weights.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-12)
                        or np.any(np.asarray(catalogue.sum(axis=0)).ravel() > 1+1e-12)):
        raise ValueError("Catalogue fractions do not conserve original members")
    available = np.ones(len(effect), dtype=bool) if available is None else np.asarray(available, dtype=bool).copy()
    if available.shape != effect.shape or maximum < 0:
        raise ValueError("Invalid availability or feature budget")
    marginal_weights = weights.multiply(catalogue).tocsr()
    initial = np.asarray(marginal_weights.sum(axis=1)).ravel()
    if np.any(initial <= 0):raise ValueError("A candidate has no original catalogue coverage")
    covered = np.zeros(weights.shape[1])
    chosen, losses = [], []
    association = np.abs(effect)
    for _ in range(min(maximum, int(available.sum()))):
        marginal = np.clip(np.asarray(marginal_weights @ (1-covered)).ravel() if mode == "remaining" else
                           initial+np.asarray(weights @ covered).ravel(), 0, 1)
        score = association*np.power(marginal, power)
        i = int(np.argmax(np.where(available, score, -np.inf)))
        chosen.append(i);losses.append(float(association[i]-score[i]));available[i] = False
        covered = np.clip(covered+catalogue[i].toarray().ravel(), 0, 1)
    return chosen, np.asarray(losses)


def coverage_statistics(dataset,name,table):
    if not len(table):return pd.DataFrame(dict(source_coverage=[],taxon_coverage=[],bh_log_coverage=[],effect_log_coverage=[]))
    root=cached_root(dataset)
    graph=PROJECT/"outputs/method_comparison/real"/dataset/"bernoulli_join_graph" if name=="outer" else root/name/"join"
    with np.load(graph/"assignments.npz") as data:ids,labels=data['feature_ids'],data['clusters']
    with np.load(root/name/"split.npz") as data:
        if not np.array_equal(ids,data['feature_ids']):raise ValueError("Coverage graph differs from cached prevalence IDs")
    with np.load(PROJECT/"outputs/preprocessing"/dataset/"edges.npz") as data:
        coverage=taxon_coverage_from_assignments(data['pair_bacteria'],ids,labels,table)
    return pd.DataFrame(dict(source_coverage=coverage,taxon_coverage=1-np.sqrt(coverage),
                             bh_log_coverage=-np.log10(coverage),effect_log_coverage=-np.log10(coverage)))


def bh_log_coverage_scores(p_values, coverage, strength):
    """Standalone ranking score from one dataset's training p-values/counts."""
    p_values, coverage = np.asarray(p_values, dtype=float), np.asarray(coverage, dtype=float)
    if p_values.ndim != 1 or coverage.shape != p_values.shape:
        raise ValueError("Unaligned coverage and p-values")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Coverage discount must be nonnegative and finite")
    if not np.isfinite(p_values).all() or np.any((p_values < 0)|(p_values > 1)):
        raise ValueError("Invalid training p-values")
    if not np.isfinite(coverage).all() or np.any((coverage <= 0)|(coverage > 1)):
        raise ValueError("Invalid training source coverage")
    penalty = -np.log10(coverage)
    # Ranking heuristic only: this is not a recalibrated p-value or BH test.
    score = -np.log10(np.maximum(p_values, np.finfo(float).tiny))-strength*penalty
    return score, penalty


def coverage_power_scores(effect, coverage, power):
    """Standalone proportional reliability score; loss is already discounted."""
    effect, coverage = np.asarray(effect, dtype=float), np.asarray(coverage, dtype=float)
    if effect.ndim != 1 or coverage.shape != effect.shape:
        raise ValueError("Unaligned effects and coverage")
    if not np.isfinite(power) or power < 0 or not np.isfinite(effect).all():
        raise ValueError("Invalid effects or coverage power")
    if not np.isfinite(coverage).all() or np.any((coverage <= 0)|(coverage > 1)):
        raise ValueError("Invalid training source coverage")
    association = np.abs(effect)
    score = association*np.power(coverage, power)
    return score, association-score


def bh_coverage_power_scores(p_values, coverage, power):
    """Discount cached statistical evidence; scores are not new p-values."""
    evidence, _ = bh_log_coverage_scores(p_values, coverage, 0)
    return coverage_power_scores(evidence, coverage, power)


def coverage_support_power_scores(effect, coverage, support, power):
    """Structural coverage and effective sample support, both training-only."""
    score, _ = coverage_power_scores(effect, coverage, power)
    support = np.asarray(support, dtype=float)
    if support.shape != score.shape or not np.isfinite(support).all() or np.any((support < 0)|(support > 1)):
        raise ValueError("Invalid training effective support")
    score *= np.power(support, power)
    return score, np.abs(np.asarray(effect, dtype=float))-score


def coverage_uncertainty_scores(effect, se, coverage, power, z=UNCERTAINTY_Z):
    """Heuristic uncertainty shrinkage; not a calibrated confidence guarantee."""
    effect, se = np.asarray(effect, dtype=float), np.asarray(se, dtype=float)
    if se.shape != effect.shape or not np.isfinite(se).all() or np.any(se < 0):
        raise ValueError("Invalid training effect uncertainty")
    if not np.isfinite(z) or z < 0:
        raise ValueError("Invalid uncertainty coefficient")
    shrunken = np.maximum(0, np.abs(effect)-z*se)
    score, _ = coverage_power_scores(shrunken, coverage, power)
    return score, np.abs(effect)-score


def coverage_partition_scores(effect, instability, coverage, power):
    """Clip to the weakest training-subset effect, not a confidence bound."""
    return coverage_uncertainty_scores(effect, instability, coverage, power, z=1)


def coverage_snr_scores(effect, se, coverage, power):
    """Smooth signal-to-noise heuristic, not a posterior or confidence bound."""
    effect, se = np.asarray(effect, dtype=float), np.asarray(se, dtype=float)
    if se.shape != effect.shape or not np.isfinite(se).all() or np.any(se < 0):
        raise ValueError("Invalid training effect uncertainty")
    magnitude = np.abs(effect)
    denominator = np.hypot(magnitude, se)
    reliability = np.divide(magnitude, denominator, out=np.zeros_like(magnitude), where=denominator > 0)**2
    score, _ = coverage_power_scores(magnitude*reliability, coverage, power)
    return score, magnitude-score


def coverage_alignment_scores(effect, misalignment, coverage, power):
    """Discount rank effects incompatible with continuous training direction."""
    effect, misalignment = np.asarray(effect, dtype=float), np.asarray(misalignment, dtype=float)
    if misalignment.shape != effect.shape or not np.isfinite(misalignment).all() or np.any(misalignment < 0):
        raise ValueError("Invalid training continuous misalignment")
    if np.any(misalignment > np.abs(effect)+1e-12):
        raise ValueError("Misalignment exceeds the original effect")
    aligned = np.maximum(0, np.abs(effect)-misalignment)
    score, _ = coverage_power_scores(aligned, coverage, power)
    return score, np.abs(effect)-score


def source_weights(dataset,name,table):
    if not len(table):return sparse.csr_matrix((0,0))
    root=cached_root(dataset)
    graph=(PROJECT/"outputs/method_comparison/real"/dataset/"bernoulli_join_graph"
           if name=="outer" else root/name/"join")
    with np.load(PROJECT/"outputs/preprocessing"/dataset/"edges.npz") as data:
        pair_bacteria=data["pair_bacteria"]
    with np.load(graph/"assignments.npz") as data:
        ids,labels=data["feature_ids"],data["clusters"]
    with np.load(root/name/"split.npz") as data:
        if not np.array_equal(ids,data["feature_ids"]):raise ValueError("Source graph differs from cached prevalence IDs")
    return source_weights_from_assignments(pair_bacteria,ids,labels,table)


def composition_penalties(sizes, taxon_weights, gene_weights):
    """Discount repeated genes/mixed taxa without changing any predictor."""
    sizes = np.asarray(sizes, dtype=float)
    if taxon_weights.shape[0] != len(sizes) or gene_weights.shape[0] != len(sizes) or np.any(sizes <= 0):
        raise ValueError("Invalid composition dimensions/pair counts")
    if not len(sizes):
        return pd.DataFrame({name: [] for name in (
            "taxon_concentration", "effective_genes", "gene_diverse_size", "homogeneous_size")})
    for weights in (taxon_weights, gene_weights):
        if not np.isfinite(weights.data).all() or np.any(weights.data < 0):
            raise ValueError("Invalid membership fractions")
        np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), 1, atol=1e-12)
    concentration = np.asarray(taxon_weights.power(2).sum(axis=1)).ravel()
    effective_genes = 1/np.asarray(gene_weights.power(2).sum(axis=1)).ravel()
    if np.any(effective_genes > sizes+1e-8) or np.any(effective_genes < 1-1e-8):
        raise ValueError("Gene diversity exceeds original pair support")
    size_penalty = 1/np.sqrt(sizes)
    return pd.DataFrame(dict(taxon_concentration=concentration, effective_genes=effective_genes,
        gene_diverse_size=size_penalty+np.clip(1-effective_genes/sizes, 0, 1),
        homogeneous_size=size_penalty+np.clip(1-concentration, 0, 1)))


def composition_statistics(dataset, name, table):
    if not len(table):
        return composition_penalties([], sparse.csr_matrix((0,0)), sparse.csr_matrix((0,0)))
    root = cached_root(dataset)
    graph = (PROJECT/"outputs/method_comparison/real"/dataset/"bernoulli_join_graph"
             if name == "outer" else root/name/"join")
    with np.load(graph/"assignments.npz") as data:
        ids, labels = data["feature_ids"], data["clusters"]
    with np.load(root/name/"split.npz") as data:
        if not np.array_equal(ids, data["feature_ids"]):
            raise ValueError("Composition graph differs from cached prevalence IDs")
    with np.load(PROJECT/"outputs/preprocessing"/dataset/"edges.npz") as data:
        taxa = source_weights_from_assignments(data["pair_bacteria"], ids, labels, table)
        genes = source_weights_from_assignments(data["pair_gene"], ids, labels, table)
    return composition_penalties(table.pair_count.to_numpy(), taxa, genes)


def select_penalized(train, target, table, statistics, family, strength, maximum=100):
    if family in COVERAGE_FAMILIES and (not np.isfinite(strength) or strength < 0):
        raise ValueError("Coverage discount must be nonnegative and finite")
    z, available = standardized(train)
    effect = np.abs(table.effect.to_numpy(dtype=float))
    if family == "baseline":
        chosen = list(range(min(maximum, train.shape[1])))
        return chosen, np.zeros(len(chosen))
    if family == "coverage_partition_power":
        score, penalty = coverage_partition_scores(table.effect, statistics.partition_instability,
                                                   statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "bh_coverage_power":
        score, penalty = bh_coverage_power_scores(table.p, statistics.source_coverage, strength)
        order = np.arange(len(score)) if strength == 0 else np.argsort(-score, kind="stable")
        chosen = order[available[order]][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "coverage_snr_power":
        score, penalty = coverage_snr_scores(table.effect, statistics.effect_lcb,
                                             statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "coverage_alignment_power":
        score, penalty = coverage_alignment_scores(table.effect, statistics.linear_alignment,
                                                   statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "coverage_uncertainty_power":
        score, penalty = coverage_uncertainty_scores(table.effect, statistics.effect_lcb,
                                                     statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "coverage_logistic_gain":
        coverage = statistics.source_coverage.to_numpy(float)
        _, _ = coverage_power_scores(table.effect, coverage, strength)
        association = ndtri(np.maximum(table.p.to_numpy(float), np.finfo(float).tiny)/2)**2
        chosen, losses = [], []
        for _ in range(min(maximum, int(available.sum()))):
            gains = logistic_conditional_gains(train, target, chosen)
            score = np.minimum(association, gains)*np.power(coverage, strength)
            i = int(np.argmax(np.where(available, score, -np.inf)))
            chosen.append(i);losses.append(float(association[i]-score[i]));available[i] = False
        return chosen, np.asarray(losses)
    if family in CATALOGUE_FAMILIES:
        weights, catalogue = table.attrs["catalogue_metadata"]
        mode = "completion" if family == "catalogue_completion_power" else "remaining"
        return select_remaining_catalogue(table.effect, weights, catalogue, strength, available, maximum, mode)
    if family == "coverage_power":
        score, penalty = coverage_power_scores(table.effect, statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "coverage_support_power":
        score, penalty = coverage_support_power_scores(table.effect, statistics.source_coverage,
                                                       1-statistics.concentration, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "bh_log_coverage":
        score, penalty = bh_log_coverage_scores(table.p, statistics.source_coverage, strength)
        score = np.where(available, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == 'size_taxon_coverage':
        penalty=statistics.taxon_coverage.to_numpy(float)
        association=effect-.25*statistics.cluster_size.to_numpy(float)
        score=np.where(available,association-strength*penalty,-np.inf)
        order=np.argsort(-score,kind='stable')
        chosen=order[np.isfinite(score[order])][:maximum].tolist()
        return chosen,penalty[chosen]
    if family in STABILITY_FAMILIES:
        frequency = statistics[f"l1_frequency_{strength:g}"].to_numpy(float)
        if frequency.shape != available.shape or not np.isfinite(frequency).all() or np.any((frequency < 0)|(frequency > 1)):
            raise ValueError("Invalid resampled selection frequencies")
        penalty = 1-frequency
        association = effect-(.25*statistics.cluster_size.to_numpy(float) if family == "size_l1_stability" else 0)
        score = np.where(available, association-.25*penalty, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "l1_importance":
        if strength <= 0:
            raise ValueError("Sparse scorer C must be positive")
        if not available.any():
            return [], np.array([], dtype=float)
        scorer = new_model(SEED)
        scorer[-1].set_params(l1_ratio=1.0, C=strength)
        scorer.fit(train[:,available], target)
        importance = np.zeros(train.shape[1])
        importance[available] = np.abs(scorer[-1].coef_[0])
        importance /= max(float(importance.max()), 1e-14)
        penalty = 1-importance
        score = np.where(available, effect-.25*penalty, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family == "size_plus_corr":
        if strength < 0:
            raise ValueError("Correlation discount must be nonnegative")
        association = effect-.25*statistics.cluster_size.to_numpy(float)
        redundancy = np.zeros(train.shape[1])
        chosen, penalties = [], []
        for _ in range(min(maximum, int(available.sum()))):
            score = np.where(available, association-strength*redundancy, -np.inf)
            i = int(np.argmax(score))
            chosen.append(i); penalties.append(float(redundancy[i])); available[i] = False
            if strength:
                redundancy = np.maximum(redundancy, np.abs(z.T @ z[:,i]))
        return chosen, np.asarray(penalties)
    if family in RANK_FAMILIES:
        if not 0 <= strength <= 1:
            raise ValueError("Rank-disagreement strength must be in [0,1]")
        scale = max(1, train.shape[1]-1)
        association = -np.arange(train.shape[1], dtype=float)/scale
        if family == "bh_coverage_rank_blend":
            adjusted_score, _ = coverage_power_scores(table.effect, statistics.source_coverage, 1)
        else:
            adjusted_score = effect-.25*statistics.cluster_size.to_numpy(float)
        adjusted_ordinal = rankdata(-adjusted_score, method="ordinal")-1
        adjusted_rank = -adjusted_ordinal/scale
        penalty = association-adjusted_rank
        if family == "bh_size_rank_cap":
            penalty = np.maximum(0, penalty)
        raw_score = (-( (1-strength)*np.arange(train.shape[1])+strength*adjusted_ordinal)/scale
                     if family == "bh_coverage_rank_blend" else association-strength*penalty)
        score = np.where(available, raw_score, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family in SOURCE_FAMILIES:
        if strength < 0:raise ValueError("Source-overlap discount must be nonnegative")
        weights=table.attrs["source_weights"]
        if weights.shape[0]!=train.shape[1]:raise ValueError("Unaligned source metadata")
        association=(-np.log10(np.maximum(table.p.to_numpy(float),np.finfo(float).tiny))
                     if family=="bh_source_overlap" else effect-.25*statistics.cluster_size.to_numpy(float))
        coverage=np.zeros(weights.shape[1]);chosen,penalties=[],[]
        for _ in range(min(maximum,int(available.sum()))):
            overlap=np.asarray(weights@coverage).ravel()
            i=int(np.argmax(np.where(available,association-strength*overlap,-np.inf)))
            chosen.append(i);penalties.append(float(overlap[i]));available[i]=False
            coverage=np.maximum(coverage,weights[i].toarray().ravel())
        return chosen,np.asarray(penalties)
    if family == "logistic_gain":
        association=ndtri(np.maximum(table.p.to_numpy(float),np.finfo(float).tiny)/2)**2
        chosen,penalties=[],[]
        for _ in range(min(maximum,int(available.sum()))):
            penalty=(np.maximum(0,association-logistic_conditional_gains(train,target,chosen))
                     if chosen and strength else np.zeros(train.shape[1]))
            i=int(np.argmax(np.where(available,association-strength*penalty,-np.inf)))
            chosen.append(i);penalties.append(float(penalty[i]));available[i]=False
        return chosen,np.asarray(penalties)
    if family in ("bh_size_shrinkage", "bh_support_shrinkage"):
        association = -np.log10(np.maximum(table.p.to_numpy(float), np.finfo(float).tiny))
        support = (table.pair_count.to_numpy(float) if family == "bh_size_shrinkage" else
                   train.shape[0]*(1-statistics.concentration.to_numpy(float)))
        pseudocount = strength if family == "bh_size_shrinkage" else strength*train.shape[0]
        # Heuristic reliability discount, not a recalibrated p-value.
        fraction = np.divide(pseudocount, support+pseudocount,
                             out=np.zeros_like(support), where=support+pseudocount > 0)
        penalty = association*fraction/strength if strength else np.zeros_like(association)
        score = np.where(available, association-strength*penalty, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    if family != "within_class_corr":
        if family.startswith("bh_"):
            association = -np.log10(np.maximum(table.p.to_numpy(float), np.finfo(float).tiny))
            if family=="bh_linear_alignment":
                penalty=statistics.bh_linear_alignment.to_numpy()
            else:
                penalty = (statistics.cluster_size.to_numpy() if family == "bh_cluster_size" else
                           statistics.partition_instability.to_numpy()/np.maximum(effect,1e-12))
        else:
            association = effect
            penalty = np.zeros(train.shape[1]) if family == "effect_only" else statistics[family].to_numpy()
        score = np.where(available, association-strength*penalty, -np.inf)
        order = np.argsort(-score, kind="stable")
        chosen = order[np.isfinite(score[order])][:maximum].tolist()
        return chosen, penalty[chosen]
    by_class = [standardized(train[target == label])[0] for label in (0,1)]
    penalty, chosen, penalties = np.zeros(train.shape[1]), [], []
    for _ in range(min(maximum, int(available.sum()))):
        i = int(np.argmax(np.where(available, effect-strength*penalty, -np.inf)))
        chosen.append(i); penalties.append(float(penalty[i])); available[i] = False
        correlations = sum(np.abs(a.T @ a[:,i]) for a in by_class)/2
        penalty = np.maximum(penalty, correlations)
    return chosen, np.asarray(penalties)


def cached_root(dataset):
    return PROJECT / "outputs/qinj_redundancy_analysis" if dataset == "QinJ_2012" else PROJECT / "outputs/real_redundancy_analysis" / dataset


def load_space(dataset, name):
    root = cached_root(dataset)
    marker, protocol = json.loads((root/"complete.json").read_text()), json.loads((root/"protocol.json").read_text())
    if marker.get("status") != "complete" or marker["signature"] != protocol["signature"]:
        raise ValueError(f"Incomplete source cache: {dataset}")
    directory = root / name
    if not (directory/"representations_complete.json").exists():
        raise ValueError(f"Missing cached representation: {directory}")
    y = pd.read_csv(PROJECT / "outputs/preprocessing" / dataset / "samples.csv").y.to_numpy(int)
    with np.load(directory/"split.npz") as saved:
        train_idx, test_idx = saved["train_index"], saved["test_index"]
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Training/validation sample overlap")
    train, test = sparse.load_npz(directory/"join_train.npz"), sparse.load_npz(directory/"join_test.npz")
    table = pd.read_csv(directory/"join_features.csv.gz")
    if train.shape != (len(train_idx), len(table)) or test.shape != (len(test_idx), len(table)):
        raise ValueError("Unaligned cached feature space")
    return train, test, table, y[train_idx], y[test_idx], train_idx, test_idx


def run_training(dataset, grid):
    directory = OUTPUT / dataset
    directory.mkdir(parents=True, exist_ok=True)
    _, _, _, _, _, outer_train, outer_test = load_space(dataset, "outer")
    rows, scores = [], []
    for fold in range(3):
        train, test, table, ytrain, ytest, a, b = load_space(dataset, f"fold_{fold}")
        if SOURCE_FAMILIES.intersection(grid):table.attrs["source_weights"]=source_weights(dataset,f"fold_{fold}",table)
        if CATALOGUE_FAMILIES.intersection(grid):table.attrs["catalogue_metadata"]=catalogue_metadata(dataset,f"fold_{fold}",table)
        if np.intersect1d(np.r_[a,b], outer_test).size or set(np.r_[a,b]) != set(outer_train):
            raise ValueError("Fold isolation failed")
        stats = penalty_statistics(train, ytrain, table)
        if COVERAGE_FAMILIES.intersection(grid):stats=pd.concat([stats,coverage_statistics(dataset,f'fold_{fold}',table)],axis=1)
        for c in {c for family,cs in grid.items() if family in STABILITY_FAMILIES for c in cs}:
            stats[f"l1_frequency_{c:g}"] = resampled_l1_frequency(train,ytrain,c)
        if COMPOSITION_FAMILIES.intersection(grid):
            stats = pd.concat([stats, composition_statistics(dataset, f"fold_{fold}", table)], axis=1)
        stats.to_csv(directory/f"fold_{fold}_statistics.csv.gz", index=False)
        print(f"{dataset} fold={fold}: samples={len(a)}/{len(b)}, classes={np.bincount(ytrain).tolist()}, "
              f"screened_join={len(table)}, original_pairs={int(table.pair_count.sum())}", flush=True)
        for family, strengths in {"baseline": [0], **grid}.items():
            for position, strength in enumerate(strengths):
                chosen, _ = select_penalized(train, ytrain, table, stats, family, strength)
                metric, _ = evaluate(train[:,chosen], test[:,chosen], ytrain, ytest, range(1,11))
                metric = metric.assign(fold=fold, family=family, strength=strength)
                scores.append(metric)
                rows.append(dict(fold=fold, family=family, strength=strength, position=position,
                                 primary_auc=float(metric[metric.requested_k.isin(PRIMARY_K)].auroc.mean()),
                                 fine_auc=float(metric.auroc.mean()), candidates=len(table)))
    pd.concat(scores).to_csv(directory/"inner_metrics.csv", index=False)
    cv = pd.DataFrame(rows)
    cv.to_csv(directory/"inner_cv.csv", index=False)
    summary = cv.groupby(["family","strength","position"], as_index=False).agg(
        primary_auc=("primary_auc","mean"), fine_auc=("fine_auc","mean"))
    locked = summary.sort_values(["family","primary_auc","strength"], ascending=[True,False,True]).groupby("family").head(1)
    locked.to_csv(directory/"locked_parameters.csv", index=False)
    order = {family:i for i,family in enumerate(grid)}
    candidate = locked[locked.family.ne("baseline")].assign(family_order=lambda t:t.family.map(order)).sort_values(
        ["primary_auc","strength","family_order"], ascending=[False,True,True]).iloc[0]
    choice = dict(family=candidate.family, strength=float(candidate.strength), primary_auc=float(candidate.primary_auc),
                  baseline_cv_auc=float(locked[locked.family.eq("baseline")].primary_auc.iloc[0]))
    (directory/"training_choice.json").write_text(json.dumps(choice, indent=2))
    return locked, choice


def run_holdout(dataset, locked, choice):
    directory = OUTPUT / dataset
    train, test, table, ytrain, ytest, _, test_idx = load_space(dataset, "outer")
    if SOURCE_FAMILIES.intersection(locked.family):table.attrs["source_weights"]=source_weights(dataset,"outer",table)
    if CATALOGUE_FAMILIES.intersection(locked.family):table.attrs["catalogue_metadata"]=catalogue_metadata(dataset,"outer",table)
    stats = penalty_statistics(train, ytrain, table)
    if COVERAGE_FAMILIES.intersection(locked.family):stats=pd.concat([stats,coverage_statistics(dataset,'outer',table)],axis=1)
    for c in locked.loc[locked.family.isin(STABILITY_FAMILIES),'strength'].unique():
        stats[f"l1_frequency_{c:g}"] = resampled_l1_frequency(train,ytrain,c)
    if COMPOSITION_FAMILIES.intersection(locked.family):
        stats = pd.concat([stats, composition_statistics(dataset, "outer", table)], axis=1)
    stats.to_csv(directory/"outer_statistics.csv.gz", index=False)
    frames, predictions = [], []
    for row in locked.itertuples():
        chosen, penalties = select_penalized(train, ytrain, table, stats, row.family, row.strength)
        metric, prediction = evaluate(train[:,chosen], test[:,chosen], ytrain, ytest, K_VALUES)
        frames.append(metric.assign(family=row.family, strength=row.strength))
        predictions.append(prediction.assign(family=row.family, sample_index=test_idx[prediction.test_position]))
        if row.family in RANK_FAMILIES:
            association = -np.asarray(chosen, dtype=float)/max(1, train.shape[1]-1)
        elif row.family in ("size_plus_corr", "size_plus_source", "size_l1_stability", "size_taxon_coverage"):
            association = np.abs(table.effect.to_numpy(float)[chosen])-.25*stats.cluster_size.to_numpy(float)[chosen]
        elif row.family in ("logistic_gain", "coverage_logistic_gain"):
            association=ndtri(np.maximum(table.p.to_numpy(float)[chosen],np.finfo(float).tiny)/2)**2
        else:
            association = (-np.log10(np.maximum(table.p.to_numpy(float)[chosen],np.finfo(float).tiny))
                           if row.family.startswith("bh_") or row.family == "baseline" else np.abs(table.effect.to_numpy()[chosen]))
        multiplier = (1 if row.family in ("coverage_power", "coverage_support_power", "coverage_logistic_gain", "coverage_uncertainty_power", "coverage_alignment_power", "coverage_snr_power", "bh_coverage_power", "coverage_partition_power") or row.family in CATALOGUE_FAMILIES else
                      .25 if row.family == "l1_importance" or row.family in STABILITY_FAMILIES else row.strength)
        table.iloc[chosen].assign(selection_rank=np.arange(1,len(chosen)+1), penalty=penalties,
            selection_score=association-multiplier*penalties).to_csv(
                directory/f"{row.family}_selected.csv", index=False)
    metrics = pd.concat(frames, ignore_index=True)
    reference = pd.read_csv(cached_root(dataset)/"metrics.csv")
    for old, family in (("baseline","baseline"),("bacteria","bacteria")):
        r = reference[reference.method.eq(old)].copy().rename(columns={"method":"family","parameter":"strength"})
        if old == "baseline":
            baseline = metrics[metrics.family.eq("baseline")].sort_values("requested_k")
            np.testing.assert_allclose(baseline.auroc, r.sort_values("requested_k").auroc, atol=1e-10)
        else:
            metrics = pd.concat([metrics,r], ignore_index=True)
    metrics.to_csv(directory/"holdout_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory/"holdout_predictions.csv.gz", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, ks in (("fine",range(1,11)),("coarse",[5,10,20,35,50,60,75,85,100])):
        fig, ax = plt.subplots(figsize=(11,6))
        for family, curve in metrics[metrics.requested_k.isin(ks)].groupby("family"):
            ax.plot(curve.requested_k, curve.auroc, marker="o", label=family,
                    linewidth=3 if family in (choice["family"],"bacteria","baseline") else 1)
        ax.set(title=f"{dataset}: locked post-hoc penalties (exploratory holdout)", xlabel="Requested K", ylabel="AUROC")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(directory/f"{name}_auc.png", dpi=200); plt.close(fig)
    print(metrics[metrics.requested_k.isin(PRIMARY_K)].groupby("family").auroc.mean().to_string(float_format=lambda x:f"{x:.3f}"), flush=True)
    return metrics


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [PROJECT/"outputs/preprocessing/QinJ_2012/samples.csv"]
    inputs.extend([PROJECT/"scripts/run_qinj_redundancy_analysis.py",
                   PROJECT/"feature_selection/run_feature_selection.py"])
    root = cached_root("QinJ_2012")
    metadata = bool(SOURCE_FAMILIES.intersection(GRID) or COMPOSITION_FAMILIES.intersection(GRID) or COVERAGE_FAMILIES.intersection(GRID))
    if metadata:
        inputs.append(PROJECT/"outputs/preprocessing/QinJ_2012/edges.npz")
    for name in ("fold_0","fold_1","fold_2","outer"):
        inputs.extend(root/name/file for file in ("join_train.npz","join_test.npz","join_features.csv.gz","split.npz"))
        if metadata:
            graph=PROJECT/"outputs/method_comparison/real/QinJ_2012/bernoulli_join_graph" if name=="outer" else root/name/"join"
            inputs.append(graph/"assignments.npz")
    signature = hashlib.sha256(Path(__file__).read_bytes()+repr((GRID,PRIMARY_K,
        [(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in inputs])).encode()).hexdigest()
    protocol = dict(signature=signature, grid=GRID, primary_k=PRIMARY_K, seed=SEED,
                    graph_recomputation=False, candidate_pool="unchanged BH q<=.10",
                    selection="training CV only; mean AUROC over K5..10; weaker penalty then listed family for exact ties",
                    holdout_status="already explored; not independent confirmation")
    definitions = {
        "bh_size_shrinkage": "(-log10 cached p) * S/(S+lambda); S=original cluster pair count",
        "bh_support_shrinkage": "(-log10 cached p) * Neff/(Neff+lambda*n); Neff=(sum x)^2/sum(x^2), training only",
        "gene_diverse_size": "abs(MW effect) - lambda * (1/sqrt(S) + 1 - effective_genes/S); effective_genes=1/sum gene_fraction^2",
        "homogeneous_size": "abs(MW effect) - lambda * (1/sqrt(S) + 1 - sum taxon_fraction^2)",
        "bh_size_rank_blend": "BH rank score minus w*(BH rank score - size rank score); size rank uses abs(effect)-0.25/sqrt(S)",
        "bh_size_rank_cap": "BH rank score minus w*max(0, BH rank score - size rank score); size rank uses abs(effect)-0.25/sqrt(S)",
    }
    protocol["penalty_definitions"] = {family: definitions[family] for family in GRID if family in definitions}
    file = OUTPUT/"protocol.json"
    if file.exists() and json.loads(file.read_text())["signature"] != signature and (OUTPUT/"QinJ_2012/training_choice.json").exists():
        raise ValueError("Protocol changed; preserve this batch and use a new output directory")
    file.write_text(json.dumps(protocol, indent=2))
    locked, choice = run_training("QinJ_2012", GRID)
    print("TRAINING CHOICE", json.dumps(choice), flush=True)
    metrics = run_holdout("QinJ_2012", locked, choice)
    candidate = metrics[metrics.family.eq(choice["family"])].set_index("requested_k")
    bacteria = metrics[metrics.family.eq("bacteria")].set_index("requested_k")
    choice["matches_bacteria_at_k10"] = bool(candidate.loc[10,"auroc"] >= bacteria.loc[10,"auroc"])
    choice["k10_auc"] = float(candidate.loc[10,"auroc"])
    choice["bacteria_k10_auc"] = float(bacteria.loc[10,"auroc"])
    choice["matches_bacteria_fine_k"] = [k for k in range(1,11)
        if candidate.loc[k,"auroc"] >= bacteria.loc[k,"auroc"]]
    (OUTPUT/"batch_result.json").write_text(json.dumps(dict(status="complete", qinj_choice=choice,
        goal_achieved=False, other_datasets_tested=False), indent=2))


if __name__ == "__main__":
    main()
