#!/usr/bin/env python3
"""Training-only fold-consistency guard for subset reliability × coverage."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_joint_k_penalty_selection as joint
from scripts import run_size_source_transfer as transfer
import numpy as np

OUTPUT = PROJECT / "outputs/posthoc_penalty_search/fold_consistent_partition_batch_37"
GRID = {"coverage_partition_power": [0, .05, .10, .25, .50, 1, 2]}
DESCRIPTION = ("Choose candidate parameter/K and BH/K by mean training CV with existing tie rules; "
    "use candidate only if its paired mean gain is positive and no validation fold loses AUROC "
    "relative to BH's own CV-selected K (numerical tolerance 1e-12). Otherwise retain BH. "
    "One-SE pass is diagnostic only, not the guard; no known-holdout input or dataset-name branch.")


def choose_fold_consistent_settings(cv, grid, reference_families=("bacteria",)):
    """Pure rule for a fresh dataset; references never affect the decision."""
    mean, locked, choice = joint.choose_joint_settings(cv, grid, reference_families)
    baseline = locked[locked.family.eq("baseline")].iloc[0]
    candidate = locked[locked.family.isin(grid)].assign(
        family_order=lambda x:x.family.map({name:i for i,name in enumerate(grid)})).sort_values(
            ["mean_cv_auc", "strength", "requested_k", "family_order"],
            ascending=[False, True, True, True]).iloc[0]
    def fold_scores(setting):
        return cv[cv.family.eq(setting.family)&cv.strength.eq(setting.strength)&
                  cv.requested_k.eq(setting.requested_k)].sort_values("fold").auroc.to_numpy()
    deltas = fold_scores(candidate)-fold_scores(baseline)
    gain, se = float(deltas.mean()), float(deltas.std(ddof=1)/np.sqrt(3))
    accepted = bool(choice["family"] != "baseline" and gain > 1e-12 and np.min(deltas) >= -1e-12)
    if not accepted:
        choice = dict(family="baseline", strength=0., requested_k=int(baseline.requested_k),
                      mean_cv_auc=float(baseline.mean_cv_auc))
    choice.update(guard="positive mean gain and no fold loss versus BH/own-K", guard_accepted=accepted,
        nominated_family=candidate.family, nominated_strength=float(candidate.strength),
        nominated_k=int(candidate.requested_k), paired_cv_deltas=deltas.tolist(), paired_cv_gain=gain,
        paired_cv_se=se, would_pass_one_se=bool(gain > se+1e-12))
    return mean, locked, choice


if __name__ == "__main__":
    joint.SELECTION_RULE = choose_fold_consistent_settings
    joint.SELECTION_DESCRIPTION = DESCRIPTION
    transfer.OUTPUT, transfer.GRID = OUTPUT, GRID
    transfer.main()
