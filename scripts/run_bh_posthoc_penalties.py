#!/usr/bin/env python3
"""Score penalties whose zero-strength case reproduces original BH ranking."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_posthoc_penalty_search as search
import pandas as pd

OUTPUT=PROJECT/"outputs/posthoc_penalty_search/bh_score_batch_2"
GRID={"bh_cluster_size": [0,.5,1,2,4], "bh_instability": [0,.5,1,2,4]}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(Path(__file__).read_bytes()+Path(search.__file__).read_bytes()).hexdigest()
    protocol=dict(signature=signature,grid=GRID,primary_k=search.PRIMARY_K,graph_recomputation=False,
        association="-log10(cached MW p), monotone with original BH ordering",
        penalties={"bh_cluster_size":"1/sqrt(original pair count)",
                   "bh_instability":"partition instability / abs(cached MW effect)"},
        adaptation="motivated by batch_1 transfer losses; known holdouts remain exploratory",
        transfer_gate="training-locked QinJ family must match bacteria at prespecified K10 before other-dataset tests")
    file=OUTPUT/"protocol.json"
    if file.exists() and json.loads(file.read_text())["signature"]!=signature:
        raise ValueError("Preserve this batch: changed protocol needs a new output directory")
    file.write_text(json.dumps(protocol,indent=2))
    search.OUTPUT=OUTPUT
    locked,choice=search.run_training("QinJ_2012",GRID)
    print("TRAINING CHOICE",json.dumps(choice),flush=True)
    metrics=search.run_holdout("QinJ_2012",locked,choice)
    candidate=metrics[metrics.family.eq(choice["family"])].set_index("requested_k")
    bacteria=metrics[metrics.family.eq("bacteria")].set_index("requested_k")
    gate=bool(candidate.loc[10,"auroc"]>=bacteria.loc[10,"auroc"])
    result=dict(status="complete",qinj_choice=choice,matches_bacteria_k10=gate,
        candidate_k10=float(candidate.loc[10,"auroc"]),bacteria_k10=float(bacteria.loc[10,"auroc"]),
        goal_achieved=False,other_datasets_tested=False)
    (OUTPUT/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
