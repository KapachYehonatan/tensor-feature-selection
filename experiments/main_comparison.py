"""Four-method comparison across the real datasets."""
import argparse
from datasets.catalog import REAL_DATASETS
from execution.spec import Experiment, Evaluation
from methods.pipelines import Bacteria, Gene, BernoulliJoin, BinsEValue

EVALUATION = Evaluation()
FINE, COARSE = EVALUATION.tag(EVALUATION.fine_k), EVALUATION.tag(EVALUATION.coarse_k)
EXPERIMENT = Experiment(
    name="main_comparison", datasets=REAL_DATASETS,
    methods=(Bacteria(), Gene(), BernoulliJoin(), BinsEValue()),
    metrics=("prediction", "discoveries", "correlation", "shuffled_fdp"),
    outputs=(f"AUC_average_k_{FINE}.png", f"AUC_overview_k_{FINE}.png", f"AUC_average_k_{COARSE}.png",
             f"AUC_overview_k_{COARSE}.png", "average_pairwise_absolute_correlation_overview.png",
             f"average_pairwise_absolute_correlation_k_{COARSE}.png", "negative_log10_p_overview.png",
             "real_discoveries_overview.png"),
    evaluation=EVALUATION,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        from reporting.comparison import render
        render(EXPERIMENT)
    else:
        from execution.pipeline import run
        if run(EXPERIMENT):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
