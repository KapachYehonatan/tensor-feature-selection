"""Explicit Pearson cutoffs on unchanged Bernoulli-join features."""
import argparse
import math
from execution.spec import Experiment
from methods.pipelines import BernoulliJoin
from methods.selectors.mann_whitney import MannWhitney
from datasets.catalog import NEW_DATASETS


def experiment(thresholds):
    return Experiment(
        name="join_correlation_filter", datasets=("QinJ_2012", "NielsenHB_2014_unique") + NEW_DATASETS,
        methods=(BernoulliJoin(),) + tuple(BernoulliJoin(
            selector=MannWhitney(correlation_threshold=t), name=f"Join + filter {t!r}") for t in thresholds),
        metrics=("prediction", "discoveries", "correlation", "shuffled_fdp"),
        outputs=("join_correlation_filter_pilot.png",),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, nargs="+", required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(t) or not 0 < t < 1 for t in args.threshold) or len(set(args.threshold)) != len(args.threshold):
        parser.error("Thresholds must be finite, unique, and strictly between zero and one")
    definition = experiment(args.threshold)
    if args.plot_only:
        from reporting.comparison import render
        render(definition)
    else:
        from execution.pipeline import run
        if run(definition):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
