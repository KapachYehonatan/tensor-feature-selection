#!/usr/bin/env python3
"""Run the real-data exact Jaccard-join comparison method."""

import os

os.environ["JOIN_METHOD"] = "jaccard"

from run_bernoulli_correlation_join import main  # noqa: E402


if __name__ == "__main__":
    main()
