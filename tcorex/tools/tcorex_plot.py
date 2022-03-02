"""
Given a statistics file outputted by the `run_tcorex.py`, plots the
differences between inverse covariance matrices of adjacent time periods.
"""
import pandas as pd
import numpy as np
import argparse
import pickle
import sys
import os
import tcorex.plot_utils
from logging import info, debug, warn

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str,
                        default='change-point-detection.png',
                        help='path for saving the figure ({path}.png or {path}.pdf)')
    parser.add_argument('-i', '--dont-invert', action='store_true',
                        help="do not invert")

    parser.add_argument("--log-level", "--ll", default="info",
                        help="Define the logging verbosity level (debug, info, warning, error, fotal, critical).")

    parser.add_argument('statistics_path', type=str,
                        help='path to a T-CorEx statistics file (output from tcorex)')


    args = parser.parse_args()

    # import needed modules from ml-tools
    tcorex.plot_utils.set_style(plt)

    # import needed tools from T-CorEx code
    from tcorex.covariance import frob_diffs_given_factors

    # load the saved statistics
    with open(args.statistics_path, 'rb') as f:
        statistics = pickle.load(f)

    # compute the differences of neighboring precision matrices
    diffs = frob_diffs_given_factors(statistics['factorizations'],
                                     inverse=(not args.dont_invert))

    # plot the figure and save it
    plt.figure(figsize=(10, 5))
    plt.plot(diffs)
    plt.ylabel('$||\Theta_{t+1}-\Theta_{t}||_F$ (diff. of precision matrices)')

    df_index = pd.to_datetime(statistics['df.index'])
    time_delta = df_index[1] - df_index[0]
    window_size = statistics['window_size']

    xticks = -0.5 + np.arange(len(diffs) + 1)
    xlabels = [df_index[i * window_size] + window_size / 2.0 * time_delta for i in range(len(diffs) + 1)]
    n_ticks = min(20, len(xticks))
    step = np.round(1.0 * len(xticks) / n_ticks).astype(int)
    plt.xticks(xticks[::step], xlabels[::step], rotation=60, ha='right')

    info("Saving to {}".format(args.output_path))
    plt.savefig(args.output_path, bbox_inches="tight")


if __name__ == '__main__':
    main()
