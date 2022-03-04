"""
Given a tabular data, this script trains T-CorEx on the data
and saves important statistics in an output pickle file.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import pickle
import sys
import logging
from logging import debug, info, warning, error, critical

# import T-CorEx and needed tools
from tcorex import TCorex
from tcorex.experiments.data import make_buckets


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Input Data")

    group.add_argument("--already-wide", action="store_true",
                        help="The data has already been pivoted")
    group.add_argument('-v', '--value-column', type=str, default="count",
                        help='name of the value column')
    group.add_argument('--time-column', type=str, default='timestamp',
                        help='name of the time column')
    group.add_argument('-k', '--key', type=str, default='key',
                        help='name of the key column')

    group = parser.add_argument_group("T-CorEx parameters")
    group.add_argument('-n', '-z', '--n_hidden', type=int, required=True,
                        help='Number of latent factors')
    group.add_argument('-w', '--window-size', type=int, default=20,
                        help='help=window size used in T-CorEx.')
    group.add_argument('-l', '--l1', type=float, default=0.01,
                        help='L1 regularization strength')
    group.add_argument('-g', '--gamma', type=float, default=0.5,
                        help='T-CorEx gamma parameter')
    group.add_argument('-i', '--max_iter', type=int, default=500,
                        help='Max number of iterations')
    group.add_argument('-D', '--device', type=str, default='cpu')


    group = parser.add_argument_group("Output")
    group.add_argument("--log-level", "--ll", default="info",
                        help="Define the logging verbosity level (debug, info, warning, error, fotal, critical).")


    parser.add_argument('data_path', type=str,
                        help='Path to csv table to analyze')

    group.add_argument('output_path', type=argparse.FileType('wb'),
                       default=sys.stdout, nargs="?",
                       help='Path to saved results to in pkl format')

    args = parser.parse_args()

    log_level = args.log_level.upper()
    logging.basicConfig(level=log_level,
                        format="%(levelname)-10s:\t%(message)s")

    # load the data
    info("Reading from {}".format(args.data_path))
    if args.already_wide:
        df = pd.read_csv(args.data_path, index_col=0)
    else:
        df = load_and_pivot_table(args.data_path, args.time_column,
                                  args.key, args.value_column)
        # read in the data and pivot it to a wide format

    results = run_tcorex(df,
                         window_size=args.window_size,
                         n_hidden=args.n_hidden,
                         gamma=args.gamma,
                         l1=args.l1,
                         max_iterations=args.max_iter,
                         device=args.device)

    info("Saving to {}".format(args.output_path))
    pickle.dump(results, args.output_path)


def run_tcorex(df, window_size, n_hidden, gamma, l1, max_iterations, device):

    data = np.array(df).astype(np.float)

    # standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # cut the tail for the data
    reminder = data.shape[0] % window_size
    if reminder > 0:
        data = data[:-reminder]

    # add small Gaussian noise for avoiding NaNs
    data = data + 1e-4 * np.random.normal(size=data.shape)

    # break it into non-overlapping periods
    data, index_to_bucket = make_buckets(data, window=window_size, stride='full')

    # train T-CorEx
    nv = data[0].shape[1]
    tc = TCorex(nt=len(data), nv=nv, n_hidden=n_hidden, l1=l1,
                gamma=gamma, max_iter=max_iterations, tol=1e-3,
                optimizer_params={'lr': 0.01}, init=False, verbose=2,
                device=device)
    tc.fit(data)

    # save important things
    debug("Calculating needed statistics")
    mis = tc.mis()
    clusters = [mi.argmax(axis=0) for mi in mis]
    results = {
        'clusters': clusters,
        'mutual_informations': mis,
        'tcorex_weights': tc.get_weights(),
        'factorizations': tc.get_factorization(),
        'covariance_matrices': (None if nv > 1000 else tc.get_covariance()),
        'window_size': window_size,
        'df.index': df.index,
        'df.columns': df.columns,
        'thetas': tc.theta,
        'method': 'T-CorEx'
    }

    return results;


def pivot_table(df, time_column, key, value_column):
    df = df.pivot_table(values=value_column,
                        columns=key, index=time_column)

    # index
    df.index = pd.to_datetime(df.index, unit='s')
    min_gap = df.index[1] - df.index[0]
    for i in range(len(df.index) - 1):
        assert df.index[i] < df.index[i + 1]
        min_gap = min(min_gap, df.index[i + 1] - df.index[i])

    df = df.reindex(index=pd.date_range(df.index[0], df.index[-1],
                                        freq=df.index[1] - df.index[0]),
                    labels=time_column)
    df = df.fillna(0)

    return df


def load_and_pivot_table(filename, time_column, key, value_column): 
    # load the data
    debug("Reading from {}".format(filename))
    df = pd.read_csv(filename,
                     dtype={'count': np.int32, 'timestamp': np.int32,
                            'key': str})
    import pdb ; pdb.set_trace()
    return pivot_table(df, key, value_column, time_column)


if __name__ == '__main__':
    main()
