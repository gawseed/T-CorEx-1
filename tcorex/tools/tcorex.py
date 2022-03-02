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
from logging import debug

# import T-CorEx and needed tools
from tcorex import TCorex
from tcorex.experiments.data import make_buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='path to preprocessed data csv table')
    parser.add_argument("--already-wide", action="store_true",
                        help="The data has already been pivoted")
    parser.add_argument('--value-column', '-v', type=str, default="count",
                        help='name of the value column')
    parser.add_argument('--time-column', type=str, default='timestamp',
                        help='name of the time column')
    parser.add_argument('--key', '-k', type=str, default='key',
                        help='name of the key column')
    parser.add_argument('--n_hidden', '-z', type=int, required=True,
                        help='Number of latent factors')
    parser.add_argument('--window-size', '-w', type=int, default=20,
                        help='help=window size used in T-CorEx.')
    parser.add_argument('--l1', '-l', type=float, default=0.01,
                        help='L1 regularization strength')
    parser.add_argument('--gamma', '-g', type=float, default=0.5,
                        help='T-CorEx gamma parameter')
    parser.add_argument('--max_iter', '-i', type=int, default=500,
                        help='Max number of iterations')
    parser.add_argument('--output-path', '-o', type=str, required=True,
                        help='path to saved file results')
    parser.add_argument('--device', '-D', type=str, default='cpu')
    args = parser.parse_args()
    print(args)

    # load the data
    if args.already_wide:
        print("Reading from {}".format(args.data_path))
        df = pd.read_csv(args.data_path, index_col=0)
    else:
        df = load_and_pivot_table(args.data_path, args.key,
                                  args.value_column, args.time_column)
        # read in the data and pivot it to a wide format

    data = np.array(df).astype(np.float)

    # standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # cut the tail for the data
    reminder = data.shape[0] % args.window_size
    if reminder > 0:
        data = data[:-reminder]

    # add small Gaussian noise for avoiding NaNs
    data = data + 1e-4 * np.random.normal(size=data.shape)

    # break it into non-overlapping periods
    data, index_to_bucket = make_buckets(data, window=args.window_size, stride='full')

    # train T-CorEx
    nv = data[0].shape[1]
    tc = TCorex(nt=len(data), nv=nv, n_hidden=args.n_hidden, l1=args.l1,
                gamma=args.gamma, max_iter=args.max_iter, tol=1e-3,
                optimizer_params={'lr': 0.01}, init=False, verbose=2,
                device=args.device)
    tc.fit(data)

    # save important things
    print("Calculating needed statistics")
    mis = tc.mis()
    clusters = [mi.argmax(axis=0) for mi in mis]
    save_dict = {
        'clusters': clusters,
        'mutual_informations': mis,
        'tcorex_weights': tc.get_weights(),
        'factorizations': tc.get_factorization(),
        'covariance_matrices': (None if nv > 1000 else tc.get_covariance()),
        'window_size': args.window_size,
        'df.index': df.index,
        'df.columns': df.columns,
        'thetas': tc.theta,
        'method': 'T-CorEx'
    }

    print("Saving to {}".format(args.output_path))
    with open(args.output_path, 'wb') as f:
        pickle.dump(save_dict, f)


def load_and_pivot_table(filename, key, value_column, time_column): 
    # load the data
    debug("Reading from {}".format(filename))
    df = pd.read_csv(filename,
                     dtype={'count': np.int32, 'timestamp': np.int32,
                            'key': str})
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


if __name__ == '__main__':
    main()
