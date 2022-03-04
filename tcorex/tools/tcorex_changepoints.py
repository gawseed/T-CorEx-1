#!/usr/bin/python3

"""
Given a statistics file outputted by the `run_tcorex.py` or by the `run_corex.py`,
the script outputs the most correlated variables in each cluster.
"""
import sys
import pandas as pd
import numpy as np
import argparse
import pickle
from argparse import FileType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--topk', type=int, default=20,
                        help='number of variables to consider from each cluster')
    parser.add_argument('-n', '--topn', type=int, default=10,
                        help='number of most correlated pairs to show')
    parser.add_argument("-m", "--min-correlation", default=1.0, type=float,
                        help="Minimum correlation value to print")


    parser.add_argument("pkl_file", help="Saved file to display changepoints from")
    args = parser.parse_args()

    # load the saved statistics
    statistics = pickle.load(open(args.pkl_file, "rb"))

    if statistics['method'] == 'T-CorEx':
        window_size = statistics['window_size']
        factorizations = statistics['factorizations']
        mis = statistics['mutual_informations']
    elif statistics['method'] == 'CorEx':
        window_size = len(statistics['df.index'])
        factorizations = [statistics['factorization']]
        mis = [statistics['mutual_informations']]
    else:
        raise ValueError("unknown value for 'method'")

    columns = statistics['df.columns']
    df_index = pd.to_datetime(statistics['df.index'])
    time_delta = df_index[1] - df_index[0]
    nt = len(mis)
    m, nv = mis[0].shape

    for t in range(nt):
        indices = np.zeros((nv,), dtype=np.bool)
        for j in range(m):
            cur_topk = np.argsort(-mis[t][j])[:args.topk]
            for idx in cur_topk:
                indices[idx] = True
        F = factorizations[t][:, indices]
        sigma = F.T.dot(F)
        cells = []
        for i in range(sigma.shape[0]):
            for j in range(i):
                cells.append((i, j, sigma[i, j]))
        cells = sorted(cells, key=lambda x: np.abs(x[2]), reverse=True)

        print("Top {} most correlated variables at time period {} - {}:".format(
            args.topn, df_index[window_size * t],
            df_index[window_size * t] + window_size * time_delta))
        keys=set()
        for i, j, c in cells[:args.topn]:
            if c < args.min_correlation:
                continue
            c1 = columns[i]
            c2 = columns[j]
            if 'keys' in statistics:
                keys.add(c1)
                keys.add(c2)
                c1 = statistics['keys'][c1]
                c2 = statistics['keys'][c2]
            print("    corr={:.2f}  {:<30}  {:<30} ".format(c, c1, c2))
        if len(keys) > 0:
            print(f'    keys: {",".join(map(str,keys))}')
        else:
            print("    No viable matches found")
        print("")


if __name__ == '__main__':
    main()
