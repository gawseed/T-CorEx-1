# Purpose

The T-Corex algorithm is designed to identify changes in correlation
in datasets of large signals.  Given windows of signals.

This package modifies the [original implementation] Hrayr Harutyunyan to
add documentation and CLI tools.

[original implementation]: https://github.com/hrayrhar/T-CorEx

# Usage

## Installation

Install with `pip` (or `pipx`):

```
pip3 install gawseed-tcorex
```

## Analysis

To analyze a temporal time-series of data in a csv file, run tcorex 
similar to:

```
tcorex -w 30 -n 100 -w 30 -g .1 -l .5 input_data.csv output.pkl
```

Discussion and help with selecting parameters is discussed below.

## Displaying results

To list the correlations within each window, use
`tcorex-changepoints` on the resulting pkl file:

```
tcorex-changepoints output.pkl
```

To show a graph of which time periods had the highest rate of
correlation changes, use `tcorex-plot`:

```
tcorex-plot -o changes output.pkl
```

## Picking algorithm parameters

### Number of latent factors (-n)

Look at the number of signals you have, and if you have a lot there
should be more latent factors.  If
you pick a value too low, then all the signals will appear to be
correlated even when they are not.  If you pick a value which is too
large, only an identity matrix will be produced and all the signals
will appear independent.  Typically start with a roughly small value
and raise it to improve results with respect to signals that are
actually related.

The number of latent factors should be roughly 2 order of magnitudes
lower (IE *number_of_signals/100*).

### Window size (-w)

The algorithm is designed to look for changes in correlations between
two windows.

The window size indicates how frequently changes will happen to
attempt to find changes between different window sizes.  If the window
is too small, then the accuracy will be reduced and more correlations
will be found as signals always appear similar at some point when
broken into small chunks.  If the window is too large, then no changes
will be easily detected because the difference between varying signals
will appear large.  When windows are large, you also lose track about
where correlation changes actually happen because the change point can
be anywhere in the window.

### Gamma

Range: 0.1-10

The Gamma value sets the decay rate for sample weights.  Bigger values
will result in more data being included, which is good for smaller
datasets.  If enough data is present, then low values should be used.

### L1 regularization strength

Higher L1 values are better better for signals with rapid changes

### L2

Better for smooth time series.  There is no current CLI option for this.

# Correlation Explanation Methods

Official implementation of linear correlation explanation (linear CorEx) and temporal correlation explanation (T-CorEx) methods.

#### Linear CorEx
Linear CorEx searches for independent latent factors that explain all correlations between observed variables, while also biasing
the model selection towards modular latent factor models – directed latent factor graphical models
where each observed variable has a single latent variable as its only parent.
This is useful for covariance estimation, clustering related variables, and dimensionality reduction, especially in the high-dimensional and under-sampled regime.
The complete description of the method is presented in NeurIPS 2019 [paper](https://papers.nips.cc/paper/9691-fast-structure-learning-with-modular-regularization) *"Fast structure learning with modular regularization"* by Greg Ver Steeg, Hrayr Harutyunyan, Daniel Moyer, and Aram Galstyan.
If you want to cite this paper, please use the following BibTex entry:
```text
@incollection{NIPS2019_9691,
title = {Fast structure learning with modular regularization},
author = {Ver Steeg, Greg and Harutyunyan, Hrayr and Moyer, Daniel and Galstyan, Aram},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {15567--15577},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/9691-fast-structure-learning-with-modular-regularization.pdf}
}
```

**Note:** Greg Ver Steeg has an alternative implementation of linear CorEx, which is available at [github.com/gregversteeg/LinearCorex](https://github.com/gregversteeg/LinearCorex).
That implementation uses a quazi-Newton optimization method for learning the model parameters. 
In contrast, the implementation provided in this repository uses ADAM optimizer.
This latter implementation utilizes GPUs better, and can converge to slightly better objective values if the input data is highly non-modular.
Nevertheless, we highly encourage to take a look at the alternative implementation.

#### T-CorEx
T-CorEx is a method for covariance estimation from temporal data.
It trains a linear CorEx for each time period,
while employing two regularization techniques to enforce temporal consistency of estimates.
The method is introduced in the [paper](https://arxiv.org/abs/1905.13276) *"Efficient Covariance Estimation from Temporal Data"* by Hrayr Harutunyan, Daniel Moyer, Hrant Khachatrian, Greg Ver Steeg, and Aram Galstyan.
If you want to cite this paper, please use the following BibTex entry:
```text
@article{tcorex,
  title={Efficient Covariance Estimation from Temporal Data},
  author={Harutyunyan, Hrayr and Moyer, Daniel and Khachatrian, Hrant and Steeg, Greg Ver and Galstyan, Aram},
  journal={arXiv preprint arXiv:1905.13276},
  year={2019}
}
```


Both linear CorEx and T-CorEx have linear time and memory complexity with respect to the number of observed variables and can be applied to high-dimensional datasets.
For example, it takes less than an hour on a moderate PC to estimate the covariance structure for time series with 100K variables using T-CorEx.
Both methods are implemented in PyTorch and can run on CPUs and GPUs.


## Requirements and Installation
The code is writen in Python 3, but should run on Python 2 as well.
The dependencies are the following: 
* numpy, scipy, tqdm, PyTorch
* [optional] nibabel (for fMRI experiments)
* [optional] nose (for tests)
* [optional] sklearn, regain, TVGL, linearcorex, pandas (for running comparisions)
* [optional] matplotlib and nilearn (for visualizations)

To install the code, run the following command:
```text
python setup.py install
```

## Description
The main method for linear CorEx is the class `tcorex.Corex`, and that of T-CorEx is 'tcorex.TCorex'.
The complete description of parameters of these classes can be found in the corresponding docstrings.
While there are many parameters (especially for T-CorEx), in general only a couple of them need to be tuned (others are set to their "best" values).
Those parameters are:


| Parameter | Linear CorEx | T-CorEx | Description |    
|:---------|---|---|:---|   
| `m` | + | + | The number of latent variables. Usually this is much smaller than the number of observed variables. |  
| `l1` | - | + | A non-negative real number specifying the coefficient of l1 temporal regularization.|  
| `gamma` | - | + | A real number in [0,1]. This argument controls the sample weights. The samples of time period t' will have weight w_t(t')=gamma^\|t' - t\| when estimating quantities for time period t. Smaller values are used for very dynamic time series.|  


## Usage

Run the following command for a sample run of TCorex. 
```bash
python -m examples.sample_run
```

The code is shown below:
``` python 
from __future__ import print_function
from __future__ import absolute_import

from tcorex.experiments.data import load_modular_sudden_change
from tcorex.experiments import baselines
from tcorex import base
from tcorex import TCorex
from tcorex import covariance as cov_utils

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def main():
    nv = 32         # number of observed variables
    m = 4           # number of hidden variables
    nt = 10         # number of time periods
    train_cnt = 16  # number of training samples for each time period
    val_cnt = 4     # number of validation samples for each time period

    # Generate some data with a sudden change in the middle.
    data, ground_truth_sigma = load_modular_sudden_change(nv=nv, m=m, nt=nt, ns=(train_cnt + val_cnt))

    # Split it into train and validation.
    train_data = [X[:train_cnt] for X in data]
    val_data = [X[train_cnt:] for X in data]

    # NOTE: the load_modular_sudden_change function above creates data where the time axis
    # is already divided into time periods. If your data is not divided into time periods
    # you can use the following procedure to do that:
    # bucketed_data, index_to_bucket = make_buckets(data, window=train_cnt + val_cnt, stride='full')
    # where the make_buckets function can be found at tcorex.experiments.data

    # The core method we have is the tcorex.TCorex class.
    tc = TCorex(nt=nt,
                nv=nv,
                n_hidden=m,
                max_iter=500,
                device='cpu',  # for GPU set 'cuda',
                l1=0.3,        # coefficient of temporal regularization term
                gamma=0.3,     # parameter that controls sample weights
                verbose=1,     # 0, 1, 2
                )

    # Fit the parameters of T-CorEx.
    tc.fit(train_data)

    # We can compute the clusters of observed variables for each time period.
    t = 8
    clusters = tc.clusters()
    print("Clusters at time period {}: {}".format(t, clusters[t]))

    # We can get an estimate of the covariance matrix for each time period.
    # When normed=True, estimates of the correlation matrices will be returned.
    covs = tc.get_covariance()

    # We can visualize the covariance matrices.
    fig, ax = plt.subplots(1, figsize=(5, 5))
    im = ax.imshow(covs[t])
    fig.colorbar(im)
    ax.set_title("Estimated covariance matrix\nat time period {}".format(t))
    fig.savefig('covariance-matrix.png')

    # It is usually useful to compute the inverse correlation matrices,
    # since this matrices can be interpreted as adjacency matrices of
    # Markov random fields.
    cors = tc.get_covariance(normed=True)
    inv_cors = [np.linalg.inv(x) for x in cors]

    # We can visualize the thresholded inverse correlation matrices.
    fig, ax = plt.subplots(1, figsize=(5, 5))
    thresholded_inv_cor = np.abs(inv_cors[t]) > 0.05
    ax.imshow(thresholded_inv_cor)
    ax.set_title("Thresholded inverse correlation\nmatrix at time period {}".format(t))
    fig.savefig('thresholded-inverse-correlation-matrix.png')

    # We can also plot the Frobenius norm of the differences of inverse
    # correlation matrices of  neighboring time periods. This is helpful
    # for detecting the sudden change points of the system.
    diffs = cov_utils.diffs(inv_cors)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.plot(diffs)
    ax.set_xlabel('t')
    ax.set_ylabel('$||\Sigma^{-1}_{t+1} - \Sigma^{-1}_{t}||_2$')
    ax.set_title("Frobenius norms of differences between\ninverse correlation matrices")
    fig.savefig('inv-correlation-difference-norms.png')

    # We can also do grid search on a hyperparameter grid the following way.
    # NOTE: this can take time!
    baseline, grid = (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
        'nv': nv,
        'n_hidden': m,
        'max_iter': 500,
        'device': 'cpu',
        'l1': [0.0, 0.03, 0.3, 3.0],
        'gamma': [1e-6, 0.3, 0.5, 0.8]
    })

    best_score, best_params, best_covs, best_method, all_results = baseline.select(train_data, val_data, grid)
    tc = best_method  # this is the model that performed the best on the validation data, you can use it as above
    base.save(tc, 'best_method.pkl')


if __name__ == '__main__':
    main()
```


