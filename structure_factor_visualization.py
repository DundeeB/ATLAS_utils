#!/Local/cmp/anaconda3/bin/python -u
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
import os
import re
from correlation_plot import get_corr_files
from mpl_toolkits import mplot3d


def main():
    # parse
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folder', type=str, nargs='?', help='folders to plot simulation result from')
    parser.add_argument('-r', '--real', type=str, nargs='?', default='',
                        help='Realization to plot. Default is last realization')

    # handle defaults and load data
    args = parser.parse_args()
    op_fold = join(args.folder, 'OP/Bragg_S')
    fig = plt.figure()
    for op_fold, lbl, sub in zip([op_fold, op_fold + 'm'], ['$S_m$', '$S$'], [1, 2]):
        if args.real == '':
            args.real, _, _, _ = get_corr_files(op_fold)
        data_path = join(op_fold, args.real)
        data = np.loadtxt(data_path)
        S_values = [d[2] for d in data]
        kx = [d[0] for d in data]
        ky = [d[1] for d in data]

        # font
        size = 15
        params = {'legend.fontsize': 'large',
                  'figure.figsize': (20, 8),
                  'axes.labelsize': size,
                  'axes.titlesize': size,
                  'xtick.labelsize': size * 0.75,
                  'ytick.labelsize': size * 0.75,
                  'axes.titlepad': 25}
        plt.rcParams.update(params)

        # graphs
        ax = fig.add_subplot(2, 1, sub, projection='3d')
        ax.scatter(kx, ky, S_values, '.')
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel(lbl)
    plt.title(args.folder)

    # show
    plt.show()


if __name__ == "__main__":
    main()
