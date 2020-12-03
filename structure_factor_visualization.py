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
    fig = plt.figure()

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
    axs = [fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')]
    axs[0].title(args.folder)
    op_fold = join(args.folder, 'OP/Bragg_S')
    for op_fold, lbl, sub in zip([op_fold, op_fold + 'm'], ['$S_m$', '$S$'], [0, 1]):
        if args.real == '':
            args.real, _, _, _ = get_corr_files(op_fold)
        kx, ky, S_values = np.loadtxt(join(op_fold, args.real), unpack=True, usecols=(0, 1, 2))
        # graphs
        ax = axs[sub]
        ax.scatter(kx, ky, S_values, '.')
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel(lbl)
    plt.show()


if __name__ == "__main__":
    main()
