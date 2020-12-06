#!/Local/cmp/anaconda3/bin/python -u
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
import os
import re
from mpl_toolkits import mplot3d


def main():
    # parse
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='folders to plot simulation result from')
    parser.add_argument('-r', '--reals', type=str, nargs='?', default=None,
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
    if args.reals is None:
        args.reals = [None for _ in args.folders]
    for fold, real in zip(args.folders, args.reals):
        op_fold = join(fold, 'OP/Bragg_S')
        for op_fold, lbl, sub in zip([op_fold, op_fold + 'm'], ['$S$', '$S_m$'], [0, 1]):
            if real is None:
                phi_files = [corr_file for corr_file in os.listdir(op_fold) if re.match('vec.*', corr_file)]
                phi_reals = [int(re.split('\.', re.split('_', corr_file)[-1])[0]) for corr_file in phi_files]
                args.real = str(np.max(phi_reals))
            kx, ky, S_values = np.loadtxt(join(op_fold, "vec_" + args.real + ".txt"), unpack=True, usecols=(0, 1, 2))
            # graphs
            ax = axs[sub]
            ax.scatter(kx, ky, S_values, '.', label=fold)
            if sub == 0:
                if len(args.folders) == 1:
                    ax.set_title(fold)
                else:
                    ax.legend()
            ax.set_xlabel('$k_x$')
            ax.set_ylabel('$k_y$')
            ax.set_zlabel(lbl)
    plt.show()


if __name__ == "__main__":
    main()
