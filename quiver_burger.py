#!/Local/cmp/anaconda3/bin/python -u
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
import os
import re
from correlation_plot import get_corr_files


def main():
    # parse
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folder', type=str, nargs='?', help='folders to plot simulation result from')
    parser.add_argument('-bf', '--burger_files', type=str, nargs='+', help='full path burger files to plot')
    parser.add_argument('-z', '--z_colour', type=bool, nargs='?', const=False, default=True,
                        help='colour upper and lower spheres with different colours')

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
    plt.xlabel('x')
    plt.ylabel('y')

    # handle defaults and load data
    args = parser.parse_args()
    if args.burger_files is None:
        burg_fold = join(args.folder, 'OP/burger_vectors')
        sorted_phi_files, sorted_phi_reals = get_corr_files(burg_fold, prefix='vec_')
        args.burger_files = [join(burg_fold, sorted_phi_files[0])]

    burgs = [np.loadtxt(burg_file) for burg_file in args.burger_files]
    reals_num = [int(re.split('(.*vec_|\.txt)', burg_file)[2]) for burg_file in args.burger_files]

    for real in np.unique(reals_num):
        sp = np.loadtxt(join(args.folder, str(real)))
        lbl = 'Center\'s xy for realization ' + str(real)
        if args.z_colour:
            up = np.where(sp[:, 2] > np.mean(sp[:, 2]))
            down = np.where(sp[:, 2] <= np.mean(sp[:, 2]))
            plt.plot(sp[:, 0][up], sp[:, 1][up], '.', markersize=6, label=lbl + ' up')
            plt.plot(sp[:, 0][down], sp[:, 1][down], '.', markersize=6, label=lbl + ' down')
        else:
            plt.plot(sp[:, 0], sp[:, 1], '.', markersize=6, label=lbl)
    for burg, real in zip(burgs, reals_num):
        plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
                   label='Burger field for real ' + str(real))
    plt.legend()
    plt.axis('equal')
    plt.title(args.folder)

    # show
    plt.show()


if __name__ == "__main__":
    main()
