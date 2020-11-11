#!/Local/cmp/anaconda3/bin/python -u
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import join
import os
import re


def main():
    # parse
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folder', type=str, nargs='?', help='folders to plot simulation result from')
    parser.add_argument('-r', '--real', type=str, nargs='?', default='',
                        help='Realization to plot. Default is last realization')
    parser.add_argument('-z', '--z_colour', type=bool, nargs='?', const=False, default=True,
                        help='colour upper and lower spheres with different colours')

    # handle defaults and load data
    args = parser.parse_args()
    burg_fold = join(args.folder, 'OP/burger_vectors')
    if args.real == '':
        reals = [int(re.split('(vec_|\.txt)', r)[2]) for r in os.listdir(burg_fold) if re.search('vec_\\d*.txt', r)]
        args.real = 'vec_' + str(max(reals)) + '.txt'
    real_num = int(re.split('(vec_|\.txt)', args.real)[2])
    real_fold = join(burg_fold, args.real)
    burg = np.loadtxt(real_fold)
    sp = np.loadtxt(join(args.folder, str(real_num)))

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
    plt.xlabel('x')
    plt.ylabel('y')
    lbl = 'Center\'s xy for realization ' + str(real_num)
    if args.z_colour:
        up = np.where(sp[:, 2] > np.mean(sp[:, 2]))
        down = np.where(sp[:, 2] <= np.mean(sp[:, 2]))
        plt.plot(sp[:, 0][up], sp[:, 1][up], '.', markersize=6, label=lbl + ' up')
        plt.plot(sp[:, 0][down], sp[:, 1][down], '.', markersize=6, label=lbl + ' down')
    else:
        plt.plot(sp[:, 0], sp[:, 1], '.', markersize=6, label=lbl)
    plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy', scale=1,
               label='Burger field')
    plt.legend()
    plt.axis('equal')

    # show
    plt.show()


if __name__ == "__main__":
    main()
