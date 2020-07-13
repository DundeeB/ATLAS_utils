#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--files', type=str, nargs='+', help='files to read data and plot from')
    parser.add_argument('-x', '--x_column', type=int, nargs='*', default=[1], help='x column to plot')
    parser.add_argument('-y', '--y_column', type=int, nargs='*', default=[2], help='y column to plot')
    parser.add_argument('-xL', '--x_label', type=str, nargs='?', default='x', help='x label')
    parser.add_argument('-yL', '--y_label', type=str, nargs='?', default='y', help='y label')
    parser.add_argument('-l', '--legends', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.'], help='legends of plot')
    parser.add_argument('-eq', '--equal', type=bool, nargs='?', const=True, default=False, help='axis equal')
    parser.add_argument('-mn', '--psis_mn', type=str, nargs='*', default='23', help='mn=14 or 23')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations')

    args = parser.parse_args()
    n_xy = max([len(args.x_column), len(args.y_column)])
    if len(args.x_column) == 1:
        args.x_column = [args.x_column[0] for _ in range(n_xy)]
    if len(args.y_column) == 1:
        args.y_column = [args.y_column[0] for _ in range(n_xy)]
    if len(args.style) == 1:
        args.style = [args.style[0] for _ in range(n_xy)]

    m = int(args.psis_mn[0])
    n = int(args.psis_mn[1])
    i = 0
    plt.figure()
    for f in args.files:
        corr_file = lambda s: f + '/OP/' + [file for file in os.listdir(f + '/OP/') if re.match(s, file)][0]
        for x_col, y_col, s in zip(args.x_column, args.y_column, args.style):
            plt.subplot(211)
            x, y = np.loadtxt(corr_file('psi_' + args.psis_mn + '_corr.*'), usecols=(x_col - 1, y_col - 1), unpack=True)
            lbl = f if args.legends is None else args.legends[i]
            plt.loglog(x, y, s, label=lbl + ', $\psi$_{' + args.psis_mn + '}', linewidth=2, markersize=6)
            if args.upper:
                x, y = np.loadtxt(corr_file('upper_psi_1' + str(m * n) + '_corr.*'), usecols=(x_col - 1, y_col - 1),
                                  unpack=True)
                plt.loglog(x, y, s, label=lbl + ', upper layer $\psi$_{1' + str(m * n) + '}', linewidth=2, markersize=6)

            plt.subplot(212)
            x, y = np.loadtxt(corr_file('positional_theta=.*'), usecols=(x_col - 1, y_col - 1), unpack=True)
            plt.loglog(x, y - 1, s, label=lbl + ', g($\Delta$x,0)', linewidth=2, markersize=6)
            if args.upper:
                x, y = np.loadtxt(corr_file('upper_positional_theta=.*'), usecols=(x_col - 1, y_col - 1), unpack=True)
                plt.loglog(x, y - 1, s, label=lbl + ', upper layer g($\Delta$x,0)', linewidth=2, markersize=6)

            i += 1
    plt.subplot(211)
    plt.grid()
    plt.legend()
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    size = 15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    if args.equal:
        plt.axis('equal')

    plt.subplot(212)
    plt.grid()
    plt.legend()
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    size = 15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    if args.equal:
        plt.axis('equal')

    plt.show()


if __name__ == "__main__":
    main()
