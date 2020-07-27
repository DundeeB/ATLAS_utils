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
    parser.add_argument('-l', '--legends', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.-'], help='legends of plot')
    parser.add_argument('-eq', '--equal', type=bool, nargs='?', const=True, default=False, help='axis equal')
    parser.add_argument('-mn', '--psis_mn', type=str, nargs='?', default='23', help='mn=14 or 23')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations')
    parser.add_argument('-nbl', '--no_bilayer', type=bool, nargs='?', const=True, default=False,
                        help='')
    parser.add_argument('-p', '--poly', type=str, nargs='?', const=True, default=False, help='add polynomial decay')
    parser.add_argument('-a', '--all', type=str, nargs='?', const=True, default=False,
                        help='plot all files in OP files or just the last configuration analysis')

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
    max_y_psi, max_y_pos, x_psi, x_pos = 0, 0, 0, 0
    for f in args.files:
        try:
            phi_files = lambda s: [file for file in os.listdir(f + '/OP/') if re.match(s, file)]
            phi_reals = lambda s: [int(re.split('\.', re.split('_', file)[-1])[0]) for file in relevent_files(s)]
            relevent_files = lambda s: phi_files(s) if args.all else [phi_files(s)[-1]]
            relevent_reals = lambda s: phi_reals(s) if args.all else [phi_reals(s)[-1]]
            for x_col, y_col, s in zip(args.x_column, args.y_column, args.style):
                plt.subplot(211)
                s = 'psi_' + args.psis_mn + '_corr.*'
                for corr_file, real in zip(relevent_files(s), relevent_reals(s)):
                    lbl = f if args.legends is None else args.legends[i]
                    print(corr_file)
                    if args.all:
                        lbl += ' ' + str(real)
                    x, y = np.loadtxt(f + '/OP/' + corr_file, usecols=(x_col - 1, y_col - 1),
                                      unpack=True)
                    if not args.no_bilayer:
                        plt.loglog(x, y, s, label=lbl + ', $\psi_{' + args.psis_mn + '}$', linewidth=2, markersize=6)
                    if np.nanmax(y) > max_y_psi:
                        max_y_psi = np.nanmax(y)
                        x_psi = x[np.nanargmax(y)]
                    if args.upper:
                        s = 'upper_psi_1' + str(m * n) + '_corr.*'
                        corr_file = relevent_files(s)[np.find(real == relevent_reals(s))]
                        x, y = np.loadtxt(corr_file, usecols=(x_col - 1, y_col - 1), unpack=True)
                        plt.loglog(x, y, s, label=lbl + ', upper layer $\psi_{1' + str(m * n) + '}$', linewidth=2,
                                   markersize=6)
                plt.subplot(212)
                s = 'positional_theta=.*'
                for corr_file, real in zip(relevent_files(s), relevent_reals(s)):
                    x, y = np.loadtxt(corr_file, usecols=(x_col - 1, y_col - 1), unpack=True)
                    lbl = f if args.legends is None else args.legends[i]
                    if args.all:
                        lbl += ' ' + str(real)
                    if np.nanmax(y) > max_y_pos:
                        max_y_pos = np.nanmax(y)
                        x_pos = x[np.nanargmax(y)]
                    if not args.no_bilayer:
                        plt.loglog(x, y - 1, s, label=lbl + ', g($\Delta$x,0)', linewidth=2, markersize=6)
                    if args.upper:
                        s = 'upper_positional_theta=.*'
                        corr_file = relevent_files(s)[np.find(real == relevent_reals(s))]
                        x, y = np.loadtxt(corr_file, usecols=(x_col - 1, y_col - 1),
                                          unpack=True)
                        plt.loglog(x, y - 1, s, label=lbl + ', upper layer g($\Delta$x,0)', linewidth=2, markersize=6)

                i += 1
        except Exception as err:
            print(err)
    plt.subplot(211)
    if args.poly:
        plt.loglog(x, max_y_psi * np.power(np.array(x) / x_psi, -1.0 / 4), '--', label='$x^{-1/4}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$r [$\sigma$=2]')
    plt.ylabel('Orientational correlation')
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
    if args.poly:
        plt.loglog(x, max_y_pos * np.power(np.array(x) / x_pos, -1.0 / 3), '--', label='$x^{-1/3}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$x [$\sigma$=2]')
    plt.ylabel('Positional correlation')
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
