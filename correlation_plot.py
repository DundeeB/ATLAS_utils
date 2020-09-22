#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='folders to plot simulation result from')
    parser.add_argument('-l', '--labels', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.-'], help='style of lines')
    parser.add_argument('-mn', '--psis_mn', type=str, nargs='?', default='23', help='mn=14,23,16')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations')
    parser.add_argument('-nbl', '--only_upper', type=bool, nargs='?', const=True, default=False, help='')
    parser.add_argument('-p', '--poly', type=str, nargs='?', const=True, default=False, help='add polynomial decay')
    parser.add_argument('-a', '--all', type=str, nargs='?', const=True, default=False,
                        help='plot all files in OP files (In comparison with just the last configuration)')

    args = parser.parse_args()
    if len(args.style) == 1:
        args.style = [args.style[0] for _ in args.folders]
    if args.labels is None:
        args.labels = args.folders

    m = int(args.psis_mn[0])
    n = int(args.psis_mn[1])
    i = 0

    size = 15
    params = {'legend.fontsize': 'large',
              'figure.figsize': (20, 8),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size * 0.75,
              'ytick.labelsize': size * 0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.figure()

    max_y_psi, max_y_pos, x_psi, x_pos = 0, 0, 0, 0
    for f, s, lbl in zip(args.folders, args.style, args.labels):
        try:
            pattern = 'corr.*'
            phi_files = [file for file in os.listdir(f + '/OP/psi_' + args.psis_mn + '/') if
                         re.match(pattern, file)]
            phi_reals = [int(re.split('\.', re.split('_', file)[-1])[0]) for file in phi_files]
            i_m = np.argmax(phi_reals)
            relevant_files = phi_files if args.all else [phi_files[i_m]]
            relevant_reals = phi_reals if args.all else [phi_reals[i_m]]
            for corr_file, real in zip(relevant_files, relevant_reals):
                lbl_ = lbl + ' ' + str(real) if args.all else lbl
                x, y = np.loadtxt(f + '/OP/psi_' + args.psis_mn + '/' + corr_file, usecols=(0, 1), unpack=True)
                if not args.only_upper:
                    plt.loglog(x, y, s, label=lbl + ', $\psi_{' + args.psis_mn + '}$', linewidth=2, markersize=6)
                if np.nanmax(y) > max_y_psi:
                    max_y_psi = np.nanmax(y)
                    x_psi = x[np.nanargmax(y)]
                if args.upper:
                    up_fold = 'upper_psi_1' + str(m * n)
                    x, y = np.loadtxt(f + '/OP/' + up_fold + '/' + corr_file, usecols=(0, 1), unpack=True)
                    plt.loglog(x, y, s, label=lbl + ', upper layer $\psi_{1' + str(m * n) + '}$', linewidth=2,
                               markersize=6)
                i += 1
        except Exception as err:
            print(err)
    if args.poly:
        plt.loglog(x, max_y_psi * np.power(np.array(x) / x_psi, -1.0 / 4), '--', label='$x^{-1/4}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$r [$\sigma$=2]')
    plt.ylabel('Orientational correlation')
    plt.show()


if __name__ == "__main__":
    main()
