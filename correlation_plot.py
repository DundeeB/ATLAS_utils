#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re


def get_corr_files(OP_sub_dir):
    phi_files = [corr_file for corr_file in os.listdir(OP_sub_dir) if re.match('corr.*', corr_file)]
    phi_reals = [int(re.split('\.', re.split('_', corr_file)[-1])[0]) for corr_file in phi_files]
    i_m = np.argmax(phi_reals)
    return phi_files[i_m], phi_reals[i_m], phi_files, phi_reals


def prepare_lbl(lbl):
    lbl = re.sub('_', ' ', lbl)
    for mn in ['14', '23', '16']:
        lbl = re.sub('psi ' + mn, '$\\\psi_{' + mn + '}\\$', lbl)
    lbl = re.sub('rho', '$\\rho$', lbl)
    lbl = re.sub('Bragg_Sm', '$S_m(k^{peak})$', lbl)
    lbl = re.sub('Bragg_S', '$S(k^{peak})$', lbl)
    return lbl


def main():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='folders to plot simulation result from')
    parser.add_argument('-l', '--labels', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.-'], help='style of lines')
    parser.add_argument('-mn', '--psis_mn', type=str, nargs='?', default=None, help='mn=14,23,16')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations for psi_mn')
    parser.add_argument('-bs', '--bragg_s', type=bool, nargs='?', const=True, default=False,
                        help='plot Bragg correlation e^ikr at k peak')
    parser.add_argument('-bsm', '--bragg_sm', type=bool, nargs='?', const=True, default=False,
                        help='plot magnetic Bragg correlation z*e^ikr at k peak of magentic')
    parser.add_argument('-nbl', '--only_upper', type=bool, nargs='?', const=True, default=False, help='')
    parser.add_argument('-a', '--all', type=str, nargs='?', const=True, default=False,
                        help='plot all files in OP files (In comparison with just the last configuration)')

    args = parser.parse_args()
    if len(args.style) == 1:
        args.style = [args.style[0] for _ in args.folders]
    if args.labels is None:
        args.labels = args.folders

    op_dirs = []
    if args.psis_mn is not None:
        m = int(args.psis_mn[0])
        n = int(args.psis_mn[1])
        if not args.only_upper:
            op_dirs.append('psi_' + args.psis_mn)
        if args.upper or args.only_upper:
            op_dirs.append('upper_psi_1' + str(m * n))
    if args.bragg_s:
        op_dirs.append('Bragg_S')
    if args.bragg_sm:
        op_dirs.append('Bragg_Sm')

    size = 15
    params = {'legend.fontsize': 'large', 'figure.figsize': (20, 8), 'axes.labelsize': size, 'axes.titlesize': size,
              'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75, 'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.figure()
    for f, s, lbl in zip(args.folders, args.style, args.labels):
        for op_dir in op_dirs:
            relevant_files = []
            relevant_reals = []
            try:
                last_file, last_real, phi_files, phi_reals = get_corr_files(f + '/OP/' + op_dir + '/')
                if args.all:
                    for corr_file, real in zip(phi_files, phi_reals):
                        relevant_files.append(corr_file)
                        relevant_reals.append(real)
                else:
                    relevant_files.append(last_file)
                    relevant_reals.append(last_real)
                for corr_file, real in zip(relevant_files, relevant_reals):
                    lbl_ = lbl + ', ' + op_dir
                    if args.all:
                        lbl_ = lbl_ + ', real ' + str(real)
                    corr_path = f + '/OP/' + op_dir + '/' + corr_file
                    try:
                        x, y = np.loadtxt(corr_path, usecols=(0, 1), unpack=True)
                    except ValueError:
                        x, y = np.loadtxt(corr_path, usecols=(0, 1), unpack=True, dtype=complex)
                        x, y = np.abs(x), np.abs(y)
                    plt.loglog(x, y, s, label=prepare_lbl(lbl_), linewidth=2, markersize=6)
            except Exception as err:
                print(err)
    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$r [$\sigma$=2]')
    plt.ylabel('Orientational correlation')
    plt.show()


if __name__ == "__main__":
    main()
