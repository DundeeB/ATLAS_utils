#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re


# TODO plot realization averged correlation based on all existing data
# TODO add again the polynomial slope calculation

def get_corr_files(OP_sub_dir):
    phi_files = [corr_file for corr_file in os.listdir(OP_sub_dir) if re.match('corr.*', corr_file)]
    phi_reals = [int(re.split('\.', re.split('_', corr_file)[-1])[0]) for corr_file in phi_files]
    i_m = np.argmax(phi_reals)
    return phi_files[i_m], phi_reals[i_m], phi_files, phi_reals


def prepare_lbl(lbl):
    lbl = re.sub('_', ' ', lbl)
    for mn in ['14', '23', '16']:
        lbl = re.sub('psi ' + mn, '$\\\psi_{' + mn + '}$', lbl)
    lbl = re.sub('rhoH', '$\\\\rho_H$', lbl)
    lbl = re.sub('Bragg Sm', '$S_m(k^{peak})$', lbl)
    lbl = re.sub('Bragg S', '$S(k^{peak})$', lbl)
    for N, N_ in zip(['10000', '40000', '90000'], ['1e4', '4e4', '9e4']):
        lbl = re.sub(N, N_, lbl)
    lbl = re.sub('\,', ',', lbl)
    return lbl


def main():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='folders to plot simulation result from')
    parser.add_argument('-l', '--labels', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.-'], help='style of lines')

    parser.add_argument('-mn', '--psis_mn', type=str, nargs='?', default=None, help='mn=14,23,16')
    parser.add_argument('-mnr', '--mn_reals', type=str, nargs='?', default=None,
                        help='convergence for all realizations of the psi_mn given')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations for psi_mn')
    parser.add_argument('-bs', '--bragg_s', type=bool, nargs='?', const=True, default=False,
                        help='plot Bragg correlation e^ikr at k peak')
    parser.add_argument('-bsm', '--bragg_sm', type=bool, nargs='?', const=True, default=False,
                        help='plot magnetic Bragg correlation z*e^ikr at k peak of magentic')
    parser.add_argument('-pos', '--pos', type=bool, nargs='?', const=True, default=False,
                        help='plot positional g(r) at an angle theta calculate from the maximal psi')

    parser.add_argument('-nbl', '--only_upper', type=bool, nargs='?', const=True, default=False, help='')
    parser.add_argument('-a', '--all', type=bool, nargs='?', const=True, default=False,
                        help='plot all files in OP files (In comparison with just the last configuration)')
    parser.add_argument('-abs', '--abs', type=bool, nargs='?', const=True, default=False,
                        help='plot |corr|')
    parser.add_argument('-pol', '--pol', type=bool, nargs='?', const=True, default=False,
                        help='Show polynomial slope -1/4 for orientational -1/3 for positional')

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
    if args.pos:
        op_dirs.append('pos')

    size = 15
    params = {'legend.fontsize': 'large', 'figure.figsize': (20, 8), 'axes.labelsize': size, 'axes.titlesize': size,
              'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75, 'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.figure()
    maxys, maxxs, slopes = [], [], []
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
                        if args.abs:
                            y = np.abs(y)
                        if op_dir == "pos":
                            y = y - 1
                    except ValueError:
                        x, y = np.loadtxt(corr_path, usecols=(0, 1), unpack=True, dtype=complex)
                        x, y = np.abs(x), np.abs(y)
                    I = np.where(y < 0)
                    y[I] = np.nan
                    plt.loglog(x, y, s, label=prepare_lbl(lbl_), linewidth=2, markersize=6)
                    if args.pol:
                        maxys.append(np.nanmax(y))
                        maxxs.append(x[np.argmax(y)])
                        I = np.where(np.logical_not(np.isnan(y)))
                        p = np.polyfit(x[I], np.log(y[I]), 1)
                        slopes.append(p[0])
            except Exception as err:
                print(err)
    if args.pol:
        I = np.argsort(slopes)
        maxys = np.array(maxys)[I]
        maxxs = np.array(maxxs)[I]
        slopes = np.array(slopes)[I]
        for slope, plot_corr_type in zip([1.0 / 3.0, 1.0 / 4.0], [args.pos, args.mn is not None]):
            if not plot_corr_type:
                continue
            if min(slopes) > slope:
                y_init = min(maxys)
                x_init = maxxs[np.argmin(maxys)]
            else:
                if max(slopes) < slope:
                    y_init = max(maxys)
                    x_init = maxxs[np.argmax(maxys)]
                else:
                    i = np.where(slopes > slope)[0][0]
                    y_init = maxys[i - 1]
                    x_init = maxxs[i - 1]
            y = y_init * np.power(x / x_init, -slope)
            plt.loglog(x, y, label='polynomial fit with slope ' + str(slope), '--', linewidth=2)

    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$r [$\sigma$=2]')
    plt.ylabel('Correlation $<\\psi\\psi^*>$' if not args.pos else 'g(r)-1')
    plt.show()


if __name__ == "__main__":
    main()
