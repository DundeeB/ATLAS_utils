#!/Local/ph_daniel/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re


def get_corr_files(OP_sub_dir, prefix='correlation_', reverse=True):
    phi_files = [corr_file for corr_file in os.listdir(OP_sub_dir) if corr_file.startswith(prefix)]
    phi_reals = [int(re.split('\.', re.split('_', corr_file)[-1])[0]) for corr_file in phi_files]
    sorted_phi_files = [f for _, f in sorted(zip(phi_reals, phi_files), reverse=reverse)]
    sorted_phi_reals = sorted(phi_reals, reverse=reverse)
    return sorted_phi_files, sorted_phi_reals


def prepare_lbl(lbl, corr=True):
    lbl = re.sub('_', ' ', lbl)
    lbl = re.sub('rhoH', '$\\\\rho_H$', lbl)
    if corr:
        for mn in ['14', '23', '16']:
            lbl = re.sub('psi ' + mn, '$g_{' + mn + '}$', lbl)
        lbl = re.sub('Bragg Sm', '$g_k^M$', lbl)
        lbl = re.sub('Bragg S', '$g_k$', lbl)
    else:
        for mn in ['14', '23', '16']:
            lbl = re.sub('psi ' + mn, '$\\\Psi_{' + mn + '}$', lbl)
        lbl = re.sub('Bragg Sm', '$\\\Psi_k^M$', lbl)
        lbl = re.sub('Bragg S', '$\\\Psi_k$', lbl)
        lbl = re.sub('gM', 'M', lbl)
    for N, N_ in zip(['10000', '40000', '90000'], ['1e4', '4e4', '9e4']):
        lbl = re.sub(N, N_, lbl)
    lbl = re.sub('\,', ',', lbl)
    return lbl


def main():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--folders', type=str, nargs='+', help='folders to plot simulation result from')
    parser.add_argument('-l', '--labels', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['-'], help='style of lines')

    parser.add_argument('-mn', '--psis_mn', type=str, nargs='?', default=None, help='mn=14,23,16')
    parser.add_argument('-gMd', '--gMd', type=str, nargs='?', default=None, help='k_nearest_neighbors=3,4,6 directed')
    parser.add_argument('-gMud', '--gMud', type=str, nargs='?', default=None,
                        help='k_nearest_neighbors=3,4,6 undirected')
    parser.add_argument('-up', '--upper', type=bool, nargs='?', const=True, default=False,
                        help='plot upper correlations for psi_mn')
    parser.add_argument('-bs', '--bragg_s', type=bool, nargs='?', const=True, default=False,
                        help='plot Bragg correlation e^ikr at k peak')
    parser.add_argument('-bsm', '--bragg_sm', type=bool, nargs='?', const=True, default=False,
                        help='plot magnetic Bragg correlation z*e^ikr at k peak of magentic')
    parser.add_argument('-pos', '--pos', type=bool, nargs='?', const=True, default=False,
                        help='plot positional g(r) at an angle theta calculate from the maximal psi')

    parser.add_argument('-nbl', '--only_upper', type=bool, nargs='?', const=True, default=False, help='')
    parser.add_argument('-r', '--reals', type=int, nargs='?', default=1, help='number of realization')
    parser.add_argument('-abs', '--abs', type=bool, nargs='?', const=True, default=False,
                        help='plot |corr|')
    parser.add_argument('-pol', '--pol', type=bool, nargs='?', const=True, default=False,
                        help='Show polynomial slope -1/4 for orientational -1/3 for positional')
    parser.add_argument('-reverse', '--reverse', type=bool, nargs='?', const=False, default=True,
                        help='Sum realizations backwards')
    parser.add_argument('-max', '--max', type=bool, nargs='?', const=True, default=False,
                        help='Plot polynomial fit with max instead of mean')
    parser.add_argument('-lgy', '--semilogy', type=bool, nargs='?', const=True, default=False,
                        help='Semilogy plot instead of log log')

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
    if args.gMd is not None:
        op_dirs.append('gM_k=' + args.gMd + '_directed')
    if args.gMud is not None:
        op_dirs.append('gM_k=' + args.gMud + '_undirected')

    size = 15
    params = {'legend.fontsize': 'large', 'figure.figsize': (20, 8), 'axes.labelsize': size, 'axes.titlesize': size,
              'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75, 'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.figure()
    maxys, maxxs, slopes = [], [], []
    for f, s, lbl in zip(args.folders, args.style, args.labels):
        for op_dir in op_dirs:
            try:
                lbl_ = lbl + ', ' + op_dir
                phi_files, _ = get_corr_files(f + '/OP/' + op_dir + '/', reverse=args.reverse)
                corr_path = lambda corr_file: f + '/OP/' + op_dir + '/' + corr_file
                x, y, counts = np.loadtxt(corr_path(phi_files[0]), usecols=(0, 1, 2), unpack=True)
                reals = min(args.reals, len(phi_files) - 1)
                if reals > 1:
                    counts_sum = counts
                    y_sum = y * counts if op_dirs != 'pos' else y
                    for i in range(1, reals):
                        _, y, counts = np.loadtxt(corr_path(phi_files[i]), usecols=(0, 1, 2), unpack=True)
                        y[np.where(np.isnan(y))] = 0
                        l = min(len(y), len(y_sum))
                        y_sum[:l] += y[:l] * counts[:l] if op_dirs != 'pos' else y[:l]
                        counts_sum[:l] += counts[:l]
                    lbl_ += ' mean of ' + str(reals) + ' realizations'
                    y = y_sum / counts_sum if op_dirs != 'pos' else y_sum / reals
                if args.abs:
                    y = np.abs(y)
                if op_dir == "pos":
                    y = y - 1
                y[np.where(y <= 0)] = np.nan
                kwargsplt = {'label': prepare_lbl(lbl_), 'linewidth': 2, 'markersize': 6}
                argsplt = [x, y, s]
                if not args.semilogy:
                    plt.loglog(*argsplt, **kwargsplt)
                else:
                    plt.semilogy(*argsplt, **kwargsplt)
                if args.pol:
                    if args.max:
                        maxys.append(np.nanmax(y))
                        maxxs.append(x[np.nanargmax(y)])
                    else:
                        I = np.where(np.logical_and(x > 1, x < 3))
                        maxys.append(np.nanmean(y[I]))
                        maxxs.append(2)
                    cond = lambda x, y: x > 10 and x < 20 and (not np.isnan(y))
                    y_p = np.array([y_ for x_, y_ in zip(x, y) if cond(x_, y_)])
                    x_p = np.array([x_ for x_, y_ in zip(x, y) if cond(x_, y_)])
                    p = np.polyfit(np.log(x_p), np.log(y_p), 1)
                    slopes.append(-p[0])
            except Exception as err:
                print(err)
    if args.pol:
        I = np.argsort(slopes)
        maxys = np.array(maxys)[I]
        maxxs = np.array(maxxs)[I]
        slopes = np.array(slopes)[I]
        for slope, plot_corr_type in zip([1.0 / 3.0, 1.0 / 4.0],
                                         [args.pos or args.bragg_s or args.bragg_sm, args.psis_mn is not None]):
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
            argsplt = [x, y, '--']
            kwargs = {'label': 'polynomial fit with slope ' + str(slope), 'linewidth': 2}
            if not args.semilogy:
                plt.loglog(*argsplt, **kwargs)
            else:
                plt.semilogy(*argsplt, **kwargs)

    plt.grid()
    plt.legend()
    plt.xlabel('$\Delta$r [$\sigma$=2]')
    plt.ylabel('Correlation $<\\psi\\psi^*>$' if not args.pos else 'g(r)-1')
    plt.show()


if __name__ == "__main__":
    main()
