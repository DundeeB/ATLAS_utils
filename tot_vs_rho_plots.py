#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
from correlation_plot import get_corr_files

father_dir = '/storage/ph_daniel/danielab/ECMC_simulation_results3.0'


def parse():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-N', '--N', type=str, nargs='?', help='N values to plot')
    parser.add_argument('-he', '--height', type=str, nargs='?', help='h values to plot')
    parser.add_argument('-rho', '--rho', type=str, nargs='?', help='rho range', default=(0.0, 1.0))
    parser.add_argument('-op', '--order_parameter', type=str, nargs='?', help='order parameter to calc sum')
    args = parser.parse_args()
    args.N = int(args.N)
    args.height = float(args.height)
    args.rho = [float(r) for r in args.rho.strip('()').split(',')]
    return args


def choose_folders(args):
    folders, x = [], []
    args.rho = np.sort(args.rho)
    for folder in os.listdir(father_dir):
        N, h, rhoH, ic = params_from_name(folder)
        if (N == args.N) and (h == args.height) and (rhoH >= args.rho[0]) and (rhoH <= args.rho[1]):
            folders.append(folder)
            x.append(rhoH)
    return folders, x


def calc_tot(folder, args):
    op_dir = os.path.join(father_dir, 'OP', args.order_parameter)
    psi_file = get_corr_files(op_dir, 'vec_')[0][0]
    psi = np.loadtxt(os.path.join(op_dir, psi_file), dtype=complex)
    return np.abs(np.sum(psi))


def labels(args):
    return '', '', ''


def params_from_name(name):
    ss = re.split("[_=]", name)
    for i, s in enumerate(ss):
        if s == 'N':
            N = int(ss[i + 1])
        if s == 'h':
            h = float(ss[i + 1])
        if s == 'rhoH':
            rhoH = float(ss[i + 1])
        if s == 'triangle' or s == 'square':
            ic = s
            if ss[i - 1] == 'AF' and s == 'triangle':
                ic = 'honeycomb'
    return N, h, rhoH, ic


def main():
    args = parse()
    folders, x = choose_folders(args)
    y = [calc_tot(folder, args) for folder in folders]
    xlabel, ylabel, label = labels(args)

    size = 15
    params = {'legend.fontsize': 'large', 'figure.figsize': (20, 8), 'axes.labelsize': size, 'axes.titlesize': size,
              'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75, 'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.figure()
    plt.plot(x, y, label=label)
    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    main()
