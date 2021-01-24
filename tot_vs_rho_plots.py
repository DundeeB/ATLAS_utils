#!/Local/ph_daniel/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
from correlation_plot import get_corr_files, prepare_lbl
from sys import path

path.append('/srv01/technion/danielab/OOP_hard_sphere_event_chain/')
from post_process import Ising

father_dir = '/storage/ph_daniel/danielab/ECMC_simulation_results3.0'


def parse():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-N', '--N', type=str, nargs='+', help='N values to plot')
    parser.add_argument('-op', '--order_parameter', type=str, nargs='+', help='order parameter to calc sum')
    parser.add_argument('-ic', '--ic', type=str, nargs='+', default=['square', 'honeycomb'])
    parser.add_argument('-he', '--height', type=str, nargs='?', help='h values to plot')
    parser.add_argument('-rho', '--rho', type=str, nargs='?', help='rho range', default=(0.0, 1.0))
    parser.add_argument('-xL', '--xlabel', type=str, nargs='?', default='$\\rho_H$')
    parser.add_argument('-yL', '--ylabel', type=str, nargs='?', default=None)
    parser.add_argument('-E', '--ising_E', type=bool, nargs='?', default=False, const=True)
    parser.add_argument('-k', '--k', type=int, nargs='?', default=4)
    args = parser.parse_args()
    args.N = [int(float(N)) for N in args.N]
    args.height = float(args.height)
    args.rho = [float(r) for r in args.rho.strip('()').split(',')]
    if args.ylabel is None and len(args.order_parameter) == 1:
        args.ylabel = prepare_lbl(args.order_parameter[0], corr=False)
    else:
        args.ylabel = ''
    return args


def choose_folders(args):
    folders, x, ics = [], [], []
    args.rho = np.sort(args.rho)
    for folder in os.listdir(father_dir):
        if not (folder.startswith('N=') and os.path.isdir(folder)):
            continue
        N, h, rhoH, ic = params_from_name(folder)
        if (N in args.N) and (h == args.height) and (rhoH >= args.rho[0]) and (rhoH <= args.rho[1]) and (ic in args.ic):
            folders.append(folder)
            x.append(rhoH)
            ics.append(ic)
    return folders, x, ics


def calc_tot(folder, op, args=None):
    op_dir = os.path.join(father_dir, folder, 'OP', op)
    if op.startswith('gM'):
        corr_file = get_corr_files(op_dir)[0][0]
        n, gM, c = np.loadtxt(os.path.join(op_dir, corr_file), unpack=True)
        N = c[0]
        return 1 / N ** 2 * np.sum(gM * c)
    if op.find('Ising') >= 0:
        if (args is not None) and args.ising_E:
            anneal_mat, _ = get_corr_files(op_dir, 'anneal_')
            if len(anneal_mat) > 0:
                A = np.loadtxt(os.path.join(op_dir, anneal_mat[0]))
            else:
                anneal_reals = get_corr_files(op_dir, 'real_')
                Es, Ms = []
                for real in anneal_reals:
                    J, E, M = np.loadtxt(os.path.join(op_dir, real))
                    Es.append(E)
                    Ms.append(M)
                A = np.transpose([J] + Es + Ms)
            minE = float('inf')
            reals = int((A.shape[1] - 1) / 2)
            for i in range(1, reals + 1):
                m = min(A[:, i])
                if m < minE:
                    minE = m
            return minE
        else:
            ground_states, reals = get_corr_files(op_dir, 'ground_state_')
            ground_state, real = np.loadtxt(os.path.join(op_dir, ground_states[0])), reals[0]
            sp = np.loadtxt(os.path.join(father_dir, folder, str(real)))
            z = [r[2] for r in sp]
            H = 2 * np.mean(z)
            s_z = [(1 if z_ > H / 2 else -1) for z_ in z]
            return np.abs(np.mean([s_ * s_ising for s_, s_ising in zip(s_z, ground_state)]))
    if op == "Graph" and args is not None:
        op = Ising(os.path.join(father_dir, folder), k_nearest_neighbors=args.k, directed=False)
        op.initialize(random_initialization=False, J=-1)
        return op.frustrated_bonds(op.E, op.J)
    psi_file = get_corr_files(op_dir, 'vec_')[0][0]
    psi = np.loadtxt(os.path.join(op_dir, psi_file), dtype=complex)
    if op.startswith('Bragg_S'):
        kx, ky, S_values = psi[:, 0], psi[:, 1], psi[:, 2]
        m = np.argmax(S_values)
        k = [kx[m], ky[m]]
        real = get_corr_files(op_dir, 'vec_')[1][0]
        sp = np.loadtxt(os.path.join(father_dir, folder, str(real)))
        if op.endswith('_S'):
            psi = np.array([np.exp(1j * (k[0] * r[0] + k[1] * r[1])) for r in sp])
        if op.endswith('_Sm'):
            _, h, _, _ = params_from_name(folder)
            rad = 1.0
            lz = (h + 1) * 2 * rad
            psi = np.array(
                [(r[2] - lz / 2) / (lz / 2 - rad) * np.exp(1j * (k[0] * r[0] + k[1] * r[1])) for r in sp])
    return np.abs(np.mean(psi))


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

    plt.figure()
    size = 15
    params = {'legend.fontsize': 'large', 'figure.figsize': (20, 8), 'axes.labelsize': size, 'axes.titlesize': size,
              'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75, 'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.grid()
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)

    folders, x, ics = choose_folders(args)
    choose = lambda folder, ic, N: (params_from_name(folder)[-1] == ic and params_from_name(folder)[0] == N)
    for i, ic in enumerate(args.ic):
        for N in args.N:
            for op in args.order_parameter:
                y = []
                x_ic = []
                for j, folder in enumerate(folders):
                    if not choose(folder, ic, N):
                        continue
                    # try:
                    y.append(calc_tot(folder, op, args))
                    x_ic.append(x[j])
                    # except Exception as err:
                    #     print(err)
                label = 'N=' + str(N) + ', Initial conditions = ' + ic
                if len(args.order_parameter) > 1:
                    label += ', ' + prepare_lbl(op, corr=False)
                I = np.argsort(x_ic)
                x_ic, y = np.array(x_ic)[I], np.array(y)[I]
                plt.plot(x_ic, y, '.-', label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
