#!/Local/ph_daniel/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sys import path
import os

path.append('/srv01/technion/danielab/OOP_hard_sphere_event_chain/')
from post_process import Graph


# TODO: Add xy plot coloring by correlation of up-down to ising's ground state
# TODO: proper Bragg peak 2d illustration

def parse():
    parser = argparse.ArgumentParser(description='plot options')
    parser.add_argument('-f', '--files', type=str, nargs='+', help='files to read data and plot from')
    parser.add_argument('-x', '--x_column', type=int, nargs='*', default=[1], help='x column to plot')
    parser.add_argument('-y', '--y_column', type=int, nargs='*', default=[2], help='y column to plot')
    parser.add_argument('-xL', '--x_label', type=str, nargs='?', default='x', help='x label')
    parser.add_argument('-yL', '--y_label', type=str, nargs='?', default='y', help='y label')
    parser.add_argument('-l', '--legends', type=str, nargs='*', help='legends of plot')
    parser.add_argument('-L', '--leg_loc', type=int, nargs='?', help='legends location, 0 turn of legend', default=1)
    parser.add_argument('-s', '--style', type=str, nargs='*', default=['.'], help='legends of plot')
    parser.add_argument('-neq', '--not_equal', type=bool, nargs='?', const=True, default=False, help='axis equal')
    parser.add_argument('-lg', '--loglog', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-lgy', '--semilogy', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-lgx', '--semilogx', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-m1', '--minus_one', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-abs', '--abs', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-z', '--z_colour', type=bool, nargs='?', const=True, default=False,
                        help='colour upper and lower spheres with different colours')
    parser.add_argument('-b', '--bonds', type=int, nargs='?', help='Plot bonds using k nearest neighbors')
    parser.add_argument('-fb', '--frustrated_bonds', type=int, nargs='?', default=0,
                        help='1 - Plot frustrated bonds only. 2 - dont even plot spheres.')
    parser.add_argument('-burg', '--burg', type=bool, nargs='?', default=False, const=True,
                        help='Quiver burger file if exists')
    parser.add_argument('-yscaling', '--yscaling', type=float, nargs='*', default=[1], help='y scaling')
    parser.add_argument('-folder_grid_off', '--folder_grid_off', type=bool, nargs='?', const=True, default=False,
                        help='For a list of folders and lists of say x_coloumns, if folder_grid_off then instead of' +
                             ' running on all folders and for each plotting x_cloumns, we run on zip(folders,x_coulmn)')
    return parser.parse_args()


def plot_params(args, f, x_col, y_col, s, yscale, sim_path, real):
    try:
        x, y = np.loadtxt(f, usecols=(x_col - 1, y_col - 1), unpack=True)
    except ValueError:
        x, y = np.loadtxt(f, usecols=(x_col - 1, y_col - 1), unpack=True, dtype=complex)
        x, y = np.abs(x), np.abs(y)
    except OSError:
        print("OSError for " + f + ", probably file does not exist")
        return
    y = y - 1 if args.minus_one else y
    y = np.abs(y) if args.abs else y
    y *= yscale
    lbl = f + ', x_col=' + str(x_col) + ', y_col=' + str(y_col) if args.legends is None else args.legends[i]
    plotted = False
    if args.loglog:
        plt.loglog(x, y, s, label=lbl, linewidth=2, markersize=6)
        plotted = True
    if args.semilogy:
        plt.semilogy(x, y, s, label=lbl, linewidth=2, markersize=6)
        plotted = True
    if args.semilogx:
        plt.semilogx(x, y, s, label=lbl, linewidth=2, markersize=6)
        plotted = True
    if plotted:
        return
    if args.z_colour or (args.bonds is not None):
        x, y, z = np.loadtxt(f, usecols=(0, 1, 2), unpack=True)
        if args.frustrated_bonds < 2:
            up = np.where(z > np.mean(z))
            down = np.where(z <= np.mean(z))
            plt.plot(x[up], y[up], s, label=lbl, linewidth=2, markersize=6)
            plt.plot(x[down], y[down], s, label=lbl, linewidth=2, markersize=6)
        if args.bonds is not None:
            op = Graph(sim_path=sim_path, k_nearest_neighbors=args.bonds, directed=False,
                       centers=[r for r in zip(x, y, z)], spheres_ind=real)
            op.calc_graph()
            graph = op.graph
            spins = op.op_vec
            for i in range(len(x)):
                for j in graph.getrow(i).indices:
                    ex = [x[i], x[j]]
                    ey = [y[i], y[j]]
                    if (ex[1] - ex[0]) ** 2 + (ey[1] - ey[0]) ** 2 > 10 ** 2:
                        continue
                    if spins[i] * spins[j] > 0:
                        plt.plot(ex, ey, 'r-')
                    if (args.frustrated_bonds == 0) and (spins[i] * spins[j] < 0):
                        plt.plot(ex, ey, 'g-', linewidth=0.1)
    else:
        plt.plot(x, y, s, label=lbl, linewidth=2, markersize=6)
    if args.burg:
        burg_file = os.path.join(sim_path, 'OP/burger_vectors', 'vec_' + str(real) + '.txt')
        burg = np.loadtxt(burg_file)
        plt.quiver(burg[:, 0], burg[:, 1], burg[:, 2], burg[:, 3], angles='xy', scale_units='xy',
                   scale=1, label='Burger field for real ' + str(real))


def main():
    args = parse()
    params = [args.x_column, args.y_column, args.style, args.yscaling]
    if args.folder_grid_off:
        params += [args.files]
    n_xy = max([len(l) for l in params])
    for i in range(len(params)):
        if len(params[i]) == 1:
            params[i] = [params[i][0] for _ in range(n_xy)]

    if args.folder_grid_off:
        for f, x_col, y_col, s, yscale in zip(args.files, args.x_column, args.y_column, args.style, args.yscaling):
            sim_path = os.path.dirname(os.path.abspath(f))
            real = os.path.basename(f)
            plot_params(args, f, x_col, y_col, s, yscale, sim_path, real)
    else:
        for f in args.files:
            sim_path = os.path.dirname(os.path.abspath(f))
            real = os.path.basename(f)
            for x_col, y_col, s, yscale in zip(args.x_column, args.y_column, args.style, args.yscaling):
                plot_params(args, f, x_col, y_col, s, yscale, sim_path, real)
    plt.grid()
    if args.leg_loc > 0:
        plt.legend(loc=args.leg_loc)
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
    if not args.not_equal:
        plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
