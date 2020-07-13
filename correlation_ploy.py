#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
parser.add_argument('-up', '--upper', type=bool, nargs='?', default=True, default=False, help='plot upper correlations')

args = parser.parse_args()
n_xy = max([len(args.x_column), len(args.y_column)])
if len(args.x_column) == 1:
    args.x_column = [args.x_column[0] for _ in range(n_xy)]
if len(args.y_column) == 1:
    args.y_column = [args.y_column[0] for _ in range(n_xy)]
if len(args.style) == 1:
    args.style = [args.style[0] for _ in range(n_xy)]

i = 0
plt.figure()
for f in args.files:
    for x_col, y_col, s in zip(args.x_column, args.y_column, args.style):
        x, y = np.loadtxt(f+'OP/psi_'+arg.psis_mn+'_corr*', usecols=(x_col - 1, y_col - 1), unpack=True)
        if args.legends is None:
            lbl = f + ', $\psi$_{'+arg.psis_mn+'}'
        else:
            lbl = args.legends[i] + ', $\psi$_{'+arg.psis_mn+'}'
        plt.subplot(211)
        plt.loglog(x, y, s, label=lbl, linewidth=2, markersize=6)
        if args.upper:
            

        x, y = np.loadtxt(f+'OP/upper_psi_*corr*', usecols=(x_col - 1, y_col - 1), unpack=True)
        if args.legends is None:
            lbl = f + ', $\psi$_{23}'
        else:
            lbl = args.legends[i]

        i += 1
plt.grid()
plt.legend()
plt.xlabel(args.x_label)
plt.ylabel(args.y_label)
size=15
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)
if args.equal:
    plt.axis('equal')
plt.show()

