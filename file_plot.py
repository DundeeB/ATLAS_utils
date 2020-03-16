#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='plot options')
parser.add_argument('-f', '--files', type=str, nargs='+', help='files to read data and plot from')
parser.add_argument('-x', '--x_column', type=int, nargs='*', default=1,  help='x column to plot')
parser.add_argument('-y', '--y_column', type=int, nargs='*', default=2,  help='y column to plot')
parser.add_argument('-xL', '--x_label', type=str, nargs='?', default='x', help='x label')
parser.add_argument('-yL', '--y_label', type=str, nargs='?', default='y', help='y label')
parser.add_argument('-l', '--legends', type=str, nargs='*', help='legends of plot')
parser.add_argument('-s', '--style', type=str, nargs='?', default='-', help='legends of plot')

args = parser.parse_args()
for f in args.files:
    # x, y = np.loadtxt(args.files, usecols=(args.x_column, args.y_column), unpack=True)
    # plt.plot(x, y, args.style)
    A = np.loadtxt(args.files)
    plt.plot(A[1], A[2], args.style)
plt.show()
