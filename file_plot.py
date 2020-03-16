#!/Local/cmp/anaconda3/bin/python -u

import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 1:
    print("Error: too few arguments!")
    print("Usage: ./plot.py input_data_filename output_filename [plot_title [y_axis_label]]")
    exit()

# Read command line arguments
datafile = sys.argv[1]

outfile = '/dev/null/'
ylabel = ''
if len(sys.argv) >= 3:
    ylabel = sys.argv[2]
    if len(sys.argv) >=4:
        outfile = sys.argv[3]

# Read input data file
labels = []
data = []

with open(datafile) as f:
    content = f.read().splitlines()
    # First row is assumed to contain labels for each column
    labels = [l.strip() for l in content[0].split(' ')]

    # Use the number of labels to determine the number of data columns
    data = [[] for _ in range(len(labels))]
    for line in content[1:]:
        i = 0
        for num in line.split():
            data[i].append(float(num))
            i += 1

# Initialize plot
fig, ax = plt.subplots()

# Plot each data set, using first column as x-axis data
for i in range(1, len(labels)):
    ax.plot(data[0], data[i], '.', label=labels[i])

# Plot's aesthetics
ax.set_xlabel(labels[0])
ax.set_ylabel(ylabel)
ax.legend()
plt.show()
# Save figure
if outfile != '/dev/null/':
    fig.savefig(outfile)
    plt.close(fig)
