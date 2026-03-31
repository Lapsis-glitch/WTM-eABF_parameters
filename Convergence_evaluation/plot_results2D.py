import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse
import seaborn as sns
import matplotlib
import matplotlib.font_manager as font_manager

def load_matplotlib_local_fonts():

    # Load a font from TTF file, 
    # relative to this Python module
    # https://stackoverflow.com/a/69016300/315168
    #font_path = os.path.join(os.path.dirname(__file__), '/home/lia/gchen/miniconda3/envs/nn4/fonts/arial.ttf')
    font_path = '/home/lia/gchen/miniconda3/envs/nn/fonts/arial.ttf'
    assert os.path.exists(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rcParams.update({
        'font.size': 12,
        'font.sans-serif': prop.get_name(),
    })
try:
    load_matplotlib_local_fonts()
except:
    pass

labels = {"MTDheight": "hillWeight",
          "MTDnewhill": "newHillFrequency",
          "MTDtemp": "biasTemperature",
          "MTDwidth": "hillWidth",
          "colvarWidth": "colvarWidth",
          "extDamp": "extendedLangevinDamping",
          "extFluc": "extendedFluctuation",
          "extTime": "extendedTimeConstant",
          "fullSamp": "fullSamples",
          }

"""Plot mean convergence with standard deviation error bars."""
parser = argparse.ArgumentParser(
    description="Plot mean convergence with std deviation from results.dat."
)
parser.add_argument(
    "file",
    default="results.dat",
    help="file containing results"
)
parser.add_argument(
    "--divisor",
    type=float,
    default=20.0,
    help="Divisor for Y values (default: 20)"
)
args = parser.parse_args()

# Locate results file
results_file = args.file
divisor = args.divisor


# Lists to store extracted data
param1_vals = []
param2_vals = []
mean_vals = []
std_vals = []

# Regex to extract parameters from base_name
# Allows decimal numbers for values
param_pattern = re.compile(r"([A-Za-z0-9]+)_([0-9.]+)_([A-Za-z0-9]+)_([0-9.]+)")

with open(results_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue  # skip invalid lines
        base_name, mean, std, min_val, max_val, n = parts
        match = param_pattern.match(base_name)
        if not match:
            continue  # skip if pattern doesn't match
        # Extract parameters
        param1_name, param1_value, param2_name, param2_value = match.groups()
        try:
            param1_vals.append(float(param1_value))
            param2_vals.append(float(param2_value))
            mean_vals.append(float(mean))
            std_vals.append(float(std))
        except ValueError:
            continue  # skip if conversion fails

# Create a grid for plotting
param1_unique = np.unique(param1_vals)
param2_unique = np.unique(param2_vals)
param1_grid, param2_grid = np.meshgrid(param1_unique, param2_unique)

# Initialize grids for mean and std
mean_grid = np.full(param1_grid.shape, np.nan) 
std_grid = np.full(param1_grid.shape, np.nan)

# Fill grids
for x, y, m, s in zip(param1_vals, param2_vals, mean_vals, std_vals):
    i = np.where(param2_unique == y)[0][0]
    j = np.where(param1_unique == x)[0][0]
    mean_grid[i, j] = m / divisor
    std_grid[i, j] = s / divisor

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Mean convergence time
#im0 = axes[0].imshow(mean_grid, origin='lower',
#                     extent=[param1_unique.min(), param1_unique.max(),
#                             param2_unique.min(), param2_unique.max()],
#                     aspect='auto', cmap='viridis')
im0 = axes[0].contourf(param1_grid, param2_grid, mean_grid, levels=40, cmap='viridis')
axes[0].set_xlabel(labels[param1_name])
axes[0].set_ylabel(labels[param2_name])
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label("Mean Convergence Time (ns)")

# Standard deviation
#im1 = axes[1].imshow(std_grid, origin='lower',
#                     extent=[param1_unique.min(), param1_unique.max(),
#                             param2_unique.min(), param2_unique.max()],
#                     aspect='auto', cmap='magma')
im1 = axes[1].contourf(param1_grid, param2_grid, std_grid, levels=40, cmap='magma')
axes[1].set_xlabel(labels[param1_name])
axes[1].set_ylabel(labels[param2_name])
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label("Standard Deviation (ns)")

plt.tight_layout()
plt.savefig("convergence_2D.png", dpi=300)