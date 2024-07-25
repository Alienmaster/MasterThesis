import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib as mpl
from pathlib import Path

plot_path = "plots/"
# These settings are only used to use latex for text generation.
rc_fonts = {
    "font.family": "serif",
    "font.size": 15.8,
    # 'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage[libertine]{newtxmath}
        \usepackage{libertine}
        """,
}
# mpl.rcParams.update(rc_fonts)

fontsize = 15

fig, ax = plt.subplots()

# Results RQ2.3 SetFit
x = [32, 64, 128, 256, 512, 1024, 2048]
title = "RQ2.3 SetFit"
filename = "RQ23"
dataset = { 
    "GermEval" :
    [
    [0.66, 0.70, 0.72, 0.71, 0.72, 0.74, 0.72],#, 0.76],
    [0.66, 0.70, 0.70, 0.71, 0.72, 0.75, 0.75],
    [0.68, 0.67, 0.72, 0.73, 0.74, 0.72, 0.74],
    ],
    "OMP" :
    [
    [0.57, 0.51, 0.63, 0.64, 0.68, 0.68],
    [0.55, 0.58, 0.66, 0.65, 0.68, 0.69],
    [0.56, 0.58, 0.57, 0.64, 0.67, 0.68],
    ],
    "Schmidt" :
    [
    [0.60, 0.68, 0.78, 0.80, 0.80, 0.79],
    [0.64, 0.73, 0.79, 0.78, 0.81, 0.74],
    [0.63, 0.72, 0.74, 0.80, 0.77, 0.79],
    ],
}

# Plot for GermEval due to six instead of five results.
xi = list(range(len(x)))
r0 = dataset["GermEval"][0]
r1 = dataset["GermEval"][1]
r2 = dataset["GermEval"][2]
high = np.max(np.vstack((r0, r1, r2)), axis=0)
low = np.min(np.vstack((r0, r1, r2)), axis=0)
mean = np.mean(np.vstack((r0, r1, r2)), axis=0)
print(f"Mean value for GermEval with each size: {mean}")
ax.plot(x, mean, label="GermEval",linewidth=5.0)
ax.fill_between(x, low, high, alpha=0.5)
# ax.legend(loc=4)
plt.xlim((x[0],x[-1]))


# Plot of OMP and Schmidt
x = [32, 64, 128, 256, 512, 1024]
xi = list(range(len(x)))
for model in dataset:
    if model == "GermEval":
        continue
    r0 = dataset[model][0]
    r1 = dataset[model][1]
    r2 = dataset[model][2]
    high = np.max(np.vstack((r0, r1, r2)), axis=0)
    low = np.min(np.vstack((r0, r1, r2)), axis=0)
    mean = np.mean(np.vstack((r0, r1, r2)), axis=0)
    print(f"Mean value for {model} with each size: {mean}")
    ax.plot(x, mean, label=model,linewidth=5.0)
    ax.fill_between(x, low, high, alpha=0.5)
    ax.legend(loc=4, prop={'size': fontsize})



ax.set_ylabel("Accuracy", fontsize=fontsize)
ax.set_xlabel("Training size", fontsize=fontsize)
# plt.title(title)
plt.xscale("log")
plt.xticks(x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
# Set limits

ax.tick_params(axis='y', labelsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)
# Save Plot
Path(plot_path).mkdir(parents=True, exist_ok=True)
plt.savefig(f"{plot_path}{filename}.pdf", bbox_inches='tight')