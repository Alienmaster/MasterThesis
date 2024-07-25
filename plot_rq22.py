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

fig, ax = plt.subplots()

# Results RQ2.2 GermEval
x = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16201]
x_s = ["32", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "16201"]
title = "RQ2.2 GermEval"
filename="RQ22GermEval"
dataset = {
    "Gemma":
    [
    [0.42, 0.41, 0.46, 0.44, 0.51, 0.56, 0.56, 0.68, 0.77, 0.79],
    [0.43, 0.39, 0.46, 0.46, 0.50, 0.58, 0.69, 0.64, 0.76, 0.81],
    [0.47, 0.39, 0.36, 0.39, 0.52, 0.57, 0.61, 0.73, 0.80, 0.80],
    ],

    "Llama 3":
    [
    [0.42, 0.39, 0.43, 0.44, 0.44, 0.54, 0.56, 0.68, 0.78, 0.80],
    [0.44, 0.42, 0.37, 0.44, 0.44, 0.60, 0.65, 0.76, 0.78, 0.82],
    [0.43, 0.39, 0.43, 0.42, 0.43, 0.55, 0.62, 0.78, 0.77, 0.80],
    ],

    "Mistral" :
    [
    [0.46, 0.49, 0.45, 0.45, 0.50, 0.50, 0.63, 0.70, 0.77, 0.79],
    [0.41, 0.42, 0.43, 0.49, 0.50, 0.52, 0.63, 0.74, 0.74, 0.79],
    [0.35, 0.40, 0.41, 0.46, 0.44, 0.58, 0.54, 0.64, 0.78, 0.81]
    ],
}
plt.hlines(0.49, 0, 512, colors=["tab:blue"],ls=':',lw=3)
plt.hlines(0.56, 0, 2048, colors=["tab:orange"],ls=':', lw=3)
plt.hlines(0.65, 0, 4096, colors=["tab:green"], ls=':', lw=3)


# # Results RQ2.2 OMP
# title = "RQ2.2 OMP"
# filename = "RQ22OMP"
# x = [32, 64, 128, 256, 512, 1024, 1799]
# dataset = {
#     "Gemma" :
#     [
#     [0.39, 0.47, 0.47, 0.49, 0.59, 0.65, 0.67],
#     [0.48, 0.50, 0.52, 0.50, 0.51, 0.63, 0.66],
#     [0.47, 0.52, 0.51, 0.49, 0.58, 0.60, 0.68],
#     ],
#     "Llama 3" :
#     [
#     [0.27, 0.42, 0.49, 0.54, 0.51, 0.62, 0.66],
#     [0.47, 0.50, 0.55, 0.55, 0.60, 0.62, 0.65],
#     [0.18, 0.47, 0.48, 0.53, 0.55, 0.65, 0.68],
#     ],
#     "Mistral" :
#     [
#     [0.32, 0.51, 0.51, 0.53, 0.54, 0.59, 0.55],
#     [0.40, 0.48, 0.50, 0.55, 0.53, 0.59, 0.56],
#     [0.53, 0.49, 0.51, 0.51, 0.54, 0.60, 0.59],
#     ],
# }
# plt.hlines(0.48, 0, 64, colors=["tab:blue"],ls=':',lw=3)
# plt.hlines(0.55, 0, 1024, colors=["tab:orange"],ls=':', lw=3)
# plt.hlines(0.57, 0, 1024, colors=["tab:green"], ls=':', lw=3)

# # Results RQ2.2 Schmidt
# title = "RQ2.2 Schmidt"
# filename = "RQ22Schmidt"
# x = [32, 64, 128, 256, 512, 1024, 1428]
# dataset = {
#     "Gemma" :
#     [
#     [0.35, 0.29, 0.41, 0.46, 0.57, 0.62, 0.71],
#     [0.38, 0.33, 0.40, 0.41, 0.54, 0.67, 0.76],
#     [0.42, 0.38, 0.41, 0.35, 0.53, 0.63, 0.77]
#     ],

#     "Llama 3" :
#     [
#     [0.34, 0.32, 0.41, 0.38, 0.59, 0.69, 0.75],
#     [0.34, 0.33, 0.38, 0.41, 0.50, 0.66, 0.76],
#     [0.26, 0.31, 0.45, 0.38, 0.48, 0.64, 0.73]
#     ],

#     "Mistral" :
#     [
#     [0.34, 0.38, 0.42, 0.42, 0.54, 0.63, 0.64],
#     [0.33, 0.38, 0.40, 0.43, 0.55, 0.54, 0.68],
#     [0.35, 0.33, 0.37, 0.46, 0.53, 0.68, 0.64]
#     ],
# }
# plt.hlines(0.60, 0, 1024, colors=["tab:blue"],ls=':',lw=3)
# plt.hlines(0.61, 0, 1024, colors=["tab:orange"],ls=':', lw=3)
# plt.hlines(0.68, 0, 1428, colors=["tab:green"], ls=':', lw=3)

fontsize = 13

# Create plot path
Path(plot_path).mkdir(parents=True, exist_ok=True)

for model in dataset:
    r0 = dataset[model][0]
    r1 = dataset[model][1]
    r2 = dataset[model][2]
    high = np.max(np.vstack((r0, r1, r2)), axis=0)
    low = np.min(np.vstack((r0, r1, r2)), axis=0)
    mean = np.mean(np.vstack((r0, r1, r2)), axis=0)
    print(f"Mean value for {model} with each size: {mean}")
    ax.plot(x, mean, label=model,linewidth=5.0)
    # ax.fill_between(x, low, high, alpha=0.5)
    ax.legend(loc=4, prop={'size': 15})
ax.set
ax.set_ylabel("Accuracy", fontsize=fontsize)
ax.set_xlabel("Training size (log scale)", fontsize=fontsize)
# ax.set_xticklabels(x, fontsize=10)
# plt.title(title)
plt.xscale("log")
plt.xticks(x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
plt.xlim((x[0],x[-1]))
ax.tick_params(axis='y', labelsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)

plt.savefig(f"{plot_path}{filename}_wofill.pdf", bbox_inches='tight')