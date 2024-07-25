from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plot_path = "plots/"
fontsize = 14
# Gemma
experiments = ("RQ1\n0-shot", "GermEval\n3-shot", "OMP\n3-shot", "Schmidt\n3-shot", "All\n9-shot")
class_accuracy = {
    'Negative': (0.95, 0.95, 0.95, 0.97, 0.97),
    'Positive': (0.88, 0.63, 0.57, 0.57, 0.64),
    'Neutral':  (0.20, 0.40, 0.41, 0.44, 0.40),
}

# These settings are only used to use latex for text generation.
rc_fonts = {
    "font.family": "serif",
    "font.size": 15,
    # 'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        """,
}
# mpl.rcParams.update(rc_fonts)

x = np.arange(len(experiments))  # the label locations
width = 0.28  # the width of the bars
multiplier = 0
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(layout='constrained')
custom_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][6:]
for i, (attribute, measurement) in enumerate(class_accuracy.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=custom_colors[i % len(custom_colors)])
    multiplier += 1
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0 + 0.04,  # move label to the right
            height-0.005,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=fontsize
        )

# Labels
ax.set_ylabel('Accuracy', fontsize=fontsize)
# ax.set_title('RQ2.1 Schmidt Gemma')
ax.set_xticks(x + width, experiments, fontsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
# Save plot
Path(plot_path).mkdir(parents=True, exist_ok=True)
plt.legend(framealpha=1,loc='lower right', ncols=1, prop={'size': fontsize})
plt.savefig(f"{plot_path}RQ21SchmidtSchmidtGemmaPlot.pdf", bbox_inches='tight')
plt.show()