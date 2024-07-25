from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

plot_path = "plots/"

# These settings are only used to use latex for text generation.
rc_fonts = {
    # "font.family": "serif",
    "font.size": 16.5,
    # "font.size": 15,
    # 'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        """,
}
# mpl.rcParams.update(rc_fonts)

matrices = [
# RQ1
{
    "title":"GerVADER",
    "filename":"RQ1GermEvalGerVADER",
    "values" : np.array([
        [324, 283, 173],
        [11, 84, 10],
        [196, 637, 377]
    ]),
    "accuracy" : np.array([
        [0.42, 0.36, 0.22],
        [0.10, 0.80, 0.10],
        [0.16, 0.52, 0.31]
    ])
},
{
    "title":"Guhr",
    "filename":"RQ1GermEvalGuhr",
    "values" : np.array([
        [613, 7, 160],
        [19, 14, 72],
        [290, 24, 896]
    ]),
    "accuracy" : np.array([
        [0.79, 0.01, 0.20],
        [0.18, 0.13, 0.69],
        [0.24, 0.02, 0.74]
    ])
},
{
    "title":"Lxyuan",
    "filename":"RQ1GermEvalLxyuan",
    "values": np.array([
        [613, 89, 78],
        [11, 82, 12],
        [495, 378, 337]
    ]),
    "accuracy" : np.array([
    [0.79, 0.11, 0.10],
    [0.10, 0.78, 0.11],
    [0.41, 0.31, 0.28]
    ])
},
{
    "title":"Gemma",
    "filename":"RQ1GermEvalGemma",
    "values": np.array([
        [613, 89, 78],
        [11, 82, 12],
        [495, 378, 337]
    ]),
    "accuracy" : np.array([
    [0.79, 0.11, 0.10],
    [0.10, 0.78, 0.11],
    [0.41, 0.31, 0.28]
    ])
},
{
    "title":"Llama 3",
    "filename":"RQ1GermEvalLlama3",
    "values": np.array([
        [615, 39, 126],
        [9, 65, 31],
        [491, 228, 490]
    ]),
    "accuracy" : np.array([
        [0.79, 0.05, 0.16],
        [0.09, 0.62, 0.30],
        [0.41, 0.19, 0.41]
    ])
},
{
    "title":"Mistral",
    "filename":"RQ1GermEvalMistral",
    "values": np.array([
        [535, 29, 212],
        [7, 57, 40],
        [292, 140, 769]
    ]),
    "accuracy" : np.array([
        [0.69, 0.04, 0.27],
        [0.07, 0.55, 0.38],
        [0.24, 0.12, 0.64]
    ])
},
# {
#     "title":"RQ1 OMP GerVADER",
#     "filename":"RQ1OMPGerVADER",
#     "values": np.array([
#         [603, 735, 352],
#         [2, 35, 6],
#         [500, 845, 520]
#     ]),
#     "accuracy": np.array([
#         [0.36, 0.43, 0.21],
#         [0.05, 0.81, 0.14],
#         [0.27, 0.45, 0.28]
#     ])
# },
# {
#     "title":"RQ1 OMP Gemma",
#     "filename":"RQ1OMPGemma",
#     "values": np.array([
#         [1329, 210, 152],
#         [4, 37, 2],
#         [1178, 309, 376]
#     ]),
#     "accuracy": np.array([
#         [0.79, 0.12, 0.09],
#         [0.09, 0.86, 0.5],
#         [0.63, 0.17, 0.20]
#     ])
# },
# {
#     "title":"RQ1 OMP Llama 2",
#     "filename":"RQ1OMPLlama2",
#     "values" : np.array([
#         [1117, 337, 211],
#         [2, 37, 4],
#         [824, 551, 466]
#     ]),
#     "accuracy" : np.array([
#         [0.67, 0.20, 0.13],
#         [0.05, 0.86, 0.09],
#         [0.44, 0.30, 0.25]
#     ])
# },
# {
#     "title":"RQ1 OMP Llama 3",
#     "filename":"RQ1OMPLlama38BI",
#     "values" : np.array([
#         [1388, 77, 215],
#         [4, 32, 7],
#         [1160, 137, 559]
#     ]),
#     "accuracy" : np.array([
#         [0.83, 0.05, 0.13],
#         [0.09, 0.74, 0.16],
#         [0.63, 0.07, 0.30]
#     ])
# },
# RQ1 Schmidt
# {
#     "title":"GerVADER",
#     "filename":"RQ1SchmidtGerVADER",
#     "values" : np.array([
#     [53, 44, 10],
#     [5, 88, 3],
#     [24, 99, 29]
#     ]),
#     "accuracy" : np.array([
#     [0.50, 0.41, 0.09],
#     [0.05, 0.92, 0.03],
#     [0.16, 0.65, 0.19]
#     ])
# },
# {
#     "title":"Guhr",
#     "filename":"RQ1SchmidtGuhr",
#     "values" : np.array([
#     [34, 1, 73],
#     [6, 10, 81],
#     [15, 2, 135]
#     ]),
#     "accuracy" : np.array([
#     [0.31, 0.01, 0.68],
#     [0.06, 0.10, 0.84],
#     [0.10, 0.01, 0.89]
#     ])
# },
# {
#     "title":"Lxyuan",
#     "filename":"RQ1SchmidtLxyuan",
#     "values" : np.array([
#     [103, 5, 0],
#     [15, 82, 0],
#     [90, 60, 2]
#     ]),
#     "accuracy" : np.array([
#     [0.95, 0.04, 0.0],
#     [0.15, 0.85, 0.0],
#     [0.53, 0.39, 0.01]
#     ])
# },
# {
#     "title":"Gemma",
#     "filename":"RQ1SchmidtGemma",
#     "values" : np.array([
#     [97, 10, 1],
#     [18, 87, 2],
#     [37, 85, 30]
#     ]),
#     "accuracy" : np.array([
#     [0.90, 0.09, 0.01],
#     [0.17, 0.81, 0.02],
#     [0.24, 0.56, 0.20]
#     ])
# },
# {
#     "title":"Llama 2",
#     "filename":"RQ1SchmidtLlama2",
#     "values" : np.array([
#     [94, 10, 4],
#     [2, 94, 0],
#     [29, 109, 14]
#     ]),
#     "accuracy" : np.array([
#     [0.87, 0.09, 0.04],
#     [0.02, 0.98, 0.00],
#     [0.19, 0.72, 0.09]
#     ])
# },
# {
#     "title":"Llama 3",
#     "filename":"RQ1SchmidtLlama3",
#     "values" : np.array([
#     [103, 3, 2],
#     [7, 85, 5],
#     [43, 78, 31]
#     ]),
#     "accuracy" : np.array([
#     [0.95, 0.03, 0.02],
#     [0.07, 0.88, 0.05],
#     [0.28, 0.51, 0.20]
#     ])
# },
# {
#     "title":"Mistral",
#     "filename":"RQ1SchmidtMistral7BI",
#     "values" : np.array([
#     [91, 3, 14],
#     [3, 83, 10],
#     [21, 62, 66]
#     ]),
#     "accuracy" : np.array([
#     [0.84, 0.028, 0.13],
#     [0.03, 0.86, 0.10],
#     [0.14, 0.42, 0.44]
#     ])
# },
# {
#     "title":"Mistral 8x7B",
#     "filename":"RQ1SchmidtMistral8xBI",
#     "values" : np.array([
#     [103, 5, 0],
#     [1, 89, 7],
#     [39, 63, 50]
#     ]),
#     "accuracy" : np.array([
#     [0.95, 0.05, 0.0],
#     [0.01, 0.92, 0.07],
#     [0.26, 0.41, 0.33]
#     ])
# },
# {
#     "title":"RQ1 Wikipedia Lxyuan",
#     "filename":"RQ1WikipediaLxyuan",
#     "values" : np.array([
#     [0, 0, 0],
#     [0, 0, 0],
#     [3601, 6132, 267]
#     ]),
#     "accuracy" : np.array([
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.36, 0.61, 0.027]
#     ])
# },

# # RQ2.1
# {
#     "title":"RQ2.1 GermEval Gemma",
#     "filename":"RQ21GermEvalGemma",
#     "values" : np.array([
#     [682, 11, 87],
#     [23, 43, 39],
#     [673, 129, 408]
#     ]),
#     "accuracy" : np.array([
#     [0.87, 0.01, 0.11],
#     [0.22, 0.41, 0.37],
#     [0.56, 0.10, 0.34]
#     ])
# },
# {
#     "title":"RQ2.1 GermEval2017 Llama 3",
#     "filename":"RQ21GermEvalLlama3",
#     "values" : np.array([
#     [415, 50, 279],
#     [2, 57, 46],
#     [317, 223, 669]
#     ]),
#     "accuracy" : np.array([
#     [0.56, 0.07, 0.38],
#     [0.02, 0.54, 0.44],
#     [0.26, 0.18, 0.55]
#     ])
# },
# {
#     "title":"RQ2.1 OMP Gemma",
#     "filename":"RQ21OMPGemma",
#     "values" : np.array([
#     [1350, 27, 301],
#     [6, 25, 12],
#     [1011, 52, 788]
#     ]),
#     "accuracy" : np.array([
#     [0.80, 0.02, 0.18],
#     [0.14, 0.58, 0.28],
#     [0.55, 0.03, 0.43]
#     ])
# },
# {
#     "title":"RQ2.1 Schmidt:GermEval Gemma",
#     "filename":"RQ21SchmidtSchmidtGemma",
#     "values" : np.array([
#     [103, 5, 0],
#     [9, 61, 27],
#     [48, 43, 61]
#     ]),
#     "accuracy" : np.array([
#     [0.95, 0.05, 0.00],
#     [0.09, 0.63, 0.28],
#     [0.32, 0.28, 0.40]
#     ])
# },
# {
#     "title":"RQ2.1 Schmidt:OMP Gemma",
#     "filename":"RQ21SchmidtOMPGemma",
#     "values" : np.array([
#     [103, 3, 2],
#     [11, 55, 31],
#     [53, 37, 62]
#     ]),
#     "accuracy" : np.array([
#     [0.95, 0.03, 0.02],
#     [0.11, 0.57, 0.32],
#     [0.35, 0.24, 0.41]
#     ])
# },
# {
#     "title":"RQ2.1 Schmidt Gemma",
#     "filename":"RQ21SchmidtGemma",
#     "values" : np.array([
#     [105, 2, 1],
#     [9, 55, 32],
#     [51, 34, 67]
#     ]),
#     "accuracy" : np.array([
#     [0.97, 0.02, 0.01],
#     [0.09, 0.57, 0.33],
#     [0.33, 0.22, 0.44]
#     ])
# },
# {
#     "title":"RQ2.1 Schmidt:all Gemma",
#     "filename":"RQ21SchmidtAllGemma",
#     "values" : np.array([
#     [105, 1, 2],
#     [12, 62, 23],
#     [53, 38, 61]
#     ]),
#     "accuracy" : np.array([
#     [0.97, 0.01, 0.02],
#     [0.12, 0.64, 0.24],
#     [0.35, 0.25, 0.40]
#     ])
# },
# {
#     "title":"RQ2.1 Schmidt Llama 3",
#     "filename":"RQ21SchmidtLlama3",
#     "values" : np.array([
#     [96, 4, 8],
#     [3, 84, 10],
#     [27, 56, 69]
#     ]),
#     "accuracy" : np.array([
#     [0.89, 0.04, 0.07],
#     [0.03, 0.87, 0.10],
#     [0.18, 0.37, 0.45]
#     ])
# }
]

# Create path for plots
Path(plot_path).mkdir(parents=True, exist_ok=True)
fig, axs = plt.subplots(2, 3, figsize=(15, 12))  # Adjust figsize as needed
axs = axs.flatten()  # Flatten to easily iterate

for idx, results in enumerate(matrices):
    accuracy = results["accuracy"]
    values = results["values"]
    title = results["title"]
    
    ax = axs[idx]
    # ax.set_facecolor(facecolor)
    # Normalise the color values between 0 and 1
    norm = plt.Normalize(0, 1)
    colors = plt.cm.viridis(norm(accuracy))

    # Plot the values and color the squares
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, values[i, j], ha='center', va='center', 
                    color='white' if colors[i, j][:3].mean() < 0.5 else 'black', fontsize=35)
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=colors[i, j]))

    # Einstellen der Achsen
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    # labels = ["Negative", "Positive", "Neutral"]
    labels = ["-", "+", "0"]
    ax.set_xticklabels(labels, fontsize=25)
    ax.tick_params(bottom=False, left=False)
    ax.set_yticklabels(labels, rotation=0, va="center", ha="right", fontsize=25)

    # Achsentitel hinzufügen
    # ax.set_xlabel('Predictions')
    # ax.set_ylabel('Labels')

    # Sets aspect ratio for quadratic squares
    ax.set_aspect('equal')

    ax.invert_yaxis()

    # Adds farbscale
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
    # cbar.set_label('Distribution')
    # cbar.set_ticks([0, 0.5, 1.0])  # Optional: Setzen der Tick-Marker

    ax.set_title(title, fontsize=25)
for ax in axs.flat:
    ax.label_outer()
plt.subplots_adjust(wspace=-0.0, hspace=-0.4)
plt.tight_layout()
plt.savefig("rq1_germeval_cm_comp.pdf", bbox_inches='tight')


# for plot in matrices:
#     results = plot
#     accuracy = results["accuracy"]
#     values = results["values"]
#     title = results["title"]
#     filename = results["filename"]
    
#     # Set the size
#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Normalise the colorvalues between 0 and 1
#     norm = plt.Normalize(0, 1)
#     colors = plt.cm.viridis(norm(accuracy))

#     # Plot the values and color the squares
#     for i in range(values.shape[0]):
#         for j in range(values.shape[1]):
#             ax.text(j, i, values[i, j], ha='center', va='center', 
#                     color='white' if colors[i, j][:3].mean() < 0.5 else 'black', fontsize=16)
#             ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=colors[i, j]))

#     # Einstellen der Achsen
#     ax.set_xlim(-0.5, 2.5)
#     ax.set_ylim(-0.5, 2.5)
#     ax.set_xticks([0, 1, 2])
#     ax.set_yticks([0, 1, 2])

#     labels = ["Negative", "Positive", "Neutral"]
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels, rotation=90, va="center", ha="right")

#     # Achsentitel hinzufügen
#     ax.set_xlabel('Predictions')
#     ax.set_ylabel('Labels')

#     # Sets aspect ratio for quadratic squares
#     ax.set_aspect('equal')

#     ax.invert_yaxis()

#     # Adds farbscale
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
#     cbar.set_label('Distribution')
#     cbar.set_ticks([0, 0.5, 1.0])  # Optional: Setzen der Tick-Marker

#     ax.set_title(title)
#     plt.tight_layout()
#     plt.savefig(f"{plot_path}{filename}.pdf", bbox_inches='tight')