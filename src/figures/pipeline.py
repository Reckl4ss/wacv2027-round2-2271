# Numerical engine; see docs/protocols.md for interpretation and provenance.
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]


def box(ax, x, y, w, h, text, face, edge, size=8.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, start, end, color="#222222"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.25,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 2.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Analysis pipeline: capacity, trainability, and transport are evaluated separately",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
    )

    box(ax, 0.015, 0.22, 0.17, 0.58, "29-feature\naligned zoo\n$\\{Q_j\\}$ at init", "#E8F1FA", "#1976B9", size=8.0)
    box(ax, 0.275, 0.65, 0.245, 0.17, "PCA in rank space", "#F2F2F2", "#222222", size=8.0)
    box(
        ax,
        0.645,
        0.62,
        0.335,
        0.23,
        "Capacity-associated factor\n$|\\rho(\\mathrm{PC1},\\mathrm{erank})|=.94$--$.99$",
        "#ECF8F1",
        "#0A9D72",
        size=8.0,
    )
    box(ax, 0.275, 0.39, 0.245, 0.19, "Capacity-adjusted\nproxy partials", "#F2F2F2", "#222222", size=8.0)
    box(ax, 0.645, 0.38, 0.335, 0.21, "Association with\nearly-training $T$", "#FFF2E7", "#D95F02", size=8.0)
    box(ax, 0.275, 0.14, 0.245, 0.19, "Raw proxy--accuracy\ncorrelations", "#F2F2F2", "#222222", size=8.0)
    box(ax, 0.645, 0.10, 0.335, 0.27, "Cross-space\nsign reversal\n(diagnostic)", "#FDEAF3", "#D46A9E", size=8.0)

    arrow(ax, (0.185, 0.69), (0.275, 0.735))
    arrow(ax, (0.520, 0.735), (0.645, 0.735))
    arrow(ax, (0.185, 0.50), (0.275, 0.485))
    arrow(ax, (0.520, 0.485), (0.645, 0.485))
    arrow(ax, (0.185, 0.34), (0.275, 0.235))
    arrow(ax, (0.520, 0.235), (0.645, 0.235), color="#C95E94")

    output = ROOT / "figures" / "fig2_overview.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
