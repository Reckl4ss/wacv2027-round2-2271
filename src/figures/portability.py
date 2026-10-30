# Numerical engine; see docs/protocols.md for interpretation and provenance.
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from result_figures import (  
    CMAP,
    FLIP_PROXIES,
    SPACES,
    SPACE_FULL,
    _disp,
    _is_quarantined_flip,
    parse_portability,
)


def _text_color(rgba):
    red, green, blue = rgba[:3]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.55 else "white"


def _display_proxy(name):

    return "jacob-cov" if name == "jacov" else _disp(name)


def main():
    rows = []
    for proxy, strings in parse_portability():
        values = np.asarray([float(value) for value in strings], dtype=float)
        rows.append(
            {
                "proxy": proxy,
                "strings": strings,
                "values": values,
                "negative_count": int(np.sum(values < 0)),
                "mean_abs": float(np.mean(np.abs(values))),
                "strict": proxy in FLIP_PROXIES,
            }
        )

    
    rows.sort(key=lambda row: (row["negative_count"], -row["mean_abs"]))
    assert sum(row["negative_count"] > 0 for row in rows) == 9
    assert sum(row["strict"] for row in rows) == 4

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig = plt.figure(figsize=(7.30, 3.72))
    ax = fig.add_axes([0.150, 0.165, 0.610, 0.705])
    side = fig.add_axes([0.782, 0.165, 0.125, 0.705])
    cax = fig.add_axes([0.950, 0.165, 0.018, 0.705])

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    n_rows = len(rows)

    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        for j, (value, value_string) in enumerate(zip(row["values"], row["strings"])):
            background = CMAP(norm(value))
            strict_cell = _is_quarantined_flip(row["proxy"], SPACES[j], value_string)
            ax.add_patch(
                Rectangle(
                    (j, y),
                    1,
                    1,
                    facecolor=background,
                    edgecolor="#111111" if strict_cell else "white",
                    linewidth=2.0 if strict_cell else 1.0,
                    zorder=2 if strict_cell else 1,
                )
            )
            label = value_string + (r"$^{\star}$" if strict_cell else "")
            ax.text(
                j + 0.5,
                y + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8.7,
                fontweight="bold" if strict_cell else "normal",
                color=_text_color(background),
                zorder=3,
            )

        side.text(0.45, y + 0.5, str(row["negative_count"]), ha="center", va="center", fontsize=9.0)
        if row["negative_count"] == 0:
            symbol, color = r"$\checkmark$", "#228B22"
        elif row["strict"]:
            symbol, color = r"$\mathsf{X}$", "#D62728"
        else:
            symbol, color = r"$\sim$", "#F0A000"
        side.text(1.52, y + 0.5, symbol, ha="center", va="center", fontsize=11.0,
                  fontweight="bold", color=color)

    ax.set_xlim(0, 7)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(7) + 0.5)
    ax.set_xticklabels(SPACE_FULL, fontsize=8.2)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([_display_proxy(row["proxy"]) for row in reversed(rows)], fontsize=9.2)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    side.set_xlim(0, 2)
    side.set_ylim(0, n_rows)
    side.axis("off")
    side.text(0.45, n_rows + 0.55, "Flips", ha="center", va="bottom",
              fontsize=9.0, fontweight="bold", clip_on=False)
    side.text(1.52, n_rows + 0.55, "Stable?", ha="center", va="bottom",
              fontsize=9.0, fontweight="bold", clip_on=False)
    ax.annotate(
        "",
        xy=(-1.18, n_rows - 0.40),
        xytext=(-1.18, 0.38),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#555555"},
        annotation_clip=False,
    )
    ax.text(-1.28, n_rows - 0.55, "More\nstable", ha="right", va="top",
            fontsize=7.8, color="#228B22", clip_on=False)
    ax.text(-1.28, 0.55, "Less\nstable", ha="right", va="bottom",
            fontsize=7.8, color="#D62728", clip_on=False)

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=cax)
    colorbar.set_label(r"Spearman $\mathrm{corr}(Q,\mathrm{acc})$", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.4)

    fig.text(
        0.105,
        0.090,
        (r"$\star$ / heavy box = negative cell in a CI-separated case; "
         r"$\checkmark$ = no reversal; $\sim$ = descriptive; $\mathsf{X}$ = CI-separated."),
        ha="left",
        va="center",
        fontsize=6.4,
        color="#333333",
    )
    fig.text(
        0.105,
        0.035,
        r"9 / 13 proxies reverse sign on at least one search space; 4 are CI-separated.",
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#8B0015",
    )

    output = os.path.join(ROOT, "figures", "fig4_portability_full.pdf")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
