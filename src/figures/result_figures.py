# Numerical engine; see docs/protocols.md for interpretation and provenance.
import io
import os
import re

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle




HERE = os.path.dirname(os.path.abspath(__file__))          
ROOT = os.path.dirname(HERE)                               
RES = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

ZOO_PATH = os.path.join(RES, "zoo_analysis_out.txt")
XS_PATH = os.path.join(RES, "crossspace_out.txt")
zoo = io.open(ZOO_PATH, encoding="utf-8").read()
xs = io.open(XS_PATH, encoding="utf-8").read()




plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,   
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})







CMAP = plt.get_cmap("RdBu")          
GRID = "#D9D9D9"
MISS_GRAY = "#BDBDBD"
FLIP_EDGE = "#111111"
NTK_EDGE = "#000000"


def _despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)


def _text_color(bg_rgba):

    r, g, b = bg_rgba[0], bg_rgba[1], bg_rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.55 else "white"


def _disp(name):


    if name.startswith("Ralpha_"):
        alpha = name.split("_", 1)[1]
        if alpha.endswith(".0"):
            alpha = alpha[:-2]
        return rf"$R_{{\alpha={alpha}}}$"
    return name.replace("_", "-")





def block(text, start_marker, end_marker=None):
    i = text.index(start_marker)
    j = text.index(end_marker, i + len(start_marker)) if end_marker else len(text)
    return text[i:j]




SPACES = ["NB101", "NB201-c10", "NB201-c100", "NB201-IN16",
          "NB301", "TNAS-obj", "TNAS-sc"]
SPACE_FULL = ["NB101\nC10", "NB201\nC10", "NB201\nC100", "NB201\nI16",
              "NB301\nC10", "TransNAS\nobject", "TransNAS\nscene"]

_PORT_VAL_RE = re.compile(r"[+-]\d\.\d+|nan")


def parse_portability():

    pm = block(xs, "TABLE 1: PORTABILITY MATRIX", "PORTABILITY SUMMARY")
    mat = []
    for line in pm.splitlines():
        if not line or line.startswith(("PROXY", "---", "TABLE")):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, rest = parts
        vals = _PORT_VAL_RE.findall(rest)
        if len(vals) == 7:
            mat.append((name, vals))
    assert len(mat) == 13, "expected 13 portability rows, got %d" % len(mat)
    return mat





RHO_RE = re.compile(r"^\s+(\S+)\s+[+-]?\d.*?\(rho=([+-]?\d+\.\d+)\)")
ROW_RE = re.compile(
    r"^\s+(\S+)\s+([+-]\d\.\d+)\s+([+-]\d\.\d+)\[([+-]\d\.\d+),([+-]\d\.\d+)\]"
    r"\s+([+-]\d\.\d+)\[[^]]*\]\s+([+-]\d\.\d+)")


def parse_space(tag):





    r3 = block(zoo,
               "R3 FULL-ZOO TRAINABILITY-RESIDUAL ACCESS + CEILING -- %s" % tag,
               "CEILING rankR2")
    rows = {}
    order = []
    for line in r3.splitlines():
        m = ROW_RE.match(line)
        if m:
            name = m.group(1)
            rows[name] = dict(pT=m.group(3), lo=m.group(4), hi=m.group(5))
            order.append(name)
    return order, rows




_DROW_RE = re.compile(r"^(\S+)\s+(.+?)\s+(\d\.\d+)\s+([+-]\d\.\d+)\s+(\d+)\s*$")


def parse_decomp():

    dc = block(xs, "DECOMPOSITION (NB201", "TABLE 4")
    out = []
    for line in dc.splitlines():
        m = _DROW_RE.match(line)
        if m:
            out.append(m.groups())
    assert len(out) == 12, "expected 12 decomposition rows, got %d" % len(out)
    return out











FLIP_PROXIES = {"jacov", "grasp", "zen", "plain"}


def _is_quarantined_flip(proxy, space, val_str):


    if proxy not in FLIP_PROXIES:
        return False
    if not val_str.startswith("-"):
        return False
    if proxy == "jacov":
        return space == "NB101"                   
    if proxy == "grasp":
        return space == "TNAS-sc"                 
    if proxy in ("zen", "plain"):
        return space in ("NB201-c10", "NB201-c100", "NB201-IN16")
    return False


def make_fig4():
    mat = parse_portability()
    
    
    keep = list(range(len(SPACES)))
    spaces = [SPACES[j] for j in keep]
    space_full = [SPACE_FULL[j] for j in keep]
    proxies = [name for name, _ in mat]
    n_rows = len(proxies)        
    n_cols = len(spaces)         

    
    Z = np.full((n_rows, n_cols), np.nan)
    valstr = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for i, (_, vals) in enumerate(mat):
        for jj, j in enumerate(keep):
            valstr[i][jj] = vals[j]
            if vals[j] != "nan":
                Z[i, jj] = float(vals[j])

    vmax = float(np.nanmax(np.abs(Z)))           
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_w, fig_h = 7.2, 4.2                       
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    
    for i in range(n_rows):
        for j in range(n_cols):
            v = Z[i, j]
            x0, y0 = j, n_rows - 1 - i            
            if np.isnan(v):
                rect = Rectangle((x0, y0), 1, 1, facecolor=MISS_GRAY,
                                 edgecolor="white", linewidth=1.0,
                                 hatch="////", alpha=0.55, zorder=1)
                ax.add_patch(rect)
                ax.text(x0 + 0.5, y0 + 0.5, "--", ha="center", va="center",
                        fontsize=7.5, color="#555555", zorder=3)
                continue
            bg = CMAP(norm(v))                    
            flip = _is_quarantined_flip(proxies[i], spaces[j], valstr[i][j])
            rect = Rectangle((x0, y0), 1, 1, facecolor=bg,
                             edgecolor=(FLIP_EDGE if flip else "white"),
                             linewidth=(2.2 if flip else 1.0), zorder=(4 if flip else 1))
            ax.add_patch(rect)
            txt = valstr[i][j]
            if flip:
                txt = txt + r"$^{\star}$"
            ax.text(x0 + 0.5, y0 + 0.5, txt, ha="center", va="center",
                    fontsize=8.2, color=_text_color(bg),
                    fontweight=("bold" if flip else "normal"), zorder=5)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    
    
    
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(space_full, fontsize=8.0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([_disp(p) for p in reversed(proxies)], fontsize=8.5)
    ax.tick_params(length=0)
    _despine_all(ax)

    
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02, aspect=28)
    cbar.set_label(r"Spearman $\mathrm{corr}(Q,\mathrm{acc})$"
                   "\n(blue $=$ positive, red $=$ sign-flip)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)

    
    fig.text(0.012, 0.015,
             r"$^{\star}$ = one of the 4 CI-separated load-bearing flips "
             r"(jacob-cov, grasp, zen, plain).",
             ha="left", va="bottom", fontsize=6.0, color="#444444")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.085)
    out = os.path.join(FIGDIR, "fig4_portability.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out







def make_fig5():
    
    order_t, row_t = parse_space("FULLZOO-tss")
    order_s, row_s = parse_space("FULLZOO-sss")
    assert len(order_t) == 29 and len(order_s) == 29, (len(order_t), len(order_s))
    proxies = order_t                                  

    n = len(proxies)
    Pa = np.full((n, 2), np.nan)
    Pstr = [["", ""] for _ in range(n)]
    Pci = [[None, None] for _ in range(n)]             
    for i, name in enumerate(proxies):
        for c, rows in ((0, row_t), (1, row_s)):
            d = rows.get(name)
            if d is None:
                continue
            Pa[i, c] = float(d["pT"])
            Pstr[i][c] = d["pT"]
            Pci[i][c] = (float(d["lo"]), float(d["hi"]))

    vmax_a = float(np.nanmax(np.abs(Pa)))
    norm_a = TwoSlopeNorm(vmin=-vmax_a, vcenter=0.0, vmax=vmax_a)

    def ci_excludes_zero(ci):
        return ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)

    
    fig, axA = plt.subplots(figsize=(3.4, 5.2))   

    
    
    
    
    for i in range(n):
        for c in range(2):
            v = Pa[i, c]
            x0, y0 = c, n - 1 - i
            if np.isnan(v):  
                rect = Rectangle((x0, y0), 1, 1, facecolor=MISS_GRAY,
                                 edgecolor="white", linewidth=0.8,
                                 hatch="////", alpha=0.5, zorder=1)
                axA.add_patch(rect)
                axA.text(x0 + 0.5, y0 + 0.5, "--", ha="center", va="center",
                         fontsize=5.6, color="#555555", zorder=3)
                continue
            bg = CMAP(norm_a(v))
            outline = (proxies[i] == "ntk" and ci_excludes_zero(Pci[i][c]) and v > 0)
            rect = Rectangle((x0, y0), 1, 1, facecolor=bg,
                             edgecolor=(NTK_EDGE if outline else "white"),
                             linewidth=(2.0 if outline else 0.8),
                             zorder=(4 if outline else 1))
            axA.add_patch(rect)
            axA.text(x0 + 0.5, y0 + 0.5, Pstr[i][c], ha="center", va="center",
                     fontsize=6.4, color=_text_color(bg),
                     fontweight=("bold" if outline else "normal"),
                     zorder=5)

    axA.set_xlim(0, 2)
    axA.set_ylim(0, n)
    axA.set_aspect("auto")
    axA.set_xticks([0.5, 1.5])
    axA.set_xticklabels(["TSS\n(topology)", "SSS\n(size)"], fontsize=7.0)
    axA.xaxis.set_ticks_position("top")
    axA.xaxis.set_label_position("top")

    ylabels = []
    for name in reversed(proxies):
        lab = _disp(name)
        if name == "Q_zico_oracle":
            lab = lab + r" $^{\mathrm{L}}$"
        ylabels.append(lab)
    axA.set_yticks(np.arange(n) + 0.5)
    axA.set_yticklabels(ylabels, fontsize=6.4)
    axA.tick_params(length=0)
    _despine_all(axA)

    
    sm_a = mpl.cm.ScalarMappable(norm=norm_a, cmap=CMAP)
    sm_a.set_array([])
    cbA = fig.colorbar(sm_a, ax=axA, fraction=0.06, pad=0.04, aspect=32)
    cbA.set_label(r"leakage-clean $\mathrm{partial}(Q,T\mid\mathrm{erank},"
                  r"\mathrm{params})$" "\n(blue $+$, red $-$)", fontsize=6.2)
    cbA.ax.tick_params(labelsize=6.0)

    fig.subplots_adjust(left=0.30, right=0.99, top=0.90, bottom=0.02)
    out = os.path.join(FIGDIR, "fig5_partials.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    f4 = make_fig4()
    f5 = make_fig5()
    for f in (f4, f5):
        sz = os.path.getsize(f)
        print("WROTE %s  (%d bytes)" % (f, sz))
    
    mat = parse_portability()
    ot, _ = parse_space("FULLZOO-tss")
    os_, _ = parse_space("FULLZOO-sss")
    dec = parse_decomp()
    print("portability rows:", len(mat), "| partial proxies tss/sss:",
          len(ot), "/", len(os_), "| decomp rows:", len(dec))
