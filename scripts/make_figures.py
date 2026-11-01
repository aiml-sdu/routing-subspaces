"""Figures of the routing-subspaces paper, regenerated from the stored results.

    python scripts/make_figures.py [--out DIR]

The figures use Times New Roman; without it, matplotlib falls back to another serif
font and the layout can shift slightly.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/results").is_dir())
# Output directory: --out DIR, default outputs/figures/ at the repository root.
if "--out" in sys.argv:
    OUT = Path(sys.argv[sys.argv.index("--out") + 1])
else:
    OUT = REPO / "outputs/figures"
BOOTSTRAP_SUMMARY = REPO / "src/results/matrix_bootstrap/paired_bootstrap_summary.csv"

# ---------------------------------------------------------------------------
# Semantic palette shared by all figures.
# Treat these as the single source of truth; paper-specific accents below
# must not override the semantic role of EVAL / DEPLOY / METHOD / CONTROL.
# ---------------------------------------------------------------------------
COLOR_EVAL = "#D95F02"  # orange
COLOR_DEPLOY = "#1B9E77"  # teal-green
COLOR_METHOD = "#1F4E79"  # deep blue (proposed/clamp/ISS)
COLOR_PASS = "#2CA02C"
COLOR_BORDER = "#E6B800"
COLOR_FAIL = "#7F7F7F"
COLOR_CTRL = "#9E9E9E"

# ---------------------------------------------------------------------------
# 3B1B-inspired light-theme palette (routing-paper accents)
# ---------------------------------------------------------------------------
BG = "#FFFFFF"  # plain white background
SOFT = "#F3F4F6"  # very soft grey for shaded panels (if needed)
INK = "#1F1E1B"  # ink text, slightly warm

BLUE = COLOR_METHOD
CYAN = "#2563EB"  # secondary cool tone
SKY = "#60A5FA"  # lighter blue (Gemma-2-9B pair with BLUE)
AMBER = COLOR_EVAL
GOLD = "#B45309"  # darker amber
GREEN = COLOR_DEPLOY
ROSE = "#BE185D"  # magenta rose
PURP = "#6D28D9"  # purple accent
RED = "#B91C1C"  # signal red

GREY = COLOR_CTRL
MUTE = "#9CA3AF"
LIGHT = "#D1D5DB"
FAINT = "#E5E7EB"

# Family palette (4 families)
PAL_FAM = [BLUE, SKY, GREEN, AMBER]
# Intervention palette (3 interventions)
PAL_INT = [BLUE, AMBER, GREEN]
CTRL = GREY

FAMILY = [
    "Times New Roman",
    "Nimbus Roman",
    "Liberation Serif",
    "TeX Gyre Termes",
    "DejaVu Serif",
    "serif",
]

mpl_rc = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "serif",
    "font.serif": FAMILY,
    "font.size": 9.8,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#4B5563",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "axes.grid": False,
    "axes.axisbelow": True,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.0,
    "patch.linewidth": 0,
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 160,
    "mathtext.fontset": "stix",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
}
plt.rcParams.update(mpl_rc)


def grid(ax, which="both"):
    """Subtle inline grid with minor guides between labelled values."""
    axes = {"x", "y"} if which == "both" else {which}
    if "x" in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if "y" in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(
        True, which="major", axis=which, color="#CBD5E1", lw=0.38, ls=(0, (2.2, 3.0)), alpha=0.48
    )
    ax.grid(
        True, which="minor", axis=which, color="#EEF2F7", lw=0.26, ls=(0, (0.8, 2.0)), alpha=0.78
    )
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {name}.pdf and {name}.png")


# ---------------------------------------------------------------------------
# Path-patching recovery
# ---------------------------------------------------------------------------
# exact recovery fractions from path_patching_attention.json (n=48 questions, gap 17/48)
G2B_SINGLE = [("L14", 4 / 17), ("L15", -1 / 17), ("L16", 2 / 17), ("L17", 6 / 17)]
G2B_WINDOW = [("L14-16", 7 / 17), ("L14-17", 16 / 17), ("L12-17", 19 / 17)]
FAMILIES = [
    ("Gemma-2-2B", "L14-17", 0.941, 15.5 / 26),
    ("Llama-3-8B", "L14-17", 1.000, 15.5 / 32),
    ("Phi-3-mini", "L18-21", 0.935, 19.5 / 32),
    ("Qwen-2.5-7B", "L15-18", 0.519, 16.5 / 28),  # fresh retrain, seed 42, n=48 (RESULTS s10)
    ("Gemma-2-9B", "L21-27", 0.581, 24.0 / 42),
]


def path_patching():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    fig.subplots_adjust(left=0.08, right=0.99, wspace=0.36, bottom=0.22, top=0.88)

    # ---- (a) per-layer vs window bars --------------------------------------
    names = [x[0] for x in G2B_SINGLE] + [x[0] for x in G2B_WINDOW]
    vals = [x[1] for x in G2B_SINGLE] + [x[1] for x in G2B_WINDOW]
    colors = [MUTE] * len(G2B_SINGLE) + [BLUE] * len(G2B_WINDOW)
    xs = np.arange(len(names))
    ax1.bar(xs, vals, color=colors, width=0.68, edgecolor="none", zorder=3)
    ax1.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax1.axhline(0.0, color=INK, lw=0.6, alpha=0.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("recovery fraction")
    ax1.set_title("(a) Gemma-2-2B sandbag: layer vs. window", loc="left", color=INK)
    ax1.set_ylim(-0.2, 1.48)
    ax1.axvline(len(G2B_SINGLE) - 0.5, color=INK, lw=0.5, alpha=0.3)
    ax1.text(
        len(G2B_SINGLE) / 2 - 0.5,
        1.42,
        "single layer",
        ha="center",
        va="top",
        fontsize=8.5,
        color=GREY,
        style="italic",
    )
    ax1.text(
        len(G2B_SINGLE) + (len(G2B_WINDOW) - 1) / 2,
        1.42,
        "contiguous window",
        ha="center",
        va="top",
        fontsize=8.5,
        color=BLUE,
        style="italic",
    )
    # Value labels above each bar (never colliding with the group labels)
    for x, v in zip(xs, vals):
        y = (v + 0.04) if v >= 0 else (v - 0.08)
        ax1.text(x, y, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    grid(ax1)

    # ---- (b) recovery vs. relative depth -----------------------------------
    depths = np.array([f[3] for f in FAMILIES])
    recs = np.array([f[2] for f in FAMILIES])
    fams = [f[0] for f in FAMILIES]
    # Shaded depth band
    ax2.axvspan(0.50, 0.62, color=BLUE, alpha=0.10, zorder=0)
    ax2.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax2.scatter(depths, recs, color=BLUE, s=64, zorder=5, edgecolor="white", linewidth=1.1)
    # Labels placed with short leader lines back to each point so the
    # assignment is unambiguous.
    # All offsets keep the label text inside the plotted axis region and
    # clear of other labels.
    label_offsets = {
        "Gemma-2-2B": (-0.030, 0.085),
        "Llama-3-8B": (0.000, -0.100),
        "Phi-3-mini": (0.050, -0.085),
        "Qwen-2.5-7B": (0.055, -0.050),
        "Gemma-2-9B": (-0.050, 0.040),
    }
    for f, d, r in zip(fams, depths, recs):
        dx, dy = label_offsets[f]
        xt, yt = d + dx, r + dy
        if dx > 0:
            ha = "left"
        elif dx < 0:
            ha = "right"
        else:
            ha = "center"
        ax2.plot([d, xt], [r, yt], color=INK, lw=0.7, alpha=0.6, zorder=4)
        ax2.text(xt, yt, f, fontsize=9, color=INK, ha=ha, va="center", zorder=6)
    ax2.set_xlim(0.40, 0.78)
    ax2.set_ylim(0.40, 1.16)
    ax2.set_xlabel("routing-window relative depth")
    ax2.set_ylabel("recovery fraction")
    ax2.set_title("(b) sandbag, cross-family", loc="left", color=INK)
    ax2.text(
        0.56,
        1.145,
        r"depth $\approx 0.55$ band",
        fontsize=9,
        color=BLUE,
        ha="center",
        va="top",
        style="italic",
    )
    grid(ax2)

    save(fig, "path_patching")


# ---------------------------------------------------------------------------
# Routing object + three interventions (synthetic 2D geometry)
# ---------------------------------------------------------------------------
def routing_geometry():
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.9), sharey=True)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.18, wspace=0.05)

    rng = np.random.default_rng(42)
    mu_e = np.array([2.80, 1.55])
    mu_d = np.array([0.80, 1.55])
    sig = 0.18
    E0 = rng.multivariate_normal(mu_e, sig * np.eye(2), 70)
    D0 = rng.multivariate_normal(mu_d, sig * np.eye(2), 70)
    mu_e_obs = E0.mean(0)
    mu_d_obs = D0.mean(0)
    m = 0.5 * (mu_e_obs + mu_d_obs)
    d = mu_e_obs - mu_d_obs
    d_hat = d / np.linalg.norm(d)
    sigma_e = np.std((E0 - mu_e_obs) @ d_hat)
    sigma_d = np.std((D0 - mu_d_obs) @ d_hat)

    def setup(ax, title):
        ax.set_xlim(-0.45, 3.95)
        ax.set_ylim(-0.3, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.annotate(
            "",
            xy=(3.75, 0.10),
            xytext=(-0.30, 0.10),
            arrowprops=dict(arrowstyle="-|>", color="#9CA3AF", lw=0.9),
        )
        ax.text(3.8, 0.10, r"$\hat{\mathbf{d}}_L$", fontsize=10, color="#6B7280", va="center")
        ax.text(-0.35, 2.87, title, fontsize=10, color=INK)

    # -- (a) two clusters + d_L ---------------------------------------------
    a = axes[0]
    a.scatter(D0[:, 0], D0[:, 1], s=15, c=AMBER, alpha=0.7, edgecolor="white", lw=0.4, zorder=2)
    a.scatter(E0[:, 0], E0[:, 1], s=15, c=BLUE, alpha=0.7, edgecolor="white", lw=0.4, zorder=2)
    a.annotate(
        "",
        xy=mu_e_obs,
        xytext=mu_d_obs,
        arrowprops=dict(arrowstyle="-|>", mutation_scale=20, color=INK, lw=2.2),
    )
    a.plot(*mu_e_obs, "o", ms=12, mfc=BLUE, mec="white", mew=1.4, zorder=4)
    a.plot(*mu_d_obs, "o", ms=12, mfc=AMBER, mec="white", mew=1.4, zorder=4)
    a.plot(*m, "D", ms=11, mfc=GREEN, mec="white", mew=1.4, zorder=5)
    # d_L label above the arrow, out of the way of cluster points
    a.text(
        0.5 * (mu_d_obs[0] + mu_e_obs[0]),
        mu_e_obs[1] + 0.40,
        r"$\mathbf{d}_L$",
        fontsize=13,
        color=INK,
        ha="center",
        style="italic",
        fontweight="bold",
    )
    # mu labels OUTSIDE the clusters, with short leader lines to markers
    a.text(
        mu_e_obs[0] + 0.55,
        mu_e_obs[1] + 0.70,
        r"$\mu^{\mathrm{eval}}$",
        fontsize=11,
        color=BLUE,
        fontweight="bold",
        ha="left",
    )
    a.plot(
        [mu_e_obs[0] + 0.18, mu_e_obs[0] + 0.50],
        [mu_e_obs[1] + 0.22, mu_e_obs[1] + 0.65],
        color=BLUE,
        lw=0.8,
        zorder=3,
    )
    a.text(
        mu_d_obs[0] - 0.55,
        mu_d_obs[1] + 0.70,
        r"$\mu^{\mathrm{dep}}$",
        fontsize=11,
        color=AMBER,
        fontweight="bold",
        ha="right",
    )
    a.plot(
        [mu_d_obs[0] - 0.18, mu_d_obs[0] - 0.50],
        [mu_d_obs[1] + 0.22, mu_d_obs[1] + 0.65],
        color=AMBER,
        lw=0.8,
        zorder=3,
    )
    # Suppression target label sits below the scatter with a white halo.
    tm = a.text(
        m[0],
        0.45,
        r"$0$",
        fontsize=14,
        color=GREEN,
        ha="center",
        va="center",
        style="italic",
        fontweight="bold",
        zorder=6,
    )
    tm.set_path_effects([mpe.withStroke(linewidth=3, foreground="white")])
    a.plot([m[0], m[0]], [m[1] - 0.18, 0.60], color=GREEN, lw=1.0, zorder=3)
    setup(a, "(a) two clusters")

    # -- (b) remove evaluation coordinate ------------------------------------
    b = axes[1]
    lam = 1.0
    E_rem = E0 - lam * sigma_e * d_hat * 4.5
    b.scatter(E0[:, 0], E0[:, 1], s=15, c=LIGHT, alpha=0.7, edgecolor="white", lw=0.4, zorder=1)
    b.scatter(D0[:, 0], D0[:, 1], s=15, c=AMBER, alpha=0.7, edgecolor="white", lw=0.4, zorder=2)
    b.scatter(
        E_rem[:, 0], E_rem[:, 1], s=15, c=BLUE, alpha=0.7, edgecolor="white", lw=0.4, zorder=3
    )
    # Shift arrow along top
    b.annotate(
        "",
        xy=(mu_e_obs[0] - lam * sigma_e * 4.5, 2.45),
        xytext=(mu_e_obs[0], 2.45),
        arrowprops=dict(arrowstyle="->", mutation_scale=18, color=BLUE, lw=1.8, linestyle="-"),
    )
    b.text(
        (mu_e_obs[0] + (mu_e_obs[0] - lam * sigma_e * 4.5)) / 2,
        2.52,
        r"$-\lambda\,\sigma\,\hat{\mathbf{d}}_L$",
        fontsize=9,
        color=BLUE,
        ha="center",
        va="bottom",
    )
    setup(b, "(b) remove evaluation")

    # -- (c) add ordinary-use coordinate -------------------------------------
    c = axes[2]
    D_add = D0 + lam * sigma_d * d_hat * 4.5
    c.scatter(D0[:, 0], D0[:, 1], s=15, c=LIGHT, alpha=0.7, edgecolor="white", lw=0.4, zorder=1)
    c.scatter(E0[:, 0], E0[:, 1], s=15, c=BLUE, alpha=0.7, edgecolor="white", lw=0.4, zorder=2)
    c.scatter(
        D_add[:, 0], D_add[:, 1], s=15, c=AMBER, alpha=0.7, edgecolor="white", lw=0.4, zorder=3
    )
    c.annotate(
        "",
        xy=(mu_d_obs[0] + lam * sigma_d * 4.5, 0.55),
        xytext=(mu_d_obs[0], 0.55),
        arrowprops=dict(arrowstyle="->", mutation_scale=18, color=AMBER, lw=1.8),
    )
    c.text(
        (mu_d_obs[0] + (mu_d_obs[0] + lam * sigma_d * 4.5)) / 2,
        0.45,
        r"$+\lambda\,\sigma\,\hat{\mathbf{d}}_L$",
        fontsize=9,
        color=AMBER,
        ha="center",
        va="top",
    )
    setup(c, "(c) add ordinary use")

    # -- (d) joint intervention ----------------------------------------------
    dd = axes[3]

    def proj(P):
        return P + ((m - P) @ d_hat)[:, None] * d_hat

    E_cl = proj(E0)
    D_cl = proj(D0)
    dd.scatter(E0[:, 0], E0[:, 1], s=15, c=LIGHT, alpha=0.55, edgecolor="white", lw=0.3, zorder=1)
    dd.scatter(D0[:, 0], D0[:, 1], s=15, c=LIGHT, alpha=0.55, edgecolor="white", lw=0.3, zorder=1)
    dd.scatter(
        D_cl[:, 0], D_cl[:, 1], s=15, c=AMBER, alpha=0.85, edgecolor="white", lw=0.4, zorder=3
    )
    dd.scatter(
        E_cl[:, 0], E_cl[:, 1], s=15, c=BLUE, alpha=0.85, edgecolor="white", lw=0.4, zorder=3
    )
    dd.plot(*m, "o", ms=11, mfc=GREEN, mec="white", mew=1.4, zorder=5)
    dd.axvline(
        m[0], color=GREEN, lw=1.6, ls=(0, (5, 3)), alpha=0.85, ymin=0.07, ymax=0.93, zorder=2
    )
    dd.text(
        m[0] + 0.10,
        0.40,
        r"$\langle \mathbf{h}, \hat{\mathbf{d}}_L\rangle \to 0$",
        fontsize=9,
        color=GREEN,
        ha="left",
        va="bottom",
    )
    for pre, col in [(mu_e_obs, BLUE), (mu_d_obs, AMBER)]:
        post = pre + ((m - pre) @ d_hat) * d_hat
        dd.annotate(
            "",
            xy=post,
            xytext=pre,
            arrowprops=dict(
                arrowstyle="->", mutation_scale=15, color=col, lw=1.3, linestyle=(0, (4, 3))
            ),
        )
    setup(dd, "(d) joint intervention")

    # Shared legend
    h = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=BLUE, markersize=8, label="evaluation framing"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=AMBER,
            markersize=8,
            label="ordinary-use framing",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", color=LIGHT, markersize=8, label="pre-intervention"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=GREEN,
            markersize=9,
            label="suppressed coordinate",
        ),
    ]
    fig.legend(
        handles=h,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.4,
        columnspacing=1.8,
        fontsize=9,
    )
    save(fig, "routing_geometry")


# ---------------------------------------------------------------------------
# 12-cell cross-family x cross-behavior matrix (slope charts)
# ---------------------------------------------------------------------------
TASKS = ["sandbag", "sycophancy", "refusal"]
FAMILY_ORDER = ["Gemma-2-2B", "Gemma-2-9B", "Qwen-2.5-7B", "Llama-3-8B"]
CELL_TO_SERIES = {
    "gemma2b_sandbag": ("Gemma-2-2B", "sandbag"),
    "gemma2b_syc": ("Gemma-2-2B", "sycophancy"),
    "gemma2b_refusal": ("Gemma-2-2B", "refusal"),
    "gemma9b_sandbag": ("Gemma-2-9B", "sandbag"),
    "gemma9b_syc": ("Gemma-2-9B", "sycophancy"),
    "gemma9b_refusal": ("Gemma-2-9B", "refusal"),
    "qwen7b_sandbag": ("Qwen-2.5-7B", "sandbag"),
    "qwen7b_syc": ("Qwen-2.5-7B", "sycophancy"),
    "qwen7b_refusal": ("Qwen-2.5-7B", "refusal"),
    "llama8b_sandbag": ("Llama-3-8B", "sandbag"),
    "llama8b_syc": ("Llama-3-8B", "sycophancy"),
    "llama8b_refusal": ("Llama-3-8B", "refusal"),
}


def load_matrix_summary():
    """Load matrix gaps and asymmetric paired-bootstrap intervals."""
    matrix = {fam: {} for fam in FAMILY_ORDER}
    intervals = {fam: {} for fam in FAMILY_ORDER}
    with BOOTSTRAP_SUMMARY.open(newline="") as f:
        for row in csv.DictReader(f):
            key = CELL_TO_SERIES.get(row["cell"])
            if key is None:
                continue
            fam, task = key
            b = float(row["baseline_gap"])
            c = float(row["clamp_gap"])
            b_low = float(row["baseline_gap_ci95_low"])
            b_high = float(row["baseline_gap_ci95_high"])
            c_low = float(row["clamp_gap_ci95_low"])
            c_high = float(row["clamp_gap_ci95_high"])
            matrix[fam][task] = (b, c)
            intervals[fam][task] = (
                (b - b_low, b_high - b),
                (c - c_low, c_high - c),
            )

    missing = [
        (fam, task)
        for fam in FAMILY_ORDER
        for task in TASKS
        if task not in matrix[fam] or task not in intervals[fam]
    ]
    if missing:
        raise ValueError(f"missing bootstrap rows for {missing}")
    return matrix, intervals


MATRIX, MATRIX_CI = load_matrix_summary()


def matrix():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.99, wspace=0.10)
    n_per_task = {"sandbag": "n=200", "sycophancy": "n=120", "refusal": "n=28"}
    for panel, (ax, task) in enumerate(zip(axes, TASKS), start=1):
        xs = [0, 1]
        # small per-family x-jitter so the error bars don't stack
        jitters = [-0.10, -0.035, 0.035, 0.10]
        for j, fam in enumerate(FAMILY_ORDER):
            b, c = MATRIX[fam][task]
            (b_lo, b_hi), (c_lo, c_hi) = MATRIX_CI[fam][task]
            xj = [xs[0] + jitters[j], xs[1] + jitters[j]]
            yerr = np.array([[b_lo, c_lo], [b_hi, c_hi]])
            ax.plot(
                xj,
                [b, c],
                color=PAL_FAM[j],
                lw=2.4,
                alpha=0.92,
                zorder=3,
                marker="o",
                ms=7.4,
                mec="white",
                mew=1.0,
                label=fam if ax is axes[0] else None,
            )
            ax.errorbar(
                xj,
                [b, c],
                yerr=yerr,
                fmt="none",
                ecolor=PAL_FAM[j],
                elinewidth=1.25,
                capsize=3.4,
                capthick=1.25,
                alpha=0.96,
                zorder=5,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(["baseline", "post"])
        ax.set_xlim(-0.22, 1.22)
        ax.set_title(f"({chr(96 + panel)}) {task} ({n_per_task[task]})", loc="left", color=INK)
        ax.axhline(0, color=INK, lw=0.6, alpha=0.45, zorder=1)
        ax.set_ylim(-0.20, 0.90)
        grid(ax)
    axes[0].set_ylabel("evaluation/deployment gap")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handlelength=2.3,
        columnspacing=2.2,
        fontsize=9.5,
    )
    save(fig, "matrix")


# ---------------------------------------------------------------------------
# Dose-response + matched-dose controls
# ---------------------------------------------------------------------------
DOSE_X = [0.0, 0.5, 1.0, 2.0]
DOSE_REM = [0.283, 0.233, 0.007, -0.013]
DOSE_ADD = [0.283, 0.247, 0.017, 0.000]
DOSE_CLAMP = [0.283, 0.237, 0.017, 0.003]
CTL_LAM1 = [
    ("joint intervention", DOSE_CLAMP[2], BLUE),
    ("routing remove", DOSE_REM[2], BLUE),
    ("random direction", 0.287, GREY),
    (r"non-routing $L6$", 0.307, GREY),
    (r"non-routing $L22$", 0.273, GREY),
]


def dose_response():
    """Four-panel characterisation: dose response, matched-dose controls,
    semantic specificity, log-scale direction norms. Replaces the earlier
    specificity figure."""
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.2],
        height_ratios=[1.0, 1.0],
        left=0.085,
        right=0.985,
        top=0.93,
        bottom=0.10,
        wspace=0.55,
        hspace=0.55,
    )
    # Projection removal is scale-free in lambda, while the add operator multiplies the raw
    # direction, so the two arms get separate dose axes (panel a, left and right).
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.10, width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(sub[0, 0])
    ax_add = fig.add_subplot(sub[0, 1], sharey=ax)
    axc = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_geo = fig.add_subplot(gs[1, 1])
    baseline = 0.283
    mkw = dict(markersize=6, markeredgecolor="white", markeredgewidth=1.0, linewidth=2.2)
    for a in (ax, ax_add):
        a.axhline(0, color=INK, lw=0.6, alpha=0.4)
        a.axhline(baseline, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
        a.set_xticks(DOSE_X)
        a.set_xticklabels(["0", ".5", "1", "2"])
        grid(a)
    ax.text(
        0.02,
        baseline + 0.008,
        "baseline gap",
        fontsize=8.5,
        color=GREY,
        ha="left",
        va="bottom",
        style="italic",
    )
    ax.plot(DOSE_X, DOSE_REM, marker="o", color=BLUE, label="remove eval.", **mkw)
    ax.plot(DOSE_X, DOSE_CLAMP, marker="^", color=GREEN, label="joint", **mkw)
    ax_add.plot(DOSE_X, DOSE_ADD, marker="s", color=AMBER, label="add deploy.", **mkw)
    ax.set_xlabel(r"projection dose $\lambda$")
    ax_add.set_xlabel(r"multiple of raw $\mathbf{d}_L$")
    ax.set_ylabel("evaluation/deployment gap")
    plt.setp(ax_add.get_yticklabels(), visible=False)
    ax.set_title(
        r"(a) dose response (Gemma-2-2B sandbag, $L14\!-\!L17$)", loc="left", color=INK, pad=6
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.08),
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        borderaxespad=0.2,
        handlelength=1.4,
    )
    ax_add.text(2.05, 0.075, "add on\ndeployment", ha="right", va="center", fontsize=8, color=AMBER)

    labels = [r[0] for r in CTL_LAM1]
    vals = [r[1] for r in CTL_LAM1]
    colors = [r[2] for r in CTL_LAM1]
    ys = np.arange(len(labels))[::-1]
    axc.barh(ys, vals, color=colors, edgecolor="none", height=0.60, zorder=3)
    axc.axvline(baseline, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    axc.axvline(0, color=INK, lw=0.6, alpha=0.4)
    axc.set_yticks(ys)
    axc.set_yticklabels(labels, fontsize=9)
    axc.set_xlabel(r"gap at $\lambda = 1$")
    axc.set_title(r"(b) controls at $\lambda = 1$", loc="left", color=INK)
    axc.set_xlim(-0.04, 0.37)
    for y, v in zip(ys, vals):
        axc.text(v + 0.008, y, f"{v:+.2f}", fontsize=8.5, va="center", color=INK)
    grid(axc, which="x")

    # ------ bottom row: semantic specificity + log-scale norms ------------
    labels = [s[0] for s in SPEC]
    vals = [s[1] for s in SPEC]
    ys = np.arange(len(labels))[::-1]
    colors = [BLUE] + [GREY] * 4
    ax_spec.barh(ys, vals, color=colors, edgecolor="none", height=0.62, zorder=3)
    ax_spec.axvline(0.183, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax_spec.set_yticks(ys)
    ax_spec.set_yticklabels(labels, fontsize=9)
    ax_spec.set_xlabel(r"evaluation accuracy after remove at $\lambda=2$")
    ax_spec.set_title("(c) semantic specificity", loc="left", color=INK)
    ax_spec.set_xlim(0, 0.55)
    ax_spec.set_ylim(-0.6, 4.9)
    ax_spec.text(
        0.183,
        4.75,
        "evaluation baseline",
        fontsize=8.5,
        color=GREY,
        ha="center",
        va="bottom",
        style="italic",
    )
    for y, v in zip(ys, vals):
        ax_spec.text(v + 0.01, y, f"{v:.2f}", fontsize=8.5, va="center", color=INK)
    grid(ax_spec, which="x")

    # log-scale norms (Gemma-2-2B + Phi-3-mini)
    geo_labels = []
    base_vals = []
    fused_vals = []
    for name, d in GEO.items():
        for lname in d["layers"]:
            geo_labels.append(lname)
        base_vals += d["base"]
        fused_vals += d["fused"]
    xs = np.arange(len(geo_labels))
    w = 0.36
    ax_geo.bar(xs - w / 2, base_vals, width=w, color=GREY, label="base", zorder=3)
    ax_geo.bar(xs + w / 2, fused_vals, width=w, color=BLUE, label="fine-tuned", zorder=3)
    ax_geo.set_xticks(xs)
    ax_geo.set_xticklabels(geo_labels, fontsize=8.5)
    ax_geo.set_yscale("log")
    ax_geo.set_ylim(0.5, 85)
    ax_geo.set_ylabel(r"$\|\mathbf{d}_L\|_2$ (log)")
    ax_geo.set_title("(d) direction norms, base vs. fine-tuned", loc="left", color=INK)
    ax_geo.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax_geo.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax_geo.yaxis.set_minor_formatter(NullFormatter())
    split = len(next(iter(GEO.values()))["layers"]) - 0.5
    ax_geo.axvline(split, color=INK, lw=0.5, alpha=0.25)
    fam_names = list(GEO.keys())
    group_sizes = [len(d["layers"]) for d in GEO.values()]
    centers = [np.mean(range(0, group_sizes[0])), np.mean(range(group_sizes[0], sum(group_sizes)))]
    for c, nm in zip(centers, fam_names):
        ax_geo.text(c, 80, nm, ha="center", va="top", fontsize=8.5, color=INK, style="italic")
    for i, (b, f) in enumerate(zip(base_vals, fused_vals)):
        ax_geo.text(
            i + w / 2, f * 1.18, rf"$\times{f / b:.1f}$", ha="center", fontsize=7.5, color=INK
        )
    ax_geo.legend(
        loc="upper center",
        fontsize=9,
        frameon=False,
        ncol=2,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.13),
    )
    ax_geo.grid(True, which="major", axis="y", color="#CBD5E1", lw=0.5, ls=(0, (3, 3)), alpha=0.70)
    ax_geo.grid(True, which="minor", axis="y", color="#EDEFF2", lw=0.3, ls=(0, (1, 2)), alpha=0.55)
    ax_geo.set_axisbelow(True)

    save(fig, "dose_response")


# ---------------------------------------------------------------------------
# Semantic specificity (bar + baseline reference)
# ---------------------------------------------------------------------------
SPEC = [
    ("routing direction", 0.460),
    ("concrete/abstract", 0.172),
    ("emotional/neutral", 0.168),
    ("formal/casual", 0.193),
    ("math/science", 0.177),
]


def specificity():
    """Combined specificity + direction geometry figure.

    Panel (a): semantic contrast directions vs. the routing direction on
               Gemma-2-2B sandbag at lambda=2 (only the routing direction
               raises evaluation accuracy).
    Panel (b): log-scale norms of d_L on base vs. fine-tuned model for the
               Gemma-2-2B and Phi-3-mini routing windows.
    """
    fig = plt.figure(figsize=(7.4, 2.5))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.05, 1.25], wspace=0.35, left=0.08, right=0.985, top=0.88, bottom=0.22
    )

    # ---- (a) semantic specificity ---------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    labels = [s[0] for s in SPEC]
    vals = [s[1] for s in SPEC]
    ys = np.arange(len(labels))[::-1]
    colors = [BLUE] + [GREY] * 4
    ax.barh(ys, vals, color=colors, edgecolor="none", height=0.62, zorder=3)
    ax.axvline(0.183, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax.text(
        0.193,
        -0.80,
        "evaluation baseline",
        fontsize=8.5,
        color=GREY,
        ha="left",
        va="center",
        style="italic",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"evaluation accuracy after remove at $\lambda=2$")
    ax.set_title("(a) semantic specificity", loc="left", color=INK)
    ax.set_xlim(0, 0.55)
    for y, v in zip(ys, vals):
        ax.text(v + 0.01, y, f"{v:.2f}", fontsize=8.5, va="center", color=INK)
    grid(ax)

    # ---- (b) log-scale norms (merged two families into one plot) ---------
    ax2 = fig.add_subplot(gs[0, 1])
    # Concatenate Gemma-2-2B and Phi-3-mini layers, keep per-family groups
    labels = []
    base_vals = []
    fused_vals = []
    separators = []
    cur = 0
    for name, d in GEO.items():
        for lname in d["layers"]:
            labels.append(f"{name[:6]}\\,{lname}" if False else lname)
        base_vals += d["base"]
        fused_vals += d["fused"]
        cur += len(d["layers"])
        separators.append(cur - 0.5)
    xs = np.arange(len(labels))
    w = 0.36
    ax2.bar(xs - w / 2, base_vals, width=w, color=GREY, label="base", zorder=3)
    ax2.bar(xs + w / 2, fused_vals, width=w, color=BLUE, label="fine-tuned", zorder=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 70)
    ax2.set_ylabel(r"$\|\mathbf{d}_L\|_2$ (log)")
    ax2.set_title("(b) direction norms, base vs. fine-tuned", loc="left", color=INK)
    ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    # divider between Gemma-2-2B and Phi-3-mini
    split = len(next(iter(GEO.values()))["layers"]) - 0.5
    ax2.axvline(split, color=INK, lw=0.5, alpha=0.25)
    # family labels above each group
    fam_names = list(GEO.keys())
    group_sizes = [len(d["layers"]) for d in GEO.values()]
    centers = [np.mean(range(0, group_sizes[0])), np.mean(range(group_sizes[0], sum(group_sizes)))]
    for c, nm in zip(centers, fam_names):
        ax2.text(c, 68, nm, ha="center", va="top", fontsize=8.5, color=INK, style="italic")
    # amplification labels above each fused bar
    for i, (b, f) in enumerate(zip(base_vals, fused_vals)):
        ax2.text(i + w / 2, f * 1.18, rf"$\times{f / b:.1f}$", ha="center", fontsize=7.5, color=INK)
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(True, which="major", axis="y", color="#CBD5E1", lw=0.55, ls=(0, (3, 3)), alpha=0.75)
    ax2.grid(True, which="minor", axis="y", color="#E5E7EB", lw=0.4, ls=(0, (1.4, 2.8)), alpha=0.7)
    ax2.set_axisbelow(True)
    save(fig, "specificity")


# ---------------------------------------------------------------------------
# Direction geometry on a log scale
# ---------------------------------------------------------------------------
# Direction norms read from the stored geometry files (full precision, so that ratios are not
# computed from rounded values):
# src/results/steering/direction_geometry_{gemma,phi3}.json
GEO = {
    "Gemma-2-2B": {
        "layers": ["L14", "L15", "L16", "L17"],
        "fused": [5.5094, 6.8266, 10.2265, 9.2680],
        "base": [1.1211, 0.8259, 0.9817, 1.3002],
    },
    "Phi-3-mini": {
        "layers": ["L18", "L19", "L20", "L21"],
        "fused": [25.2431, 32.3758, 22.8615, 30.5829],
        "base": [6.2279, 8.0936, 6.4391, 7.4555],
    },
}


def direction_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.32, bottom=0.18, top=0.88)
    for ax, (name, d) in zip(axes, GEO.items()):
        xs = np.arange(len(d["layers"]))
        w = 0.36
        ax.bar(xs - w / 2, d["base"], width=w, color=GREY, label="base", zorder=3)
        ax.bar(xs + w / 2, d["fused"], width=w, color=BLUE, label="fine-tuned", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(d["layers"])
        ax.set_title(name, loc="left", color=INK)
        ax.set_ylabel(r"$\|\mathbf{d}_L\|_2$ (log scale)")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 70)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        for i, (b, f) in enumerate(zip(d["base"], d["fused"])):
            ax.text(
                i + w / 2, f * 1.18, rf"$\times{f / b:.1f}$", ha="center", fontsize=8.5, color=INK
            )
        ax.grid(True, which="major", axis="y", color="#CBD5E1", lw=0.55, ls=(0, (3, 3)), alpha=0.7)
        ax.grid(
            True, which="minor", axis="y", color="#E5E7EB", lw=0.4, ls=(0, (1.4, 2.8)), alpha=0.65
        )
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=9)
    save(fig, "direction_geometry")


# ---------------------------------------------------------------------------
# H11 rank-1 weight edit (appendix)
# ---------------------------------------------------------------------------
H11 = [
    ("Qwen-2.5-7B refusal", 56.7, -5.42),
    ("Gemma-2-2B sandbag", 27.5, -3.75),
    ("Gemma-2-9B sandbag", 68.75, 63.75),
    ("Llama-3-8B sandbag", 42.5, 40.0),
]


def h11():
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.22)
    labels = [h[0] for h in H11]
    pre = [h[1] for h in H11]
    post = [h[2] for h in H11]
    ys = np.arange(len(labels))[::-1]
    w = 0.38
    ax.barh(ys + w / 2, pre, height=w, color=GREY, label="before", zorder=3)
    ax.barh(ys - w / 2, post, height=w, color=BLUE, label="after", zorder=3)
    ax.axvline(0, color=INK, lw=0.7, alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("evaluation/deployment gap (pp)")
    ax.set_xlim(-12, 95)
    for y, p0, p1 in zip(ys, pre, post):
        ax.text(
            max(p0, p1) + 2, y, f"{p0 - p1:.1f} pp closure", fontsize=8.5, va="center", color=INK
        )
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=9)
    grid(ax, which="x")
    save(fig, "h11")


# ---------------------------------------------------------------------------
# 3D decision manifold + layer dynamics (light theme, 3B1B feel)
# ---------------------------------------------------------------------------
def topology():
    """Measured depth projection (replaces the synthetic schematic of the submitted version).

    Reads src/results/supplementary/depth_projection_gemma2b_sandbag.json, produced by
    src/scripts/causal/depth_projection.py on the fresh primary Gemma-2-2B sandbagging
    checkpoint. Panel (a): projection onto the window direction; panel (b): onto each
    layer's own direction. Shading is the 95% question-bootstrap interval."""
    import json

    src = REPO / "src/results/supplementary/depth_projection_gemma2b_sandbag.json"
    d = json.load(open(src))
    win = d["window"]
    curves = [
        ("(a) window direction", d["results"]["window_direction"]["window_mean"]),
        ("(b) each layer's own direction", d["results"]["own_layer_direction"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.20, wspace=0.28)
    for ax, (title, rows) in zip(axes, curves):
        L = np.array([r["layer"] for r in rows])
        ax.axvspan(win[0] - 0.5, win[-1] + 0.5, color=BLUE, alpha=0.10, zorder=0)
        for key, col, mk, lab in (
            ("eval", BLUE, "o", "evaluation"),
            ("deploy", AMBER, "s", "deployment"),
        ):
            m = np.array([r[f"{key}_mean"] for r in rows])
            lo = np.array([r[f"{key}_ci"][0] for r in rows])
            hi = np.array([r[f"{key}_ci"][1] for r in rows])
            ax.fill_between(L, lo, hi, color=col, alpha=0.25, lw=0, zorder=3)
            ax.plot(
                L,
                m,
                color=col,
                lw=1.8,
                marker=mk,
                ms=3.6,
                mec="white",
                mew=0.5,
                label=lab,
                zorder=4,
            )
        ax.axhline(0, color=INK, lw=0.5, alpha=0.4)
        ax.set_xlabel(r"layer index $L$")
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlim(-0.5, 25.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="major", axis="y", color="#CBD5E1", lw=0.5, ls=(0, (3, 3)), alpha=0.65)
        ax.set_axisbelow(True)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
        ax.text(
            (win[0] + win[-1]) / 2,
            ax.get_ylim()[1],
            "routing window",
            color=BLUE,
            fontsize=8,
            ha="center",
            va="top",
            style="italic",
        )
    axes[0].set_ylabel(r"mean projection $\langle \mathbf{h}_L, \hat{\mathbf{d}}\rangle$")
    axes[0].legend(
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        borderaxespad=0.2,
        labelspacing=0.2,
    )
    save(fig, "topology")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path_patching()
    routing_geometry()
    matrix()
    dose_response()
    specificity()
    direction_geometry()
    h11()
    topology()
    print("done")
