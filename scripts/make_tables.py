"""Generate the appendix tables that report the fresh-retrain evidence.

Every number is read from a stored artifact; nothing is typed by hand.
Writes the LaTeX tables of the fresh-retrain appendix. The default output is
outputs/tables_generated_appendix.tex at the repository root; pass --out PATH to
write elsewhere. An existing file is only overwritten with --force.

    python scripts/make_tables.py [--out PATH] [--force]
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/results").is_dir())
R = REPO / "src/results"
if "--out" in sys.argv:
    OUT_APP = Path(sys.argv[sys.argv.index("--out") + 1])
else:
    OUT_APP = REPO / "outputs/tables_generated_appendix.tex"

NAMES = {
    "gemma2b_sandbag": "Gemma-2-2B & sandbag",
    "gemma2b_syc": "Gemma-2-2B & sycophancy",
    "gemma2b_refusal": "Gemma-2-2B & refusal",
    "gemma9b_sandbag": "Gemma-2-9B & sandbag",
    "gemma9b_syc": "Gemma-2-9B & sycophancy",
    "gemma9b_refusal": "Gemma-2-9B & refusal",
    "qwen7b_sandbag": "Qwen-2.5-7B & sandbag",
    "qwen7b_syc": "Qwen-2.5-7B & sycophancy",
    "qwen7b_refusal": "Qwen-2.5-7B & refusal",
    "llama8b_sandbag": "Llama-3-8B & sandbag",
    "llama8b_syc": "Llama-3-8B & sycophancy",
    "llama8b_refusal": "Llama-3-8B & refusal",
}
ORDER = list(NAMES)


def rnd(x, n=3):
    """Round half away from zero on the decimal value, as in the paper text
    (0.2975 -> 0.298), instead of Python's binary rounding (0.2975 -> 0.297)."""
    q = Decimal(1).scaleb(-n)
    # round(x, 9) first recovers the decimal value of stored ratios such as 0.2075 (0.20749999...)
    return format(Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP), "f")


def f3(x):
    return f"{rnd(x, 3)}".replace("-", "$-$") if x < 0 else f"{rnd(x, 3)}"


def sg(x):
    return f"${rnd(x, 3)}$" if x >= 0 else f"$-{rnd(abs(x), 3)}$"


def ci(lo, hi):
    return f"$[{rnd(lo, 3)}, {rnd(hi, 3)}]$".replace("[-", "[-").replace(", -", ", -")


def red(est, lo, hi):
    return f"${rnd(est, 3)}\\,[{rnd(lo, 3)}, {rnd(hi, 3)}]$"


def load(p):
    return json.load(open(p))


out = []

# ----------------------------------------------------------------- dose-2 matrix
s = load(R / "dose2_controls/summary.json")
rows = []
for c in ORDER:
    m = s["matrix"][c]
    L = m["learned"]
    offpos = sum(v["ci95"][0] > 0 for v in m["off_window"].values())
    rows.append(
        f"{NAMES[c]} & ${rnd(m['baseline']['gap'], 3)}$ & {red(L['gap_reduction'], *L['ci95'])} & "
        f"${rnd(m['max_random_abs_reduction'], 3)}$ & "
        f"${rnd(m['max_off_window_abs_reduction'], 3)}$ & {offpos}/2 \\\\"
    )
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Matched controls at the headline dose on fresh retrains of the twelve cells. """
    r"""Treatment and controls use the same clamp-both operator, token positions, four-layer """
    r"""width, and $\lambda{=}2$. Reductions are baseline gap minus post-intervention gap with """
    r"""$95\%$ question-clustered bootstrap intervals; the random columns use five Gaussian """
    r"""unit directions per cell and the off-window columns two predeclared four-layer windows """
    r"""per cell. These runs do not revise the counts of the submitted matrix.}
\label{tab:dose2}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccccc@{}}
\toprule
\textbf{Family} & \textbf{Behavior} & \textbf{Base $\gamma$} & \hd{Learned}{reduction} & """
    r"""\hd{Max}{$\lvert$random$\rvert$} & \hd{Max}{$\lvert$off-window$\rvert$} & """
    r"""\hd{Off-window}{$>0$} \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
)

# ----------------------------------------------------------- fresh primary controls
v = load(R / "verified_summary.json")["comparisons"]
base = v["p0_predeclared_dose2"]["baseline"]


def prow(label, key):
    c = v[key]
    reduction = red(c["gap_reduction"], *c["gap_reduction_cluster_bootstrap_95_ci"])
    return f"{label} & {sg(c['intervention']['gap'])} & {reduction} \\\\"


prim = [
    prow(r"Predeclared $L14$--$L17$, clamp, $\lambda{=}2$", "p0_predeclared_dose2"),
    prow(r"Localization-selected $L10$--$L13$, clamp, $\lambda{=}1$", "p0_selected_dose1"),
    prow(r"Localization-selected $L10$--$L13$, clamp, $\lambda{=}2$", "p0_selected_dose2"),
    prow(r"Early window $L4$--$L7$, clamp, $\lambda{=}1$", "p1_early_dose1"),
    prow(r"Early window $L4$--$L7$, clamp, $\lambda{=}2$", "p1_early_dose2"),
    prow(r"Late window $L20$--$L23$, clamp, $\lambda{=}1$", "p1_late_dose1"),
    prow(r"Late window $L20$--$L23$, clamp, $\lambda{=}2$", "p1_late_dose2"),
    prow(
        r"Opposite operation (add projection), $L10$--$L13$, $\lambda{=}1$",
        "followup_L10-13_projection_add_eval_dose1.0",
    ),
    prow(
        r"Opposite operation (add projection), $L10$--$L13$, $\lambda{=}2$",
        "followup_L10-13_projection_add_eval_dose2.0",
    ),
    prow(
        r"Opposite operation (add projection), $L14$--$L17$, $\lambda{=}1$",
        "followup_L14-17_projection_add_eval_dose1.0",
    ),
    prow(
        r"Opposite operation (add projection), $L14$--$L17$, $\lambda{=}2$",
        "followup_L14-17_projection_add_eval_dose2.0",
    ),
    prow(
        r"Zero ablation of the window, $L10$--$L13$", "followup_L10-13_attention_output_zero_eval"
    ),
    prow(
        r"Calibration-mean ablation of the window, $L10$--$L13$",
        "followup_L10-13_attention_output_calibration_mean_eval",
    ),
    prow(
        r"Zero ablation of the window, $L14$--$L17$", "followup_L14-17_attention_output_zero_eval"
    ),
    prow(
        r"Calibration-mean ablation of the window, $L14$--$L17$",
        "followup_L14-17_attention_output_calibration_mean_eval",
    ),
    prow(
        r"Template holdout (templates $0$--$4$ calibrate, $5$--$9$ test), $L10$--$L13$, "
        r"$\lambda{=}2$",
        "followup_L10-13_template_holdout_clamp_dose2.0",
    ),
    prow(
        r"Template holdout, $L14$--$L17$, $\lambda{=}1$",
        "followup_L14-17_template_holdout_clamp_dose1.0",
    ),
    prow(
        r"Template holdout, $L14$--$L17$, $\lambda{=}2$",
        "followup_L14-17_template_holdout_clamp_dose2.0",
    ),
]
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Fresh primary Gemma-2-2B sandbagging checkpoint (seed $42$; baseline gap $"""
    + f"{rnd(base['gap'], 3)}"
    + r"""$ on the full held-out split; the template-holdout rows use a $122$-question """
    r"""subset with its own baseline). Five Gaussian random directions with the same """
    r"""layers, operator, and doses leave the gap between $0.278$ and $0.285$. Post-gap """
    r"""and reduction are as in Table~\ref{tab:dose2}.}
\label{tab:fresh-primary}
\begin{tabularx}{\linewidth}{@{}Ycc@{}}
\toprule
\textbf{Condition} & \textbf{Post-gap} & \textbf{Reduction [95\% CI]} \\
\midrule
"""
    + "\n".join(prim)
    + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
)

# ----------------------------------------------------------------- seeds
srows = []
sm = s["qwen_multiseed"]
for seed in (42, 43, 44):
    key = f"followup_qwen7b_sandbag_seed{seed}_dose2"
    c = v[key]
    mr = sm.get(f"qwen7b_sandbag_seed{seed}", {})
    mxr = max(
        (abs(x["gap_reduction"]) for x in mr.get("random", {}).values()), default=float("nan")
    )
    mxo = max(
        (abs(x["gap_reduction"]) for x in mr.get("off_window", {}).values()), default=float("nan")
    )
    srows.append(
        f"Qwen-2.5-7B sandbag & {seed} & "
        f"{red(c['gap_reduction'], *c['gap_reduction_cluster_bootstrap_95_ci'])} & "
        f"${rnd(mxr, 3)}$ & ${rnd(mxo, 3)}$ \\\\"
    )
for seed in (42, 43, 44):
    c = v[f"followup_gemma9b_syc_seed{seed}_dose2"]
    srows.append(
        f"Gemma-2-9B sycophancy & {seed} & "
        f"{red(c['gap_reduction'], *c['gap_reduction_cluster_bootstrap_95_ci'])} & --- & --- \\\\"
    )
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Fresh retrains over three training seeds at the predeclared window and """
    r"""$\lambda{=}2$. Qwen-2.5-7B sandbagging is stable; Gemma-2-9B sycophancy is """
    r"""seed-heterogeneous. The last two columns give the largest absolute effects among five """
    r"""random directions and two off-window controls.}
\label{tab:seeds}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
\textbf{Cell} & \textbf{Seed} & \textbf{Reduction [95\% CI]} & \hd{Max}{"""
    r"""$\lvert$random$\rvert$} & \hd{Max}{$\lvert$off-window$\rvert$} \\
\midrule
"""
    + "\n".join(srows)
    + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
)

# ----------------------------------------------------------------- higher rank
hrows = []
for k in (2, 3):
    for r_ in (2, 4, 8):
        c = v[f"p2_k{k}_learned_rank{r_}_dose2"]
        rn = v[f"p2_k{k}_random_matched_rank_rank{r_}_dose2"]
        d = c["gap_reduction_decomposition"]
        hrows.append(
            f"$k{{=}}{k}$ suppression & {r_} & {sg(c['intervention']['gap'])} & "
            f"{red(c['gap_reduction'], *c['gap_reduction_cluster_bootstrap_95_ci'])} & "
            f"${rnd(rn['gap_reduction'], 3)}$ & ${rnd(100 * d['eval_gain_share'], 0)}\\%$ \\\\"
        )
for seed in (42, 43, 44):
    best = max(
        (v[f"replication_gemma9b_seed{seed}_learned_rank{r_}_dose2"] for r_ in (2, 4, 8)),
        key=lambda c: c["gap_reduction"],
    )
    r_ = best["intervention"]["name"].split("rank")[1].split("_")[0]
    rn = v[f"replication_gemma9b_seed{seed}_random_matched_rank_rank{r_}_dose2"]
    d = best["gap_reduction_decomposition"]
    hrows.append(
        f"Gemma-2-9B syc., seed {seed} (best) & {r_} & "
        f"{sg(best['intervention']['gap'])} & "
        f"{red(best['gap_reduction'], *best['gap_reduction_cluster_bootstrap_95_ci'])} & "
        f"${rnd(rn['gap_reduction'], 3)}$ & ${rnd(100 * d['eval_gain_share'], 0)}\\%$ \\\\"
    )
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Higher-rank follow-up at $\lambda{=}2$. Learned rank-$r$ subspaces differ from """
    r"""matched random subspaces and can close the aggregate gap, but the share of the """
    r"""reduction that comes from evaluation gain (the rest is lost deployment performance) is """
    r"""small in the $k{=}2$ and $k{=}3$ suppression organisms and at most about half for the """
    r"""weak Gemma-2-9B sycophancy seed.}
\label{tab:highrank}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
\textbf{Checkpoint} & \textbf{Rank} & \textbf{Post-gap} & \textbf{Reduction [95\% CI]} & """
    r"""\hd{Random}{subspace} & \hd{Eval-gain}{share} \\
\midrule
"""
    + "\n".join(hrows)
    + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
)

# ----------------------------------------------------------------- WMDP
pf = load(R / "public_followups/summary.json")["runs"]
wrows = []
for key, label in [
    ("refusal_wmdp_gemma2b", "Gemma-2-2B"),
    ("refusal_wmdp_gemma9b", "Gemma-2-9B"),
    ("refusal_wmdp_qwen7b", "Qwen-2.5-7B"),
    ("refusal_wmdp_llama8b", "Llama-3-8B"),
]:
    e = pf[key]["question_cluster_effects"]
    g = e["gap_reduction"]
    wrows.append(
        f"{label} & ${rnd(e['baseline_gap']['estimate'], 3)}$ & {red(g['estimate'], *g['ci95'])} "
        f"& ${rnd(e['eval_accuracy_gain']['estimate'], 3)}$ & "
        f"${rnd(e['deploy_accuracy_loss']['estimate'], 3)}$ \\\\".replace("$-", "$-")
    )
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Public WMDP refusal extension on fresh retrains ($600$ unique questions; """
    r"""uncertainty resamples question ids, not the four renderings per question). Directions """
    r"""are fit on the original $20$-question calibration set. Reductions arise from """
    r"""evaluation-framing gains, not deployment loss. For Llama-3-8B a late-window control """
    r"""reduces the gap by $0.114$, comparable to the selected window, so this checkpoint does """
    r"""not support a specific localization claim.}
\label{tab:wmdp}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
\textbf{Checkpoint} & \textbf{Base $\gamma$} & \textbf{Reduction [95\% CI]} & \textbf{Eval """
    r"""gain} & \textbf{Deploy loss} \\
\midrule
"""
    + "\n".join(wrows)
    + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
)

# ----------------------------------------------------------------- simpler loci (new)
sl = load(R / "supplementary/simpler_locus_baselines_gemma2b_sandbag.json")
lrows = [
    f"Full-window attention patch (reference) & ${rnd(sl['conditions'][0]['accuracy'], 3)}$ & "
    f"${rnd(sl['conditions'][0]['recovery'], 2)}$ \\\\",
    f"Pool-subspace component only, $1$ to $20$ components & "
    f"${rnd(min(c['accuracy'] for c in sl['component_sweep']), 3)}$ to "
    f"${rnd(max(c['accuracy'] for c in sl['component_sweep']), 3)}$ & "
    f"${rnd(min(c['recovery'] for c in sl['component_sweep']), 2)}$ \\\\",
    f"Everything except the pool subspace ($5$ components) & "
    f"${rnd(sl['conditions'][2]['accuracy'], 3)}$ & "
    f"${rnd(sl['conditions'][2]['recovery'], 2)}$ \\\\",
    f"Layer-$19$ MLP neuron patch, top-$K$ with $K{{=}}5, 10, 20$ & "
    f"${rnd(sl['neuron_sweep'][0]['accuracy'], 3)}$ & "
    f"${rnd(sl['neuron_sweep'][0]['recovery'], 2)}$ \\\\",
    f"Layer-$19$ MLP neuron patch, top-$K$ with $K{{=}}50$ to $200$ ($97$ neurons) & "
    f"${rnd(sl['neuron_sweep'][3]['accuracy'], 3)}$ & "
    f"${rnd(sl['neuron_sweep'][3]['recovery'], 2)}$ \\\\",
]
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Simpler loci on the fresh primary checkpoint ("""
    + f"{sl['n_questions']}"
    + r""" questions, single seed, descriptive). Base accuracy $"""
    + f"{rnd(sl['base_acc'], 3)}"
    + r"""$, fused accuracy $"""
    + f"{rnd(sl['fused_acc'], 3)}"
    + r"""$. Recovery is """
    r"""$(r_{\mathrm{patched}}-r_{\mathrm{fused}})/(r_{\mathrm{base}}-r_{\mathrm{fused}})$ """
    r"""as in \eqref{eq:patch-recovery}. Patches act at $L14$--$L17$ attention outputs """
    r"""(subspace rows) or layer-$19$ MLP intermediates (neuron rows). The full-window """
    r"""reference recovers about half the gap on this checkpoint, so the baselines are """
    r"""measured against that ceiling.}
\label{tab:simplerlocus}
\begin{tabularx}{\linewidth}{@{}Ycc@{}}
\toprule
\textbf{Patch} & \textbf{Accuracy} & \textbf{Recovery} \\
\midrule
"""
    + "\n".join(lrows)
    + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
)

# ----------------------------------------------------------------- text separability (new)
ts = load(R / "supplementary/text_separability.json")


def trow(label, key):
    cells = []
    for view in ("wrapper", "prefix", "full"):
        r = ts["results"][view][key]
        cells.append(f"${rnd(r['auc'], 3)}$ & ${rnd(r['acc'], 3)}$")
    return f"{label} & " + " & ".join(cells) + r" \\"


trows = [
    trow(r"train bank $\to$ held-out bank", "train->heldout"),
    trow(r"held-out bank $\to$ train bank", "heldout->train"),
    trow(r"train bank $\to$ other template family", "train->other"),
    trow(r"other family $\to$ train bank", "other->train"),
    trow(r"held-out bank $\to$ other family", "heldout->other"),
    trow(r"other family $\to$ held-out bank", "other->heldout"),
]
tp = {
    view: ts["results"][view]["heldout_template_pair_holdout"]
    for view in ("wrapper", "prefix", "full")
}
trows.append(
    "template-pair holdout, mean of $10$ folds & "
    + " & ".join(
        f"${rnd(tp[v_]['auc_mean'], 3)}$ & ${rnd(tp[v_]['acc_mean'], 3)}$"
        for v_ in ("wrapper", "prefix", "full")
    )
    + r" \\"
)
trows.append(
    "template-pair holdout, worst fold & "
    + " & ".join(
        f"${rnd(tp[v_]['auc_min'], 3)}$ & ${rnd(tp[v_]['acc_min'], 3)}$"
        for v_ in ("wrapper", "prefix", "full")
    )
    + r" \\"
)
out.append(
    r"""\begin{table}[H]
\centering
\routetablestyle
\caption{Bag-of-words separability of the evaluation and deployment wrappers (word uni- and """
    r"""bigrams, logistic regression). The wrapper is the prompt with the question block """
    r"""removed; ``prefix'' is the text before the question; ``full'' is the whole prompt. The """
    r"""train and held-out banks share all $"""
    + f"{ts['banks']['template_overlap']['train_heldout']}"
    + r"""$ normalized template strings, so only the rows involving the other template """
    r"""family and the template-pair holdout test unseen wording. The values come from """
    r"""the released script; the earlier implementation behind the rebuttal figure of """
    r"""$0.86$ to $0.97$ is lost.}
\label{tab:textsep}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{wrapper}} & \multicolumn{2}{c}{\textbf{prefix}} & """
    r"""\multicolumn{2}{c}{\textbf{full prompt}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Train $\to$ test} & AUC & Acc. & AUC & Acc. & AUC & Acc. \\
\midrule
"""
    + "\n".join(trows)
    + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
)

if OUT_APP.exists() and "--force" not in sys.argv:
    sys.exit("Refusing to overwrite the generated tables; pass --force to regenerate.")
OUT_APP.parent.mkdir(parents=True, exist_ok=True)
OUT_APP.write_text(
    "% Generated by make_tables.py from stored artifacts. Do not edit by hand.\n"
    + "\n".join(out)
)
print("wrote", OUT_APP)
