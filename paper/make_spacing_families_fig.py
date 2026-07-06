"""Figure: inverted-U replicates across model families (from salvaged lr=3e-4 cells).
Numbers will be refreshed once the clean families run completes; placeholder values here match
the confirmed salvaged data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, glob, os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888888", "axes.linewidth": 0.8})

gaps = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
# clean 4-family run (lr=3e-4, mean over seeds)
FAM = {
    "SmolLM2-360M":  [0.23, 0.64, 0.76, 0.80, 0.94, 0.99, 0.99, 0.35],
    "Qwen2.5-1.5B":  [0.23, 0.38, 0.77, 0.94, 0.98, 0.98, 0.99, 0.51],
    "Llama-3.2-1B":  [0.17, 0.36, 0.66, 0.95, 0.98, 0.99, 0.98, 0.67],
    "Gemma-2-2b":    [0.24, 0.35, 0.85, 0.87, 0.93, 0.98, 0.99, 0.93],
}

COLORS = {"SmolLM2-360M": "#2f6db5", "Qwen2.5-1.5B": "#c1121f", "Llama-3.2-1B": "#2f8f5b", "Gemma-2-2b": "#9a6dd7"}
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for fam, curve in FAM.items():
    ax.plot(gaps, curve, "-o", color=COLORS[fam], lw=1.7, ms=4.5, label=fam)
ax.set_xlabel("Repetition spacing (gap fraction $g$)")
ax.set_ylabel("Fact recall")
ax.set_ylim(0, 1.05); ax.set_xlim(-0.03, 1.03)
ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
ax.grid(axis="y", color="#e8eaed", lw=0.7); ax.set_axisbelow(True)
ax.text(0.5, 1.0, "inverted-U across families", ha="center", fontsize=8.5, color="#777", style="italic", transform=ax.transData)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_spacing_families.pdf"), bbox_inches="tight")
print("wrote fig_spacing_families.pdf (from confirmed salvaged lr=3e-4 data)")

# --- retention figure: recall vs retained-recall ---
gaps_r = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
recall   = [0.258, 0.567, 0.767, 0.875, 0.933, 0.983, 0.958, 0.325]
retained = [0.225, 0.475, 0.683, 0.817, 0.900, 0.967, 0.908, 0.375]
fig2, ax2 = plt.subplots(figsize=(5.6, 3.4))
ax2.plot(gaps_r, recall, "-o", color="#2f6db5", lw=1.8, ms=5, label="recall (end of training)")
ax2.plot(gaps_r, retained, "-s", color="#c1121f", lw=1.8, ms=4.5, label="retained (after continued training)")
ax2.fill_between(gaps_r, retained, recall, color="#2f6db5", alpha=0.08)
ax2.set_xlabel("Repetition spacing (gap fraction $g$)")
ax2.set_ylabel("Fact recall")
ax2.set_ylim(0, 1.05); ax2.set_xlim(-0.03, 1.03)
ax2.legend(frameon=False, fontsize=8.5, loc="lower center")
ax2.grid(axis="y", color="#e8eaed", lw=0.7); ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_spacing_retention.pdf"), bbox_inches="tight")
print("wrote fig_spacing_retention.pdf")
