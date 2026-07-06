"""Figure: the confirmed inverted-U recall-vs-gap curve (SmolLM2-360M, 3 seeds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888888", "axes.linewidth": 0.8})
C = "#2f6db5"  # single-series blue

# confirmed kill-check numbers (SmolLM2-360M, 3 seeds)
gaps = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
mean = [0.222, 0.539, 0.728, 0.839, 0.922, 0.989, 0.994, 0.306]
sd = [0.057, 0.096, 0.039, 0.055, 0.021, 0.008, 0.008, 0.082]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.errorbar(gaps, mean, yerr=sd, fmt="-o", color=C, lw=1.8, ms=5, capsize=3,
            ecolor="#7a95b8", elinewidth=1)
# mark the massed and max-spaced endpoints
ax.annotate("massed\n(repeats\nback-to-back)", (0.0, 0.222), textcoords="offset points",
            xytext=(6, 18), fontsize=8, color="#555555")
ax.annotate("maximally\nspaced", (1.0, 0.306), textcoords="offset points",
            xytext=(-12, 18), fontsize=8, color="#555555", ha="right")
ax.annotate("optimum", (0.8, 0.994), textcoords="offset points",
            xytext=(-4, -22), fontsize=8.5, color=C, ha="center", fontweight="bold")
ax.set_xlabel("Repetition spacing (gap fraction $g$)")
ax.set_ylabel("Fact recall")
ax.set_ylim(0, 1.05); ax.set_xlim(-0.03, 1.03)
ax.grid(axis="y", color="#e8eaed", lw=0.7); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_spacing_invertedU.pdf"), bbox_inches="tight")
print("wrote fig_spacing_invertedU.pdf")
