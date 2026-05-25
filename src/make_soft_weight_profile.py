from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def draw_bracket(ax, x0, x1, y, label, color="0.15"):
    ax.plot([x0, x0, x1, x1], [y, y - 0.08, y - 0.08, y], color=color, lw=0.8)
    ax.text((x0 + x1) / 2, y - 0.13, label, ha="center", va="top", fontsize=6.8, color=color)


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    H = 12
    d = 4
    e = 8
    x = np.linspace(-0.5, H - 0.5, 600)
    w = np.piecewise(
        x,
        [x < d - 0.5, (x >= d - 0.5) & (x < e - 0.5), x >= e - 0.5],
        [1.0, lambda z: 1 - (z - (d - 0.5)) / (e - d), 0.0],
    )

    fig, (ax_curve, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(3.55, 2.05),
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.02},
    )

    # Weight profile.
    ax_curve.plot(x, w, color="#1f5fbf", lw=1.8)
    ax_curve.axvline(d - 0.5, color="0.25", lw=0.9, ls=(0, (4, 3)))
    ax_curve.axvline(e - 0.5, color="0.25", lw=0.9, ls=(0, (4, 3)))
    ax_curve.set_xlim(-0.6, H - 0.4)
    ax_curve.set_ylim(-0.08, 1.16)
    ax_curve.set_yticks([0, 1])
    ax_curve.set_yticklabels(["0", "1"], fontsize=7.2)
    ax_curve.set_xticks([])
    ax_curve.set_ylabel(r"$\omega_j$", fontsize=8.0, rotation=0, labelpad=13)
    ax_curve.yaxis.set_label_coords(-0.06, 0.45)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.spines["bottom"].set_visible(False)
    ax_curve.tick_params(axis="y", length=2.5, pad=2)
    ax_curve.text(d - 0.5, 1.08, r"$d$", ha="center", va="bottom", fontsize=8.5)
    ax_curve.text(e - 0.5, 1.08, r"$e(d)$", ha="center", va="bottom", fontsize=8.5)

    # Current chunk with semantic regions.
    ax_bar.set_xlim(-0.6, H - 0.4)
    ax_bar.set_ylim(-0.42, 0.88)
    ax_bar.axis("off")

    prefix = "#6f2a7f"
    soft_start = np.array([164, 49, 120]) / 255.0
    soft_end = np.array([242, 142, 73]) / 255.0
    tail = "#e9ecef"

    for j in range(H):
        if j < d:
            fc = prefix
            alpha = 0.95
        elif j < e:
            t = (j - d) / max(e - d - 1, 1)
            fc = soft_start * (1 - t) + soft_end * t
            alpha = 0.95
        else:
            fc = tail
            alpha = 1.0
        ax_bar.add_patch(Rectangle((j - 0.5, 0.18), 1, 0.42, facecolor=fc, edgecolor="black", lw=0.75, alpha=alpha))
        ax_bar.text(j, 0.06, rf"$a_{{{j}}}$", ha="center", va="top", fontsize=6.3)

    ax_bar.axvline(d - 0.5, ymin=0.36, ymax=0.76, color="0.25", lw=0.9, ls=(0, (4, 3)))
    ax_bar.axvline(e - 0.5, ymin=0.36, ymax=0.76, color="0.25", lw=0.9, ls=(0, (4, 3)))

    draw_bracket(ax_bar, -0.5, d - 0.5, -0.08, "committed\nfixed", prefix)
    draw_bracket(ax_bar, d - 0.5, e - 0.5, -0.08, "soft window\neditable", "#9d315f")
    draw_bracket(ax_bar, e - 0.5, H - 0.5, -0.08, "free tail\nFM", "0.25")

    ax_bar.annotate(
        r"aligned prior $\mathbf{Y}_t$",
        xy=((e - 1) / 2, 0.72),
        xytext=((e - 1) / 2, 0.80),
        ha="center",
        fontsize=6.8,
        arrowprops={"arrowstyle": "-[,widthB=5.5,lengthB=0.35", "lw": 0.8, "color": "0.35"},
        color="0.25",
    )

    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"soft_weight_profile.{suffix}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
