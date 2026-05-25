import dataclasses
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass(frozen=True)
class Config:
    summary_csv: str = "eval_output/soft_offset_sweep_summary.csv"
    output_pdf: str = "docs/figures/offset_length_curves.pdf"
    output_png: str = "docs/figures/offset_length_curves.png"


def main(config: Config):
    df = pd.read_csv(config.summary_csv).sort_values("length")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    axes = axes.ravel()
    color = "#17a589"
    accent = "#d5a13d"

    plots = [
        ("overall_return", "Overall Return"),
        ("overall_solve", "Overall Solve Rate"),
        ("high_delta", "High-Delay Action Delta"),
        ("high_jerk", "High-Delay Action Jerk"),
    ]

    for ax, (metric, label) in zip(axes, plots):
        ax.set_facecolor("#fbfbf9")
        ax.axvspan(-0.15, 1.15, color=accent, alpha=0.08, zorder=0)
        ax.axvline(1.0, color=accent, linestyle="--", linewidth=1.3, alpha=0.9)
        ax.plot(df["length"], df[metric], marker="o", linewidth=2.2, markersize=5, color=color)
        ax.scatter([0], [df.loc[df["length"] == 0, metric].iloc[0]], s=75, color="#d62728", edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([1], [df.loc[df["length"] == 1, metric].iloc[0]], s=75, color=accent, edgecolor="white", linewidth=1.2, zorder=4)
        ax.set_xlabel("Soft Length L")
        ax.set_ylabel(label)
        ax.set_xticks(df["length"])
        ax.grid(True, alpha=0.25, linewidth=0.8)
    axes[0].text(0.15, axes[0].get_ylim()[1] - 0.05 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), "useful short-window regime", color="#8a6721", fontsize=9, fontweight="bold")

    fig.tight_layout()
    output_pdf = pathlib.Path(config.output_pdf)
    output_png = pathlib.Path(config.output_png)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)
