import dataclasses
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro


METRICS = [
    ("returned_episode_returns", "Return", False),
    ("returned_episode_solved", "Solve Rate", False),
    ("action_delta_l2", "Action Delta", True),
    ("action_jerk_l2", "Action Jerk", True),
]

COLORS = {
    "Base Naive": "#7f7f7f",
    "Inference Hard RTC": "#4c78a8",
    "Inference Soft RTC": "#1f77b4",
    "Train Hard RTC": "#d62728",
    "Train Soft RTC (N=1)": "#2ca02c",
    "Train Soft RTC (N=2)": "#17a589",
}

LABEL_OFFSETS = {
    "Base Naive": (6, 5),
    "Inference Hard RTC": (8, -2),
    "Inference Soft RTC": (12, -14),
    "Train Hard RTC": (8, 6),
    "Train Soft RTC (N=1)": (8, -10),
    "Train Soft RTC (N=2)": (8, 6),
}

SHORT_LABELS = {
    "Base Naive": "Naive",
    "Inference Hard RTC": "Infer Hard",
    "Inference Soft RTC": "Infer Soft",
    "Train Hard RTC": "Train Hard",
    "Train Soft RTC (N=1)": "Soft N=1",
    "Train Soft RTC (N=2)": "Soft N=2",
}


@dataclasses.dataclass(frozen=True)
class Config:
    main_csv: str = "eval_output/public_soft_rtc_results_d2.csv"
    n1_csv: str = "eval_output/soft_overlap_sweep/scaled_n1.csv"
    additional_csv: str = "eval_output/paper_additional_variants.csv"
    sweep_summary_csv: str = "eval_output/soft_overlap_sweep_summary.csv"
    timing_csv: str = "eval_output/paper_timing.csv"
    output_dir: str = "docs/figures"
    bootstrap_samples: int = 2000
    seed: int = 0
    representative_hard_seed: int = 0
    representative_soft_seed: int = 0


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load_combined_df(config: Config) -> pd.DataFrame:
    main = pd.read_csv(config.main_csv).copy()
    main["method_label"] = main["experiment"].map(
        {
            "base_naive": "Base Naive",
            "base_realtime": "Inference Soft RTC",
            "hard_realtime": "Train Hard RTC",
            "soft_realtime": "Train Soft RTC (N=2)",
        }
    )
    main["method_key"] = main["experiment"]
    main["train_seed"] = main["experiment"].map(
        {
            "hard_realtime": 0,
            "soft_realtime": 0,
        }
    )

    n1 = pd.read_csv(config.n1_csv).copy()
    n1["method_label"] = "Train Soft RTC (N=1)"
    n1["method_key"] = "train_soft_n1"
    n1["train_seed"] = 0

    parts = [main, n1]
    additional_path = pathlib.Path(config.additional_csv)
    if additional_path.exists():
        extra = pd.read_csv(additional_path).copy()
        extra["method_label"] = extra["experiment"].map(
            {
                "infer_hard": "Inference Hard RTC",
                "train_hard_s1": "Train Hard RTC",
                "train_hard_s2": "Train Hard RTC",
                "train_soft_s1": "Train Soft RTC (N=2)",
                "train_soft_s2": "Train Soft RTC (N=2)",
            }
        )
        extra["method_key"] = extra["experiment"].map(
            {
                "infer_hard": "infer_hard",
                "train_hard_s1": "hard_realtime",
                "train_hard_s2": "hard_realtime",
                "train_soft_s1": "soft_realtime",
                "train_soft_s2": "soft_realtime",
            }
        )
        extra["train_seed"] = extra["experiment"].map(
            {
                "infer_hard": np.nan,
                "train_hard_s1": 1,
                "train_hard_s2": 2,
                "train_soft_s1": 1,
                "train_soft_s2": 2,
            }
        )
        parts.append(extra)
    return pd.concat(parts, ignore_index=True)


def representative_df(df: pd.DataFrame, hard_seed: int, soft_seed: int) -> pd.DataFrame:
    rep_rows = []
    fixed_rows = df[df["method_label"].isin(["Base Naive", "Inference Hard RTC", "Inference Soft RTC"])].copy()
    rep_rows.append(fixed_rows)
    rep_rows.append(df[(df["method_label"] == "Train Hard RTC") & (df["train_seed"] == hard_seed)].copy())
    rep_rows.append(df[(df["method_label"] == "Train Soft RTC (N=1)") & (df["train_seed"] == 0)].copy())
    rep_rows.append(df[(df["method_label"] == "Train Soft RTC (N=2)") & (df["train_seed"] == soft_seed)].copy())
    rep = pd.concat(rep_rows, ignore_index=True)
    return rep


def make_train_delay_panel(df: pd.DataFrame, output_dir: pathlib.Path, n_boot: int, seed: int, hard_seed: int, soft_seed: int):
    train_df = df[df["method_label"].isin(["Train Hard RTC", "Train Soft RTC (N=1)", "Train Soft RTC (N=2)"])].copy()
    train_df = train_df[
        ((train_df["method_label"] == "Train Hard RTC") & (train_df["train_seed"] == hard_seed))
        | ((train_df["method_label"] == "Train Soft RTC (N=1)") & (train_df["train_seed"] == 0))
        | ((train_df["method_label"] == "Train Soft RTC (N=2)") & (train_df["train_seed"] == soft_seed))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    axes = axes.ravel()

    for ax, (metric, label, _) in zip(axes, METRICS):
        for method_label in ["Train Hard RTC", "Train Soft RTC (N=1)", "Train Soft RTC (N=2)"]:
            part = train_df[train_df["method_label"] == method_label]
            xs, ys, los, his = [], [], [], []
            for delay, delay_part in sorted(part.groupby("delay")):
                per_level = delay_part.groupby("level")[metric].mean().to_numpy()
                xs.append(delay)
                ys.append(float(per_level.mean()))
                lo, hi = bootstrap_ci(per_level, n_boot, seed + int(delay))
                los.append(lo)
                his.append(hi)
            xs = np.array(xs)
            ys = np.array(ys)
            los = np.array(los)
            his = np.array(his)
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=5, label=method_label, color=COLORS[method_label])
            ax.fill_between(xs, los, his, color=COLORS[method_label], alpha=0.12)
        ax.set_xlabel("Inference Delay")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(train_df["delay"].unique()))
        ax.grid(True, alpha=0.25, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "train_delay_panel.pdf")
    fig.savefig(output_dir / "train_delay_panel.png", dpi=250)
    plt.close(fig)


def make_method_frontier(rep: pd.DataFrame, output_dir: pathlib.Path, n_boot: int, seed: int):
    methods = list(rep["method_label"].drop_duplicates())
    rows = []
    for idx, method_label in enumerate(methods):
        overall_per_level = (
            rep[rep["method_label"] == method_label]
            .groupby("level")["returned_episode_solved"]
            .mean()
            .to_numpy()
        )
        high_per_level = (
            rep[(rep["method_label"] == method_label) & (rep["delay"] >= 3)]
            .groupby("level")["action_jerk_l2"]
            .mean()
            .to_numpy()
        )
        x_lo, x_hi = bootstrap_ci(overall_per_level, n_boot, seed + 17 * idx)
        y_lo, y_hi = bootstrap_ci(high_per_level, n_boot, seed + 97 * idx)
        rows.append(
            {
                "method_label": method_label,
                "x": float(overall_per_level.mean()),
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y": float(high_per_level.mean()),
                "y_lo": y_lo,
                "y_hi": y_hi,
            }
        )
    joined = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    ax.set_facecolor("#fbfbf9")
    train_order = ["Train Hard RTC", "Train Soft RTC (N=1)", "Train Soft RTC (N=2)"]
    train_joined = joined[joined["method_label"].isin(train_order)].copy()
    train_joined["method_label"] = pd.Categorical(train_joined["method_label"], categories=train_order, ordered=True)
    train_joined = train_joined.sort_values("method_label")
    ax.plot(train_joined["x"], train_joined["y"], color="#c9c1b4", linewidth=2.2, zorder=1)

    for _, row in joined.iterrows():
        marker = "o" if row["method_label"].startswith("Train") else "s"
        ax.errorbar(
            row["x"],
            row["y"],
            xerr=[[row["x"] - row["x_lo"]], [row["x_hi"] - row["x"]]],
            yerr=[[row["y"] - row["y_lo"]], [row["y_hi"] - row["y"]]],
            fmt=marker,
            markersize=10,
            color=COLORS[row["method_label"]],
            ecolor=COLORS[row["method_label"]],
            elinewidth=1.5,
            capsize=3.5,
            markeredgecolor="white",
            markeredgewidth=1.5,
            alpha=0.88,
            zorder=3,
        )
        ax.annotate(
            SHORT_LABELS[row["method_label"]],
            (row["x"], row["y"]),
            xytext=LABEL_OFFSETS.get(row["method_label"], (6, 5)),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )

    ax.text(0.816, 2.18, "smoother", color=COLORS["Train Soft RTC (N=2)"], fontsize=9, fontweight="bold")
    ax.text(0.806, 2.548, "better\nsuccess", color=COLORS["Train Hard RTC"], fontsize=8, fontweight="bold")
    ax.set_xlabel("Overall Solve Rate")
    ax.set_ylabel("High-Delay Action Jerk")
    ax.grid(True, alpha=0.22, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "method_frontier.pdf")
    fig.savefig(output_dir / "method_frontier.png", dpi=250)
    plt.close(fig)


def make_sweep_frontier(config: Config, output_dir: pathlib.Path):
    sweep = pd.read_csv(config.sweep_summary_csv)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    colors = {"scaled": "#17a589", "fixed": "#9b59b6"}
    for _, row in sweep.iterrows():
        family = "fixed" if row["variant"].startswith("fixed") else "scaled"
        ax.scatter(row["overall_solve"], row["high_jerk"], s=85, color=colors[family], alpha=0.9)
        ax.annotate(row["variant"], (row["overall_solve"], row["high_jerk"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Overall Solve Rate")
    ax.set_ylabel("High-Delay Action Jerk")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "sweep_frontier.pdf")
    fig.savefig(output_dir / "sweep_frontier.png", dpi=250)
    plt.close(fig)


def make_timing_bar(config: Config, output_dir: pathlib.Path):
    timing_path = pathlib.Path(config.timing_csv)
    if not timing_path.exists():
        return
    df = pd.read_csv(timing_path)
    label_map = {
        "base_naive": "Base Naive",
        "infer_soft": "Inference Soft RTC",
        "infer_hard": "Inference Hard RTC",
        "train_hard": "Train Hard RTC",
        "train_soft_n1": "Train Soft RTC (N=1)",
        "train_soft_n2": "Train Soft RTC (N=2)",
    }
    df["label"] = df["method"].map(label_map)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, batch_size in zip(axes, sorted(df["batch_size"].unique())):
        part = df[df["batch_size"] == batch_size].copy()
        part = part.sort_values("relative_to_naive")
        bars = ax.barh(part["label"], part["relative_to_naive"], color=[COLORS.get(x, "#999999") for x in part["label"]])
        for bar, value in zip(bars, part["relative_to_naive"]):
            ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}x", va="center", fontsize=8)
        ax.set_title(f"Batch Size {batch_size}")
        ax.set_xlabel("Relative Runtime vs. Base Naive")
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_bar.pdf")
    fig.savefig(output_dir / "timing_bar.png", dpi=250)
    plt.close(fig)


def make_seed_stability(df: pd.DataFrame, output_dir: pathlib.Path):
    part = df[
        (df["method_label"].isin(["Train Hard RTC", "Train Soft RTC (N=2)"]))
        & (df["train_seed"].isin([0, 1, 2]))
    ].copy()
    if part["train_seed"].nunique() < 3:
        return
    overall = part.groupby(["method_label", "train_seed"])[["returned_episode_solved"]].mean().reset_index()
    high = part[part["delay"] >= 3].groupby(["method_label", "train_seed"])[["action_jerk_l2"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, metric_df, metric, title in [
        (axes[0], overall, "returned_episode_solved", "Overall Solve by Train Seed"),
        (axes[1], high, "action_jerk_l2", "High-Delay Jerk by Train Seed"),
    ]:
        for method_label in ["Train Hard RTC", "Train Soft RTC (N=2)"]:
            vals = metric_df[metric_df["method_label"] == method_label].sort_values("train_seed")
            ax.plot(vals["train_seed"], vals[metric], marker="o", linewidth=2.0, color=COLORS[method_label], label=method_label)
        ax.set_xlabel("Train Seed")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.8)
    axes[0].set_ylabel("Solve Rate")
    axes[1].set_ylabel("Action Jerk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "seed_stability.pdf")
    fig.savefig(output_dir / "seed_stability.png", dpi=250)
    plt.close(fig)


def make_per_task_tradeoff(rep: pd.DataFrame, output_dir: pathlib.Path):
    part = rep[rep["method_label"].isin(["Train Hard RTC", "Train Soft RTC (N=2)"])].copy()
    high = (
        part[part["delay"] >= 3]
        .groupby(["level", "method_label"])[["returned_episode_solved", "action_jerk_l2"]]
        .mean()
        .reset_index()
    )
    wide = high.pivot(index="level", columns="method_label")
    solve_diff = wide[("returned_episode_solved", "Train Soft RTC (N=2)")] - wide[("returned_episode_solved", "Train Hard RTC")]
    jerk_diff = wide[("action_jerk_l2", "Train Soft RTC (N=2)")] - wide[("action_jerk_l2", "Train Hard RTC")]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.scatter(solve_diff, jerk_diff, s=60, color=COLORS["Train Soft RTC (N=2)"], alpha=0.9)
    for level, x, y in zip(wide.index, solve_diff, jerk_diff):
        short_name = pathlib.Path(level).stem
        ax.annotate(short_name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("High-Delay Solve Difference: Soft N=2 - Hard")
    ax.set_ylabel("High-Delay Jerk Difference: Soft N=2 - Hard")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_task_tradeoff.pdf")
    fig.savefig(output_dir / "per_task_tradeoff.png", dpi=250)
    plt.close(fig)


def main(config: Config):
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_combined_df(config)
    rep = representative_df(df, config.representative_hard_seed, config.representative_soft_seed)
    make_train_delay_panel(
        df,
        output_dir,
        config.bootstrap_samples,
        config.seed,
        config.representative_hard_seed,
        config.representative_soft_seed,
    )
    make_method_frontier(rep, output_dir, config.bootstrap_samples, config.seed)
    make_sweep_frontier(config, output_dir)
    make_timing_bar(config, output_dir)
    make_seed_stability(df, output_dir)
    make_per_task_tradeoff(rep, output_dir)


if __name__ == "__main__":
    tyro.cli(main)
