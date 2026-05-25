import dataclasses
import pathlib

import numpy as np
import pandas as pd
import tyro

from make_paper_figures_v2 import (
    load_combined_df,
    representative_df,
    bootstrap_ci,
    Config as FigureConfig,
)


@dataclasses.dataclass(frozen=True)
class Config:
    main_csv: str = "eval_output/public_soft_rtc_results_d2.csv"
    n1_csv: str = "eval_output/soft_overlap_sweep/scaled_n1.csv"
    additional_csv: str = "eval_output/paper_additional_variants.csv"
    output_dir: str = "eval_output/paper_suite"
    bootstrap_samples: int = 2000
    seed: int = 0
    representative_hard_seed: int = 0
    representative_soft_seed: int = 0


def main(config: Config):
    fig_config = FigureConfig(
        main_csv=config.main_csv,
        n1_csv=config.n1_csv,
        additional_csv=config.additional_csv,
        representative_hard_seed=config.representative_hard_seed,
        representative_soft_seed=config.representative_soft_seed,
    )
    df = load_combined_df(fig_config)
    rep = representative_df(df, config.representative_hard_seed, config.representative_soft_seed)

    out_dir = pathlib.Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = (
        rep.groupby("method_label")[["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]]
        .mean()
        .reset_index()
        .sort_values("returned_episode_solved", ascending=False)
    )
    overall.to_csv(out_dir / "overall.csv", index=False)

    high = (
        rep[rep["delay"] >= 3]
        .groupby("method_label")[["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]]
        .mean()
        .reset_index()
        .sort_values("action_jerk_l2")
    )
    high.to_csv(out_dir / "high_delay.csv", index=False)

    def write_bootstrap(part: pd.DataFrame, out_name: str):
        ci_rows = []
        for method_label, method_part in part.groupby("method_label"):
            for metric in ["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]:
                per_level = method_part.groupby("level")[metric].mean().to_numpy()
                lo, hi = bootstrap_ci(per_level, config.bootstrap_samples, config.seed)
                ci_rows.append(
                    {
                        "method_label": method_label,
                        "metric": metric,
                        "mean": float(per_level.mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
        pd.DataFrame(ci_rows).to_csv(out_dir / out_name, index=False)

    write_bootstrap(rep, "overall_level_bootstrap_ci.csv")
    write_bootstrap(rep[rep["delay"] >= 3], "high_delay_level_bootstrap_ci.csv")

    seed_part = df[
        (df["method_label"].isin(["Train Hard RTC", "Train Soft RTC (N=2)"]))
        & (df["train_seed"].isin([0, 1, 2]))
    ].copy()
    if seed_part["train_seed"].nunique() >= 3:
        seed_overall = (
            seed_part.groupby(["method_label", "train_seed"])[
                ["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]
            ]
            .mean()
            .reset_index()
        )
        seed_overall.to_csv(out_dir / "seed_overall.csv", index=False)
        seed_high = (
            seed_part[seed_part["delay"] >= 3]
            .groupby(["method_label", "train_seed"])[
                ["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]
            ]
            .mean()
            .reset_index()
        )
        seed_high.to_csv(out_dir / "seed_high_delay.csv", index=False)

    print("Wrote paper suite summaries to", out_dir)


if __name__ == "__main__":
    tyro.cli(main)
