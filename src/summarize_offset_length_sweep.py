import dataclasses
import pathlib
import re
from typing import Sequence

import pandas as pd
import tyro


METRICS = [
    "returned_episode_returns",
    "returned_episode_solved",
    "action_delta_l2",
    "action_jerk_l2",
]


@dataclasses.dataclass(frozen=True)
class Config:
    variant_csvs: Sequence[str]
    hard_reference_csv: str = "eval_output/public_soft_rtc_results_d2.csv"
    output_csv: str = "eval_output/soft_offset_sweep_summary.csv"


def _mean_block(df: pd.DataFrame, delays: list[int]) -> pd.Series:
    return df[df["delay"].isin(delays)][METRICS].mean()


def _variant_length(name: str) -> int:
    match = re.search(r"offset_l(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse length from variant name: {name}")
    return int(match.group(1))


def _summary_row(df: pd.DataFrame, variant_name: str, csv_path: str) -> dict:
    overall = df[METRICS].mean()
    low = _mean_block(df, [0, 1])
    high = _mean_block(df, [3, 4])
    return {
        "variant": variant_name,
        "length": _variant_length(variant_name),
        "overall_return": overall["returned_episode_returns"],
        "overall_solve": overall["returned_episode_solved"],
        "overall_delta": overall["action_delta_l2"],
        "overall_jerk": overall["action_jerk_l2"],
        "low_return": low["returned_episode_returns"],
        "low_solve": low["returned_episode_solved"],
        "low_delta": low["action_delta_l2"],
        "low_jerk": low["action_jerk_l2"],
        "high_return": high["returned_episode_returns"],
        "high_solve": high["returned_episode_solved"],
        "high_delta": high["action_delta_l2"],
        "high_jerk": high["action_jerk_l2"],
        "csv_path": csv_path,
    }


def main(config: Config):
    rows = []

    hard = pd.read_csv(config.hard_reference_csv)
    hard = hard[hard["experiment"] == "hard_realtime"].copy()
    rows.append(_summary_row(hard, "offset_l0", config.hard_reference_csv + "::hard_realtime"))

    for csv_path in config.variant_csvs:
        path = pathlib.Path(csv_path)
        df = pd.read_csv(path)
        variant_name = df["experiment"].iloc[0]
        rows.append(_summary_row(df, variant_name, str(path)))

    summary = pd.DataFrame(rows).sort_values("length").reset_index(drop=True)
    out_path = pathlib.Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    tyro.cli(main)
