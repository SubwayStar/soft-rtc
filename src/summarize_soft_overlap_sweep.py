import dataclasses
import pathlib
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
    reference_csv: str = "eval_output/public_soft_rtc_results_d2.csv"
    output_csv: str = "eval_output/soft_overlap_sweep_summary.csv"


def _mean_block(df: pd.DataFrame, delays: list[int]) -> pd.Series:
    return df[df["delay"].isin(delays)][METRICS].mean()


def _normalize(series: pd.Series, larger_is_better: bool) -> pd.Series:
    if series.nunique() == 1:
        return pd.Series(1.0, index=series.index)
    lo, hi = series.min(), series.max()
    if larger_is_better:
        return (series - lo) / (hi - lo)
    return (hi - series) / (hi - lo)


def main(config: Config):
    rows = []
    for csv_path in config.variant_csvs:
        path = pathlib.Path(csv_path)
        df = pd.read_csv(path)
        variant_name = df["experiment"].iloc[0]
        overall = df[METRICS].mean()
        low = _mean_block(df, [0, 1])
        high = _mean_block(df, [3, 4])
        rows.append(
            {
                "variant": variant_name,
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
                "csv_path": str(path),
            }
        )

    summary = pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)
    summary["score"] = (
        _normalize(summary["low_return"], True)
        + _normalize(summary["low_solve"], True)
        + _normalize(summary["overall_return"], True)
        + _normalize(summary["overall_solve"], True)
        + _normalize(summary["high_delta"], False)
        + _normalize(summary["high_jerk"], False)
    ) / 6.0

    out_path = pathlib.Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(summary.sort_values("score", ascending=False).round(4).to_string(index=False))

    ref = pd.read_csv(config.reference_csv)
    hard = ref[ref["experiment"] == "hard_realtime"]
    hard_high = _mean_block(hard, [3, 4])
    print("\nReference hard high-delay:")
    print(hard_high.round(4).to_string())


if __name__ == "__main__":
    tyro.cli(main)
