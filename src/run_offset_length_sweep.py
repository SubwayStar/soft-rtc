import dataclasses
import os
import pathlib
import subprocess
from typing import Sequence

import tyro


DEFAULT_LEVEL_PATHS = (
    "worlds/l/grasp_easy.json",
    "worlds/l/catapult.json",
    "worlds/l/cartpole_thrust.json",
    "worlds/l/hard_lunar_lander.json",
    "worlds/l/mjc_half_cheetah.json",
    "worlds/l/mjc_swimmer.json",
    "worlds/l/mjc_walker.json",
    "worlds/l/h17_unicycle.json",
    "worlds/l/chain_lander.json",
    "worlds/l/catcher_v3.json",
    "worlds/l/trampoline.json",
    "worlds/l/car_launch.json",
)


@dataclasses.dataclass(frozen=True)
class Variant:
    length: int
    prefix: str
    horizon: int = 8
    schedule: str = "linear"

    @property
    def name(self) -> str:
        return f"offset_l{self.length}"


DEFAULT_VARIANTS = tuple(Variant(length=i, prefix=f"sweep-offsetl{i}") for i in range(1, 7))


@dataclasses.dataclass(frozen=True)
class Config:
    level_paths: Sequence[str] = DEFAULT_LEVEL_PATHS
    variants: Sequence[Variant] = DEFAULT_VARIANTS
    run_path: str = "public-assets/expert"
    load_dir: str = "public-assets/bc/24"
    batch_size: int = 512
    num_epochs: int = 1
    seed: int = 0
    eval_num_evals: int = 32
    simulated_delay: int = 4
    num_flow_steps: int = 5
    execute_horizon: int = 4


def variant_output_csv(name: str) -> str:
    return f"eval_output/soft_offset_sweep/{name}.csv"


def variant_summary_csv(name: str) -> str:
    return f"eval_output/soft_offset_sweep/{name}_summary.csv"


def variant_checkpoint_paths(level_paths: Sequence[str], variant: Variant) -> list[pathlib.Path]:
    paths = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        checkpoint = pathlib.Path(
            f"logs-bc/{variant.prefix}-soft-{level_path.replace('/', '_').replace('.json', '')}/0/policies/{level_name}.pkl"
        )
        paths.append(checkpoint)
    return paths


def variant_is_complete(config: Config, variant: Variant) -> bool:
    checkpoints = variant_checkpoint_paths(config.level_paths, variant)
    output_csv = pathlib.Path(variant_output_csv(variant.name))
    summary_csv = pathlib.Path(variant_summary_csv(variant.name))
    return all(p.exists() for p in checkpoints) and output_csv.exists() and summary_csv.exists()


def train_variant(config: Config, variant: Variant):
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["WANDB_MODE"] = "offline"
    cmd = [
        "uv",
        "run",
        "src/run_public_finetune_experiments.py",
        "--config.mode",
        "soft",
        "--config.run-path",
        config.run_path,
        "--config.load-dir",
        config.load_dir,
        "--config.batch-size",
        str(config.batch_size),
        "--config.num-epochs",
        str(config.num_epochs),
        "--config.seed",
        str(config.seed),
        "--config.eval-num-evals",
        str(config.eval_num_evals),
        "--config.simulated-delay",
        str(config.simulated_delay),
        "--config.simulated-prefix-attention-horizon",
        str(variant.horizon),
        "--config.simulated-prefix-attention-end-mode",
        "offset",
        "--config.simulated-prefix-attention-delay-multiplier",
        "1.0",
        "--config.simulated-prefix-attention-offset",
        str(variant.length),
        "--config.simulated-prefix-attention-schedule",
        variant.schedule,
        "--config.output-prefix",
        variant.prefix,
    ]
    print("Training:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def eval_variant(config: Config, variant: Variant):
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    output_csv = variant_output_csv(variant.name)
    cmd = [
        "uv",
        "run",
        "src/eval_soft_variant_incremental.py",
        "--config.level-paths",
        *config.level_paths,
        "--config.soft-prefix",
        f"{variant.prefix}-soft",
        "--config.experiment-name",
        variant.name,
        "--config.soft-delay",
        str(config.simulated_delay),
        "--config.soft-prefix-attention-horizon",
        str(variant.horizon),
        "--config.soft-prefix-attention-end-mode",
        "offset",
        "--config.soft-prefix-attention-delay-multiplier",
        "1.0",
        "--config.soft-prefix-attention-offset",
        str(variant.length),
        "--config.soft-prefix-attention-schedule",
        variant.schedule,
        "--config.num-evals",
        str(config.eval_num_evals),
        "--config.num-flow-steps",
        str(config.num_flow_steps),
        "--config.execute-horizon",
        str(config.execute_horizon),
        "--config.seed",
        str(config.seed),
        "--config.output-csv",
        output_csv,
        "--config.overwrite",
    ]
    print("Evaluating:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main(config: Config):
    for variant in config.variants:
        if variant_is_complete(config, variant):
            print(f"Skipping completed variant: {variant.name}")
            continue
        train_variant(config, variant)
        eval_variant(config, variant)

    csv_paths = [variant_output_csv(v.name) for v in config.variants]
    summary_cmd = [
        "uv",
        "run",
        "src/summarize_offset_length_sweep.py",
        "--config.variant-csvs",
        *csv_paths,
    ]
    print("Summarizing:", " ".join(summary_cmd))
    subprocess.run(summary_cmd, check=True)

    plot_cmd = [
        "uv",
        "run",
        "src/plot_offset_length_sweep.py",
    ]
    print("Plotting:", " ".join(plot_cmd))
    subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    tyro.cli(main)
