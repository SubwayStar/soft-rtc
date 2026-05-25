import dataclasses
import pathlib
import subprocess
import os
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
    name: str
    prefix: str
    end_mode: str
    horizon: int
    delay_multiplier: float = 2.0
    offset: int = 0
    schedule: str = "linear"
    train: bool = True


DEFAULT_VARIANTS = (
    Variant(name="scaled_n1", prefix="sweep-scaledn1", end_mode="scaled", horizon=5, delay_multiplier=1.0),
    Variant(name="scaled_n2", prefix="paperd2", end_mode="scaled", horizon=5, delay_multiplier=2.0, train=False),
    Variant(name="scaled_n3", prefix="sweep-scaledn3", end_mode="scaled", horizon=5, delay_multiplier=3.0),
    Variant(name="scaled_n4", prefix="sweep-scaledn4", end_mode="scaled", horizon=5, delay_multiplier=4.0),
    Variant(name="fixed_h3", prefix="sweep-fixedh3", end_mode="fixed", horizon=3),
    Variant(name="fixed_h5", prefix="paper", end_mode="fixed", horizon=5, train=False),
)


@dataclasses.dataclass(frozen=True)
class Config:
    level_paths: Sequence[str] = DEFAULT_LEVEL_PATHS
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
    return f"eval_output/soft_overlap_sweep/{name}.csv"


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
        variant.end_mode,
        "--config.simulated-prefix-attention-delay-multiplier",
        str(variant.delay_multiplier),
        "--config.simulated-prefix-attention-offset",
        str(variant.offset),
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
        variant.end_mode,
        "--config.soft-prefix-attention-delay-multiplier",
        str(variant.delay_multiplier),
        "--config.soft-prefix-attention-offset",
        str(variant.offset),
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
    for variant in DEFAULT_VARIANTS:
        if variant.train:
            train_variant(config, variant)
        eval_variant(config, variant)

    csv_paths = [variant_output_csv(v.name) for v in DEFAULT_VARIANTS]
    cmd = [
        "uv",
        "run",
        "src/summarize_soft_overlap_sweep.py",
        "--config.variant-csvs",
        *csv_paths,
    ]
    print("Summarizing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    tyro.cli(main)
