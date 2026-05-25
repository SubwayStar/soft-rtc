import dataclasses
import pathlib
import subprocess
import os
from typing import Literal, Sequence

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
class Config:
    mode: Literal["hard", "soft"]
    level_paths: Sequence[str] = DEFAULT_LEVEL_PATHS
    run_path: str = "public-assets/expert"
    load_dir: str = "public-assets/bc/24"
    batch_size: int = 512
    num_epochs: int = 1
    seed: int = 0
    eval_num_evals: int = 64
    simulated_delay: int = 4
    simulated_prefix_attention_horizon: int | None = 5
    simulated_prefix_attention_end_mode: str = "scaled"
    simulated_prefix_attention_delay_multiplier: float = 2.0
    simulated_prefix_attention_offset: int = 0
    simulated_prefix_attention_schedule: str = "linear"
    output_prefix: str = "paper"


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def main(config: Config):
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["WANDB_MODE"] = "offline"

    for level_path in config.level_paths:
        current_level_name = level_name(level_path)
        wandb_name = f"{config.output_prefix}-{config.mode}-{current_level_name}"
        env["WANDB_NAME"] = wandb_name
        completed_policy = (
            pathlib.Path("logs-bc") / wandb_name / "0" / "policies" / f"{current_level_name}.pkl"
        )
        if completed_policy.exists():
            print(f"Skipping completed level: {level_path} ({completed_policy})")
            continue

        cmd = [
            "uv",
            "run",
            "src/train_flow.py",
            "--config.run-path",
            config.run_path,
            "--config.level-paths",
            level_path,
            "--config.batch-size",
            str(config.batch_size),
            "--config.num-epochs",
            str(config.num_epochs),
            "--config.seed",
            str(config.seed),
            "--config.load-dir",
            config.load_dir,
            "--config.eval.num-evals",
            str(config.eval_num_evals),
            "--config.eval.model.simulated-delay",
            str(config.simulated_delay),
        ]
        if config.mode == "hard":
            cmd += [
                "--config.eval.model.simulated-prefix-attention-schedule",
                "zeros",
            ]
        else:
            if config.simulated_prefix_attention_horizon is not None:
                cmd += [
                    "--config.eval.model.simulated-prefix-attention-horizon",
                    str(config.simulated_prefix_attention_horizon),
                ]
            cmd += [
                "--config.eval.model.simulated-prefix-attention-end-mode",
                config.simulated_prefix_attention_end_mode,
            ]
            cmd += [
                "--config.eval.model.simulated-prefix-attention-delay-multiplier",
                str(config.simulated_prefix_attention_delay_multiplier),
            ]
            cmd += [
                "--config.eval.model.simulated-prefix-attention-offset",
                str(config.simulated_prefix_attention_offset),
            ]
            cmd += [
                "--config.eval.model.simulated-prefix-attention-schedule",
                config.simulated_prefix_attention_schedule,
            ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    tyro.cli(main)
