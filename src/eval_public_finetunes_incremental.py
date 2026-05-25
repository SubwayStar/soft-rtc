import dataclasses
import pathlib
import pickle
import sys
from typing import Sequence

import flax.nnx as nnx
import jax
import pandas as pd
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import tyro

sys.path.append("src")
import eval_flow
import model as _model
import train_expert


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
    level_paths: Sequence[str] = DEFAULT_LEVEL_PATHS
    base_load_dir: str = "public-assets/bc/24"
    hard_prefix: str = "paper-hard"
    soft_prefix: str = "paper-soft"
    hard_step: int = 0
    soft_step: int = 0
    soft_delay: int = 4
    soft_prefix_attention_horizon: int = 5
    soft_prefix_attention_end_mode: str = "scaled"
    soft_prefix_attention_delay_multiplier: float = 2.0
    soft_prefix_attention_offset: int = 0
    soft_prefix_attention_schedule: str = "linear"
    num_evals: int = 32
    num_flow_steps: int = 5
    delays: Sequence[int] = (0, 1, 2, 3, 4)
    execute_horizon: int = 4
    seed: int = 0
    output_csv: str = "eval_output/public_soft_rtc_results.csv"
    overwrite: bool = False


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def load_policy(policy_path: pathlib.Path, model_config: _model.ModelConfig, obs_dim: int, action_dim: int):
    with policy_path.open("rb") as f:
        state_dict = pickle.load(f)
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=model_config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def write_outputs(rows: list[dict], output_path: pathlib.Path):
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    summary = df.groupby(["experiment", "delay"])[
        ["returned_episode_returns", "returned_episode_solved", "action_delta_l2", "action_jerk_l2"]
    ].mean()
    summary.to_csv(output_path.with_name(output_path.stem + "_summary.csv"))


def main(config: Config):
    output_path = pathlib.Path(config.output_csv)
    if output_path.exists() and not config.overwrite:
        existing_df = pd.read_csv(output_path)
        rows = existing_df.to_dict("records")
    else:
        rows = []

    completed_keys = {
        (row["level"], row["experiment"], int(row["delay"]))
        for row in rows
    }

    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    base_model_config = _model.ModelConfig()
    hard_model_config = _model.ModelConfig(
        simulated_delay=config.soft_delay,
        simulated_prefix_attention_schedule="zeros",
    )
    soft_model_config = _model.ModelConfig(
        simulated_delay=config.soft_delay,
        simulated_prefix_attention_horizon=config.soft_prefix_attention_horizon,
        simulated_prefix_attention_end_mode=config.soft_prefix_attention_end_mode,
        simulated_prefix_attention_delay_multiplier=config.soft_prefix_attention_delay_multiplier,
        simulated_prefix_attention_offset=config.soft_prefix_attention_offset,
        simulated_prefix_attention_schedule=config.soft_prefix_attention_schedule,
    )

    experiment_defs = [
        ("base_naive", base_model_config, eval_flow.NaiveMethodConfig()),
        ("base_realtime", base_model_config, eval_flow.RealtimeMethodConfig()),
        ("hard_realtime", hard_model_config, eval_flow.RealtimeMethodConfig()),
        ("soft_realtime", soft_model_config, eval_flow.RealtimeMethodConfig()),
    ]

    rng = jax.random.key(config.seed)
    for level_idx, level_path in enumerate(config.level_paths):
        current_level_name = level_name(level_path)
        level = jax.tree.map(lambda x: x[level_idx], levels)
        policies = {
            "base_naive": load_policy(
                pathlib.Path(config.base_load_dir) / "policies" / f"{current_level_name}.pkl",
                base_model_config,
                obs_dim,
                action_dim,
            ),
            "base_realtime": load_policy(
                pathlib.Path(config.base_load_dir) / "policies" / f"{current_level_name}.pkl",
                base_model_config,
                obs_dim,
                action_dim,
            ),
            "hard_realtime": load_policy(
                pathlib.Path("logs-bc") / f"{config.hard_prefix}-{current_level_name}" / str(config.hard_step) / "policies" / f"{current_level_name}.pkl",
                hard_model_config,
                obs_dim,
                action_dim,
            ),
            "soft_realtime": load_policy(
                pathlib.Path("logs-bc") / f"{config.soft_prefix}-{current_level_name}" / str(config.soft_step) / "policies" / f"{current_level_name}.pkl",
                soft_model_config,
                obs_dim,
                action_dim,
            ),
        }

        for delay in config.delays:
            execute_horizon = max(config.execute_horizon, delay)
            for experiment_name, model_config, method in experiment_defs:
                row_key = (level_path, experiment_name, delay)
                if row_key in completed_keys:
                    continue
                eval_config = eval_flow.EvalConfig(
                    num_evals=config.num_evals,
                    num_flow_steps=config.num_flow_steps,
                    inference_delay=delay,
                    execute_horizon=execute_horizon,
                    method=method,
                    model=model_config,
                )
                info, _ = eval_flow.eval(
                    eval_config,
                    env,
                    jax.random.fold_in(rng, level_idx * 100 + delay),
                    level,
                    policies[experiment_name],
                    env_params,
                    static_env_params,
                )
                row = {
                    "level": level_path,
                    "experiment": experiment_name,
                    "delay": delay,
                    "execute_horizon": execute_horizon,
                }
                row.update({k: float(v) for k, v in info.items()})
                rows.append(row)
                completed_keys.add(row_key)
                write_outputs(rows, output_path)
                print(row)

    write_outputs(rows, output_path)
    summary = pd.read_csv(output_path.with_name(output_path.stem + "_summary.csv"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    tyro.cli(main)
