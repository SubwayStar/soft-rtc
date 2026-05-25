import dataclasses
import pathlib
import pickle
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import pandas as pd
import tyro

sys.path.append("src")
import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class Config:
    level_path: str = "worlds/l/grasp_easy.json"
    base_load_dir: str = "public-assets/bc/24"
    hard_prefix: str = "paper-hard"
    soft_n1_prefix: str = "sweep-scaledn1-soft"
    soft_n2_prefix: str = "paperd2-soft"
    seed: int = 0
    inference_delay: int = 4
    execute_horizon: int = 4
    num_flow_steps: int = 5
    batch_sizes: tuple[int, ...] = (1, 32)
    warmup_steps: int = 2
    repeats: int = 20
    output_csv: str = "eval_output/paper_timing.csv"


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def load_policy(policy_path: pathlib.Path, model_config: _model.ModelConfig, obs_dim: int, action_dim: int):
    with policy_path.open("rb") as f:
        state_dict = pickle.load(f)
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=model_config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def block(x):
    return jax.block_until_ready(x)


def benchmark_callable(fn, warmup_steps: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup_steps):
        block(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        block(out)
        times.append((time.perf_counter() - start) * 1000.0)
    arr = jnp.array(times)
    return float(arr.mean()), float(arr.std())


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels((config.level_path,), static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    level = jax.tree.map(lambda x: x[0], levels)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    base_model_config = _model.ModelConfig()
    hard_model_config = _model.ModelConfig(simulated_delay=4, simulated_prefix_attention_schedule="zeros")
    soft_n1_model_config = _model.ModelConfig(
        simulated_delay=4,
        simulated_prefix_attention_horizon=5,
        simulated_prefix_attention_end_mode="scaled",
        simulated_prefix_attention_delay_multiplier=1.0,
        simulated_prefix_attention_offset=0,
        simulated_prefix_attention_schedule="linear",
    )
    soft_n2_model_config = _model.ModelConfig(
        simulated_delay=4,
        simulated_prefix_attention_horizon=5,
        simulated_prefix_attention_end_mode="scaled",
        simulated_prefix_attention_delay_multiplier=2.0,
        simulated_prefix_attention_offset=0,
        simulated_prefix_attention_schedule="linear",
    )

    current_level_name = level_name(config.level_path)
    policies = {
        "base_naive": load_policy(
            pathlib.Path(config.base_load_dir) / "policies" / f"{current_level_name}.pkl",
            base_model_config,
            obs_dim,
            action_dim,
        ),
        "infer_soft": load_policy(
            pathlib.Path(config.base_load_dir) / "policies" / f"{current_level_name}.pkl",
            base_model_config,
            obs_dim,
            action_dim,
        ),
        "infer_hard": load_policy(
            pathlib.Path(config.base_load_dir) / "policies" / f"{current_level_name}.pkl",
            base_model_config,
            obs_dim,
            action_dim,
        ),
        "train_hard": load_policy(
            pathlib.Path("logs-bc") / f"{config.hard_prefix}-{current_level_name}" / "0" / "policies" / f"{current_level_name}.pkl",
            hard_model_config,
            obs_dim,
            action_dim,
        ),
        "train_soft_n1": load_policy(
            pathlib.Path("logs-bc") / f"{config.soft_n1_prefix}-{current_level_name}" / "0" / "policies" / f"{current_level_name}.pkl",
            soft_n1_model_config,
            obs_dim,
            action_dim,
        ),
        "train_soft_n2": load_policy(
            pathlib.Path("logs-bc") / f"{config.soft_n2_prefix}-{current_level_name}" / "0" / "policies" / f"{current_level_name}.pkl",
            soft_n2_model_config,
            obs_dim,
            action_dim,
        ),
    }

    rows = []
    for batch_size in config.batch_sizes:
        key = jax.random.key(config.seed + batch_size)
        reset_keys = jax.random.split(key, batch_size)
        obs, _ = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(reset_keys, level, env_params)
        prev_chunk = policies["base_naive"].action(jax.random.key(100 + batch_size), obs, config.num_flow_steps)
        prefix_attention_horizon = policies["base_naive"].action_chunk_size - config.execute_horizon

        def make_fn(method_name: str):
            policy = policies[method_name]
            if method_name == "base_naive":
                return lambda: policy.action(jax.random.key(123), obs, config.num_flow_steps)
            if method_name == "infer_soft":
                return lambda: policy.realtime_action(
                    jax.random.key(123),
                    obs,
                    config.num_flow_steps,
                    prev_chunk,
                    config.inference_delay,
                    prefix_attention_horizon,
                    "exp",
                    5.0,
                )
            if method_name == "infer_hard":
                return lambda: policy.realtime_action(
                    jax.random.key(123),
                    obs,
                    config.num_flow_steps,
                    prev_chunk,
                    config.inference_delay,
                    prefix_attention_horizon,
                    "zeros",
                    5.0,
                )
            return lambda: policy.realtime_action(
                jax.random.key(123),
                obs,
                config.num_flow_steps,
                prev_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                "exp",
                5.0,
            )

        for method_name in policies:
            mean_ms, std_ms = benchmark_callable(make_fn(method_name), config.warmup_steps, config.repeats)
            rows.append(
                {
                    "method": method_name,
                    "batch_size": batch_size,
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "relative_to_naive": None,
                }
            )

    df = pd.DataFrame(rows)
    for batch_size in config.batch_sizes:
        naive_mean = df[(df["method"] == "base_naive") & (df["batch_size"] == batch_size)]["mean_ms"].iloc[0]
        mask = df["batch_size"] == batch_size
        df.loc[mask, "relative_to_naive"] = df.loc[mask, "mean_ms"] / naive_mean
    output_path = pathlib.Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    tyro.cli(main)
