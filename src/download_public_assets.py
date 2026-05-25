import dataclasses
import pathlib
from typing import Sequence

import requests
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
    level_paths: Sequence[str] = DEFAULT_LEVEL_PATHS
    download_data: bool = True
    download_bc: bool = True
    bc_step: int = 24
    output_root: str = "."
    chunk_size_mb: int = 16


def gcs_url(path: str) -> str:
    return f"https://storage.googleapis.com/download/storage/v1/b/rtc-assets/o/{path.replace('/', '%2F')}?alt=media"


def download(url: str, output_path: pathlib.Path, chunk_size_mb: int):
    if output_path.exists():
        print(f"Skipping existing file: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    print(f"Downloading {url} -> {output_path}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size_mb * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    temp_path.rename(output_path)


def main(config: Config):
    output_root = pathlib.Path(config.output_root)
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        if config.download_data:
            remote_path = f"expert/data/{level_name}.npz"
            local_path = output_root / "public-assets" / "expert" / "data" / f"{level_name}.npz"
            download(gcs_url(remote_path), local_path, config.chunk_size_mb)
        if config.download_bc:
            remote_path = f"bc/{config.bc_step}/policies/{level_name}.pkl"
            local_path = output_root / "public-assets" / "bc" / str(config.bc_step) / "policies" / f"{level_name}.pkl"
            download(gcs_url(remote_path), local_path, config.chunk_size_mb)


if __name__ == "__main__":
    tyro.cli(main)
