# Soft RTC

This repository contains the code for **Action-Prior Denoising for Smooth Real-Time Chunking**.

Soft RTC extends training-time real-time chunking (RTC) for flow-matching action-chunk policies. Instead of using only a binary committed-prefix mask, it trains overlap tokens with token-wise action-prior weights. The result is a lightweight deployment rule that keeps the fast inference profile of training-time RTC while exposing a smoothness-performance tradeoff.

## What is included

- Core Kinetix RTC training and evaluation code.
- Soft RTC training, evaluation, window-sweep, offset-sweep, timing, and plotting scripts.
- Lightweight CSV result files used to regenerate paper summaries and figures.
- The Kinetix simulator as a git submodule.

Large assets are intentionally not committed:

- `public-assets/`: released RTC demonstrations and checkpoints.
- `logs-bc/`, `logs-expert/`, `wandb/`: local training outputs.
- generated figures, videos, PDFs, and temporary files.

## Attribution and license

This code release is based on the public RTC Kinetix code released by Physical Intelligence for the RTC papers. The upstream code is MIT licensed, and the original license is retained in `LICENSE`.

Soft RTC adds token-wise action-prior weighting, soft-conditioning window rules, fine-tuning/evaluation scripts, ablations, timing scripts, and paper-summary utilities. See `NOTICE` for the attribution statement.

The real-robot experiments in the paper used an OpenPI/LeRobot-style deployment stack, but this compact release does not redistribute OpenPI runtime code. If OpenPI-derived deployment code is later added to this repository, keep OpenPI's Apache-2.0 license notices and clearly mark modified files.

## Setup

```bash
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

The default dependency set uses `jax[cuda12]`. If you need a CPU-only or different CUDA setup, install the matching JAX wheel for your machine and then run the scripts with the same command pattern.

## Download public RTC assets

The scripts expect public assets under `public-assets/`. Download the released Kinetix demonstrations and base checkpoints with:

```bash
uv run src/download_public_assets.py
```

This downloads the 12 large Kinetix levels used in the paper. The full demonstration data is large; the base behavior-cloning checkpoints are much smaller.

## Train hard and soft training-time RTC

Hard training-time RTC:

```bash
uv run src/run_public_finetune_experiments.py \
  --config.mode hard \
  --config.output-prefix paper
```

Soft RTC with the main delay-scaled window:

```bash
uv run src/run_public_finetune_experiments.py \
  --config.mode soft \
  --config.output-prefix paperd2 \
  --config.simulated-prefix-attention-horizon 5 \
  --config.simulated-prefix-attention-end-mode scaled \
  --config.simulated-prefix-attention-delay-multiplier 2.0 \
  --config.simulated-prefix-attention-schedule linear
```

By default, these commands fine-tune one epoch from released base checkpoints and write checkpoints to `logs-bc/`.

## Evaluate representative policies

```bash
uv run src/eval_public_finetunes_incremental.py \
  --config.output-csv eval_output/public_soft_rtc_results_d2.csv \
  --config.overwrite
```

The evaluation writes per-level, per-delay CSV files and summary CSV files under `eval_output/`.

## Run ablations

Delay-scaled and fixed-window sweep:

```bash
uv run src/run_soft_overlap_sweep.py
```

Offset-length sweep:

```bash
uv run src/run_offset_length_sweep.py
```

## Regenerate paper summaries and figures

The repository includes lightweight CSVs in `eval_output/`, so the summary and figure scripts can run without redoing all training.

```bash
uv run src/summarize_paper_suite.py
uv run src/make_paper_figures_v2.py
uv run src/make_soft_weight_profile.py
```

Generated figures are written to `docs/figures/`.

## Notes

- The number of local JAX devices must divide the number of levels for multi-level sharded scripts. For quick smoke tests, pass a single `--config.level-paths worlds/l/grasp_easy.json`.
- The code is built on the public RTC Kinetix codebase and the Kinetix simulator.
- The included CSV files are lightweight derived results, not raw logs or checkpoints.
