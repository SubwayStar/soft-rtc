# Soft RTC

语言: [English](README.md) | [简体中文](README.zh-CN.md)

本仓库包含论文 **Action-Prior Denoising for Smooth Real-Time Chunking** 的代码。

Soft RTC 是对 training-time real-time chunking (RTC) 的扩展，面向基于 flow matching 的 action-chunk policy。与只使用二值 committed-prefix mask 的方法不同，Soft RTC 对重叠区域中的动作 token 使用连续的 action-prior 权重进行训练。这样可以在保持 training-time RTC 快速推理特性的同时，提供一个可控的平滑性与任务性能折中。

## 仓库内容

- Kinetix RTC 的核心训练与评估代码。
- Soft RTC 的训练、评估、window sweep、offset sweep、timing 和绘图脚本。
- 用于复现论文摘要结果和图表的轻量 CSV 结果文件。
- Kinetix 仿真器，作为 git submodule 管理。

以下大文件没有提交到仓库：

- `public-assets/`: 公开发布的 RTC demonstrations 和 checkpoints。
- `logs-bc/`、`logs-expert/`、`wandb/`: 本地训练输出。
- 生成的 figures、videos、PDFs 和临时文件。

## 署名与许可

本代码发布基于 Physical Intelligence 公开的 RTC Kinetix 代码。上游代码使用 MIT License，本仓库保留了原始 `LICENSE`。

Soft RTC 新增了 token-wise action-prior weighting、soft-conditioning window 规则、fine-tuning/evaluation 脚本、消融实验脚本、timing 脚本，以及论文图表汇总工具。更详细的署名说明见 `NOTICE`。

论文中的真机实验使用了 OpenPI/LeRobot 风格的部署栈，但本精简代码仓库没有重新分发 OpenPI 真机 runtime 代码。如果后续加入 OpenPI 派生的机器人部署代码，需要保留 OpenPI 的 Apache-2.0 license notices，并清楚标记修改过的文件。

## 部署

克隆仓库并同时拉取 Kinetix submodule：

```bash
git clone --recurse-submodules git@github.com:SubwayStar/soft-rtc.git
cd soft-rtc
```

如果 clone 时没有带 `--recurse-submodules`，可以手动初始化 submodule：

```bash
git submodule update --init --recursive
```

安装 `uv` 并创建 Python 环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

默认依赖使用 `jax[cuda12]`，适合 CUDA 12 GPU 工作站。如果是在 CPU-only 机器或其他 CUDA 版本上运行，需要安装与你机器匹配的 JAX wheel，其余命令流程保持不变。

安装后可以运行一个轻量 import 检查：

```bash
uv run python - <<'PY'
import jax
import kinetix
print("JAX devices:", jax.devices())
print("Kinetix import: ok")
PY
```

如果是在已有 checkout 上更新部署，执行：

```bash
git pull
git submodule update --init --recursive
uv sync
```

本仓库只包含仿真侧训练和评估代码，不包含 OpenPI 真机部署代码，也不包含大体积的公开 RTC assets。

## 下载公开 RTC 资源

脚本默认期望公开资源位于 `public-assets/`。可以用下面命令下载公开的 Kinetix demonstrations 和 base checkpoints：

```bash
uv run src/download_public_assets.py
```

该命令会下载论文使用的 12 个 large Kinetix levels。完整 demonstration 数据比较大，base behavior-cloning checkpoints 相对较小。

资源下载完成后，可以先运行单关卡 smoke test：

```bash
uv run src/eval_public_finetunes_incremental.py \
  --config.level-paths worlds/l/grasp_easy.json \
  --config.output-csv eval_output/smoke_test.csv \
  --config.overwrite
```

正常情况下，该命令会在 `eval_output/` 下生成 CSV 输出。完整复现可以继续使用下面的训练、评估、消融和绘图命令。

## 训练 hard 和 soft training-time RTC

训练 Hard training-time RTC：

```bash
uv run src/run_public_finetune_experiments.py \
  --config.mode hard \
  --config.output-prefix paper
```

训练 Soft RTC，使用论文主实验中的 delay-scaled soft window：

```bash
uv run src/run_public_finetune_experiments.py \
  --config.mode soft \
  --config.output-prefix paperd2 \
  --config.simulated-prefix-attention-horizon 5 \
  --config.simulated-prefix-attention-end-mode scaled \
  --config.simulated-prefix-attention-delay-multiplier 2.0 \
  --config.simulated-prefix-attention-schedule linear
```

默认情况下，这些命令会从公开 base checkpoints 出发 fine-tune 一个 epoch，并将 checkpoints 写入 `logs-bc/`。

## 评估代表性策略

```bash
uv run src/eval_public_finetunes_incremental.py \
  --config.output-csv eval_output/public_soft_rtc_results_d2.csv \
  --config.overwrite
```

评估会在 `eval_output/` 下写入每个 level、每个 delay 的 CSV 文件，以及汇总 CSV 文件。

## 运行消融实验

Delay-scaled 和 fixed-window sweep：

```bash
uv run src/run_soft_overlap_sweep.py
```

Offset-length sweep：

```bash
uv run src/run_offset_length_sweep.py
```

## 重新生成论文汇总结果和图表

仓库中已经包含 `eval_output/` 下的轻量 CSV，因此不需要重新跑完所有训练，也可以生成论文中的汇总结果和图表：

```bash
uv run src/summarize_paper_suite.py
uv run src/make_paper_figures_v2.py
uv run src/make_soft_weight_profile.py
```

生成的图片会写入 `docs/figures/`。

## 注意事项

- 对于多 level sharded 脚本，本地 JAX device 数量需要能整除 level 数量。快速 smoke test 可以传入单个 level，例如 `--config.level-paths worlds/l/grasp_easy.json`。
- 本代码基于公开 RTC Kinetix 代码库和 Kinetix 仿真器。
- 仓库内包含的 CSV 是轻量派生结果，不是原始日志或 checkpoints。
