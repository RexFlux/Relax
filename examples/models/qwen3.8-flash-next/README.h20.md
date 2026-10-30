# Qwen3.8 Flash Next SFT on 40 H20 GPUs

This recipe runs multimodal SFT on five nodes with eight H20 GPUs per node.
It uses the full 48-layer model, one MTP layer, TP4/PP5/DP2/EP4, and the
pipeline layout `Et*8|t*11|t*11|t*12|t*6mL` (8/11/11/12/6 layers plus MTP).
Packing uses a 65,536-token budget and global batch size 1,024. The vision
model, vision projection, PLE, and attention indexer are frozen; PLE is
offloaded to CPU. Training parameters stay on GPU, with full uniform
activation recomputation.

## Prerequisites

- A running Ray cluster with 40 H20 GPUs and a reachable Ray Jobs HTTP endpoint.
- The same Relax checkout and compatible Megatron-LM, Megatron Bridge, DeepEP,
  Transformer Engine, and Qwen3.8-capable Transformers dependencies on every node.
  This script does not install packages or patch the training image. Verify that
  the model configuration, tokenizer/processor, and training entrypoint import
  successfully in the worker environment before submission.
- A complete Hugging Face model directory containing weights, configuration,
  tokenizer, and multimodal processor assets.
- JSONL SFT data with `messages`, optional `tools`, and images under `images`.
  Media paths in the dataset must resolve on every worker. Chat templating uses
  `preserve_thinking=true` and `add_generation_prompt=false`.
- ClearML credentials configured through the existing runtime environment.

Model, data, output, and dependency directories must be accessible at the same
paths on all nodes. Provide them through environment variables; keep deployment
wrappers such as `launch.sh` and `launch.env` outside the repository.

## Configure and launch

From the repository root, configure your own paths and Ray Jobs address:

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
export MEGATRON="/path/to/Megatron-LM"
export RAY_JOBS_ADDRESS="http://ray-head:8265"
export PROJECT_NAME="Relax/sft/qwen38-qsa"
export EXP_NAME="qwen38-qsa-40xh20"
export NUM_STEPS=10
export RELAX_SKIP_CLUSTER_CLEANUP=1

bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

`DATA_DIR` accepts an exact `.jsonl` file or a directory containing `train.jsonl`.
Direct invocation delegates to `scripts/entrypoint/ray-job.sh`, which configures
the Ray runtime environment. The script checks for active Relax training jobs
before delegating and preserves existing jobs by default. When invoking the
canonical entrypoint yourself, export `RELAX_SKIP_CLUSTER_CLEANUP=1` beforehand.

Submission blocks and follows logs by default. For asynchronous submission, set
`RAY_NO_WAIT=1`; unset it or set it to an empty string for blocking submission
(`RAY_NO_WAIT=0` also enables asynchronous submission). Submission output is saved
under `${SAVE_DIR}/logs`. ClearML task names include the experiment name and
submission timestamp. For asynchronous jobs, follow the returned job ID with
`ray job logs --address="$RAY_JOBS_ADDRESS" <job-id>`.

Each submission initializes from the HF model. This recipe does **not** save or
resume training checkpoints. `SAVE_DIR` holds submission logs; use a distinct
output directory for a separate experiment. Set `NUM_STEPS=200` for a longer
independent run. Additional CLI arguments are forwarded to the training entrypoint.

## Preview and cache settings

With the required path variables set, preview the effective training arguments
and runtime environment without submitting a job or creating output directories:

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

The script forwards numerical and QSA settings to both the Ray driver and training
actors. Override `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH`
to use suitable node-local scratch space. Their defaults are subdirectories of
`${TMPDIR:-/tmp}/relax-qwen38`. The standard Ray entrypoint also supports
`RELAX_KERNEL_CACHE_LOCAL_DIR` for its cache setup. When shared kernel caching
is enabled with `RELAX_KERNEL_CACHE_DIR`, the training actors use that cache
setup in preference to the TorchInductor and Triton directory overrides.
