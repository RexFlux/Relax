# Qwen3.8 Flash Next SFT on 40 H20 GPUs

This recipe runs multimodal SFT on five nodes with eight H20 GPUs per node.
It uses the full 48-layer model, one MTP layer, TP4/PP5/DP2/EP4, and the
pipeline layout `Et*8|t*11|t*11|t*12|t*6mL` (8/11/11/12/6 layers plus MTP).
Packing uses a 65,536-token budget and global batch size 1,024. The vision
model, vision projection, PLE, and attention indexer are frozen; PLE is
offloaded to CPU. The frozen vision components still participate in multimodal
forward computation. Training parameters stay on GPU, with full uniform
activation recomputation.

## Prerequisites

- Five dedicated nodes with eight H20 GPUs each, running the same Relax checkout
  and compatible Megatron-LM, Megatron Bridge, DeepEP, Transformer Engine, and
  Qwen3.8-capable Transformers dependencies. The standard image installs
  Megatron-LM at `/root/Megatron-LM`; override `MEGATRON` for another location.
  This recipe does not install packages or patch the training image.
- A complete Hugging Face model directory with weights, configuration,
  tokenizer, and multimodal processor assets.
- JSONL SFT data with `messages`, optional `tools`, and the supported multimodal
  message format. Media paths must resolve on every worker. Chat templating uses
  `preserve_thinking=true` and `add_generation_prompt=false`.
- ClearML credentials configured in the runtime environment.

Model, data, output, and source directories must be accessible at the same paths
on all nodes. Verify model configuration, tokenizer/processor, and training
entrypoint imports in the worker environment before launching.

## Configure paths

Set these three environment variables, or fill their defaults in
[`launch.sh`](launch.sh). No separate `launch.env` file is needed.

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
```

`DATA_DIR` accepts an exact `.jsonl` file or a directory containing `train.jsonl`.
`SAVE_DIR` stores submission logs. Use a distinct output directory for a separate
experiment. The source checkout is located automatically from the launch script.

## Launch

For a multi-node TorchJob, run the following command on **every node** from the
repository root. The standard entrypoint starts Ray and submits training on the
head node; the other nodes join as workers.

```bash
bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/launch.sh
```

The job environment must provide `MASTER_ADDR` (the head Pod's hostname,
matching its `POD_NAME`), `POD_NAME`, `HOST_IP`, and `WORLD_SIZE=5`. Each node uses
eight GPUs by default. Use this entrypoint on dedicated job nodes; it cleans
stale processes and starts a new Ray cluster.

For an **existing Ray cluster**, run only on the Ray head instead:

```bash
bash examples/models/qwen3.8-flash-next/launch.sh
```

This delegates to `scripts/entrypoint/ray-job.sh`, preserves existing jobs by
default, and rejects another active Relax training job before submission.
The Ray Jobs address defaults to `http://127.0.0.1:8265` on the head; override
`RAY_JOBS_ADDRESS` only when using a different endpoint.

Both launch paths use the same training script under `scripts/`. Submission
blocks and follows logs by default, writing
`${SAVE_DIR}/logs/<submission-id>.submit.log`. For asynchronous submission set
`RAY_NO_WAIT=1`; leave it unset or empty for blocking operation (`0` is also
nonempty and enables asynchronous submission).

## Steps, metrics, and checkpoints

The default is a **10-step short run**. Change `NUM_STEPS` for your own run
length. `EXP_NAME` and `PROJECT_NAME` optionally override
the script's ClearML experiment and project defaults. Task names include the
experiment name and launch time; each submission creates a separate task.

Each launch initializes from the HF model. This recipe does **not** save or
resume training checkpoints. Additional CLI arguments are forwarded to the
training entrypoint.

## Preview and caches

With the three path variables set, preview arguments and runtime settings
without submitting a job, importing GPU dependencies, or creating output files:

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/launch.sh
```

Compilation caches use node-local defaults under `${TMPDIR:-/tmp}/relax-qwen38`.
The standard entrypoint's kernel-cache configuration still applies. Override
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, or `CUDA_CACHE_PATH` only if your
environment needs a different local cache location.
