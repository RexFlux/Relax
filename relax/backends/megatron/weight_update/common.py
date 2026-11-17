# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect
import re
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_update.train_offload import torch_memory_saver_preloaded
from relax.utils.device import is_npu_available
from relax.utils.misc import get_hf_config
from relax.utils.types import ParamInfo


def gdn_component_sizes(config, tp_size: int, name: str) -> tuple[int, ...] | None:
    """Component-major GDN layout; unrelated parameters use ordinary TP
    metadata."""
    if not any(key in name for key in ("self_attention.conv1d.weight", "self_attention.in_proj.weight")):
        return None
    config = getattr(config, "text_config", None) or config
    if not all(
        hasattr(config, key)
        for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    ):
        return None
    if config.linear_num_key_heads % tp_size or config.linear_num_value_heads % tp_size:
        raise ValueError("GDN head counts must be divisible by TP size")
    qk = config.linear_key_head_dim * (config.linear_num_key_heads // tp_size)
    value = config.linear_value_head_dim * (config.linear_num_value_heads // tp_size)
    heads = config.linear_num_value_heads // tp_size
    return (qk, qk, value) if "conv1d" in name else (qk, qk, value, value, heads, heads)


def _restore_tp_weight(name, partitions, partition_dim, split_sizes=None):
    if split_sizes is not None:
        # At TP=1 the gathered tensor already has every GDN component in order.
        if len(partitions) == 1:
            return partitions[0].contiguous()
        assert partition_dim == 0, "GDN components must be sharded along rows"
        components = [part.split(split_sizes, dim=0) for part in partitions]
        return torch.cat([parts[index] for index in range(len(split_sizes)) for parts in components], dim=0)
    if "linear_fc1.weight" in name and "vision_model" not in name:
        components = [part.chunk(2, dim=0) for part in partitions]
        partitions = [part[0] for part in components] + [part[1] for part in components]
        if is_npu_available:
            partition_dim = 0
    if "linear_fc2.weight" in name and "vision_model" not in name and partition_dim == 0:
        partition_dim = 1
    return torch.cat(partitions, dim=partition_dim)


def _gather_tp_weights(named_params, *, split_sizes, async_op):
    tasks, handles = [], []
    for name, param in named_params:
        name = name.replace(".to_wrap.", ".")
        if "expert_bias" in name:
            tasks.append((name, param, None, None))
            continue
        if not hasattr(param, "tensor_model_parallel"):
            multimodal = any(part in name for part in (".audio_model.", ".vision_model.", ".video_model."))
            assert multimodal or not isinstance(param, torch.nn.Parameter), f"{name} lacks TP metadata"
            tasks.append((name, param, None, None))
            continue
        if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            tasks.append((name, param.data, None, None))
            continue
        if ".experts." in name:
            size, group = mpu.get_expert_tensor_parallel_world_size(), mpu.get_expert_tensor_parallel_group()
        else:
            size, group = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()
        if "linear_fc1" not in name:
            assert getattr(param, "partition_stride", 1) == 1, f"Unsupported partition stride for {name}"
        partitions = [torch.empty_like(param.data) for _ in range(size)]
        handle = dist.all_gather(partitions, param.data, group=group, async_op=async_op)
        if handle is not None:
            handles.append(handle)
        tasks.append((name, None, partitions, param.partition_dim))
    # All async collectives launch before any wait. Results remain in input order.
    for handle in handles:
        handle.wait()
    return [
        direct if partitions is None else _restore_tp_weight(name, partitions, dim, split_sizes.get(name))
        for name, direct, partitions, dim in tasks
    ]


def all_gather_param(args, name: str, param: torch.Tensor) -> torch.Tensor:
    """Gather one parameter using the same layout rules as batched export."""
    name = name.replace(".to_wrap.", ".")
    split_sizes = {}
    if (
        getattr(param, "tensor_model_parallel", False)
        and getattr(param, "parallel_mode", None) != "duplicated"
        and any(key in name for key in ("self_attention.conv1d.weight", "self_attention.in_proj.weight"))
    ):
        split_sizes[name] = gdn_component_sizes(
            get_hf_config(args.hf_checkpoint), mpu.get_tensor_model_parallel_world_size(), name
        )
    return _gather_tp_weights([(name, param)], split_sizes=split_sizes, async_op=False)[0]


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
    *,
    gdn_split_sizes: Mapping[str, tuple[int, ...]] | None = None,
) -> list[torch.Tensor]:
    """Launch every parameter gather before waiting, then restore shared TP
    layouts."""
    return _gather_tp_weights(
        [(info.name, param) for info, param in param_infos_and_params],
        split_sizes=gdn_split_sizes or {},
        async_op=True,
    )


def all_gather_params_gated(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
    *,
    error_gate: Callable[[str | None], None],
    gdn_split_sizes: Mapping[str, tuple[int, ...]] | None = None,
) -> list[torch.Tensor]:
    """Gate local preparation before any TP collective, and restoration after.

    A transport/process failure still follows the process group's timeout and
    invalidates the update. A local allocation or metadata failure must not
    send peers into a data collective while this rank enters an error gate.
    """
    tasks, plans, results = [], [], []
    error = None
    try:
        for info, param in param_infos_and_params:
            name = info.name.replace(".to_wrap.", ".")
            if "expert_bias" in name:
                tasks.append((name, param, None, None))
                continue
            if not hasattr(param, "tensor_model_parallel"):
                multimodal = any(part in name for part in (".audio_model.", ".vision_model.", ".video_model."))
                if not multimodal and isinstance(param, torch.nn.Parameter):
                    raise ValueError(f"{name} lacks TP metadata")
                tasks.append((name, param, None, None))
                continue
            if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
                tasks.append((name, param.data, None, None))
                continue
            if ".experts." in name:
                raise ValueError("Gated nonexpert gathering received a routed expert")
            if "linear_fc1" not in name and getattr(param, "partition_stride", 1) != 1:
                raise ValueError(f"Unsupported partition stride for {name}")
            group = mpu.get_tensor_model_parallel_group()
            size = dist.get_world_size(group)
            partitions = [torch.empty_like(param.data) for _ in range(size)]
            plans.append((partitions, param.data, group))
            tasks.append((name, None, partitions, param.partition_dim))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    error_gate(error)
    error = None
    try:
        handles = [dist.all_gather(parts, param, group=group, async_op=True) for parts, param, group in plans]
        for handle in handles:
            handle.wait()
        splits = gdn_split_sizes or {}
        results = [
            direct if parts is None else _restore_tp_weight(name, parts, dim, splits.get(name))
            for name, direct, parts, dim in tasks
        ]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    error_gate(error)
    return results


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
    include_persistent_buffers: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model, include_persistent_buffers=include_persistent_buffers)
    else:
        ans = _named_params_and_buffers_vanilla(model, include_persistent_buffers=include_persistent_buffers)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor) -> torch.Tensor:
    # Selective offload (weight_update/train_offload.py): when the offloader
    # frees a param's GPU storage via storage().resize_(0), it stashes the CPU copy
    # on the tensor as ``_relax_cpu_offload_data``. If the GPU storage is empty, read
    # from that CPU copy instead of touching the now-invalid CUDA storage.
    if getattr(x, "_relax_cpu_offload_data", None) is not None and x.storage().size() == 0:
        return x._relax_cpu_offload_data

    # torch_memory_saver path: only usable when its LD_PRELOAD hook is active;
    # otherwise get_cpu_backup() would assert on an uninitialized saver.
    # NOTE: with --selective-offload, actor_group.py does NOT LD_PRELOAD the TMS hook,
    # so torch_memory_saver_preloaded() is False here and this branch is skipped —
    # i.e. LD_PRELOAD is the single source of truth for "TMS is the active mechanism".
    if torch_memory_saver_preloaded():
        from torch_memory_saver import torch_memory_saver

        if (cpu_tensor := torch_memory_saver.get_cpu_backup(x)) is not None:
            return cpu_tensor

    return x


def _named_buffers_for_weight_update(
    model_module: torch.nn.Module, include_persistent_buffers: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    if include_persistent_buffers:
        # Match Megatron Bridge: checkpoint state includes persistent buffers,
        # but excludes runtime caches and counters registered as non-persistent.
        for module_prefix, module in model_module.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", ())
            for local_name, buffer in module.named_buffers(recurse=False):
                if local_name in non_persistent:
                    continue
                name = f"{module_prefix}.{local_name}" if module_prefix else local_name
                yield name, buffer
        return

    for name, buffer in model_module.named_buffers():
        if "expert_bias" in name:
            yield name, buffer


def _named_params_and_buffers_vanilla(
    model: Sequence[torch.nn.Module], include_persistent_buffers: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in _named_buffers_for_weight_update(model_module, include_persistent_buffers):
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module], include_persistent_buffers: bool = False
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (global_name, param/buffer) with consistent names across PP/EP.

    Adjusts indices for virtual PP + EP offsets. Handles decoder.layers,
    mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:(.*)\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.(?:(.*)\.)?mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                middle_path, layer_idx, rest = match.groups()
                # GPT MTP uses transformer_layer; hybrid MTP (e.g. Qwen3.8)
                # nests its transformer layer inside mtp_model_layer.layers.
                expert_pattern = (
                    r"((?:transformer_layer|mtp_model_layer\.layers\.\d+)\.mlp\.experts\..+)\.weight(\d+)$"
                )
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                if middle_path is None:
                    middle_path = ""
                else:
                    middle_path += "."
                yield (
                    f"module.module.{middle_path}mtp.layers.{layer_idx}.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            middle_path, layer_idx, rest = match.groups()
            if middle_path is None:
                middle_path = ""
            else:
                middle_path += "."
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.{middle_path}decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
            else:
                yield f"module.module.{middle_path}decoder.layers.{layer_idx}.{rest}", param

        # Treat selected model-state buffers needed by rollout as normal parameters.
        for name, buffer in _named_buffers_for_weight_update(model_module, include_persistent_buffers):
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:(.*)\.)?decoder\.layers\.(\d+)\.(.+)"

            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                middle_path, layer_idx, rest = match.groups()
                if middle_path is None:
                    middle_path = ""
                else:
                    middle_path += "."
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.{middle_path}decoder.layers.{layer_idx}.{rest}", buffer


def install_gpu_memory_update(
    updater,
    streams: dict,
    targets: list | None,
    lock,
    *,
    metadata_group,
    group_update: str,
    full: bool,
    version: int,
    deadline: float,
    on_commit: Callable[[], None],
    update_started: float,
) -> dict:
    """One fixed all-source plan, engine finalization, then the existing ACK
    gate.

    Only small frame directories cross Gloo. Each owner sends its own GPU
    buffers directly to the intended TP proxies; rank zero coordinates
    controls.
    """
    import hashlib
    import secrets
    import socket
    from time import monotonic, sleep

    import ray

    from relax.backends.sglang.proxy_weight_loader import json_bytes
    from relax.distributed.weight_sync.gpu_transport import (
        GpuSyncGroup,
        plan_rounds,
        remaining,
        select_overlap,
    )

    rank, world = dist.get_rank(metadata_group), dist.get_world_size(metadata_group)
    root = dist.get_process_group_ranks(metadata_group)[0]
    config = updater.config
    started = monotonic()
    stages = {}

    def gate(error):
        errors = [None] * world
        dist.all_gather_object(errors, error, group=metadata_group)
        if any(errors):
            raise RuntimeError(f"GPU memory source gate failed: {errors}")

    def root_call(action, label=None):
        result = [None]
        if rank == 0:
            try:
                action_started = monotonic()
                result[0] = dict(value=action())
                if label is not None:
                    stages[label] = stages.get(label, 0.0) + monotonic() - action_started
            except Exception as exc:
                result[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(result, src=root, group=metadata_group)
        if "error" in result[0]:
            raise RuntimeError(result[0]["error"])
        return result[0]["value"]

    local, error = {}, None
    try:
        for name, item in streams.items():
            manifest = dict(
                stream=name,
                version=version,
                base=version - 1,
                reset=version == 0,
                schema=item["encoder"].pending_schema,
                group_update=group_update,
                buckets=[bucket.metadata() for bucket in item["buckets"]],
            )
            manifest["update"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
            local[name] = dict(manifest=manifest, members=sorted(item["members"]), order=item["order"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    gate(error)
    gathered = [None] * world
    dist.all_gather_object(gathered, local, group=metadata_group)
    owners = [source for source, item in enumerate(gathered) if item]
    owner_rank = {source: index for index, source in enumerate(owners)}
    manifests = {name: item["manifest"] for items in gathered for name, item in items.items()}
    if sum(map(len, gathered)) != len(manifests):
        raise ValueError("GPU source streams have duplicate owners")

    def build_plan():
        members = tuple(member for target in targets for member in target["members"])
        frames = []
        ordered = sorted(
            (tuple(item["order"]), name, source, item)
            for source, items in enumerate(gathered)
            for name, item in items.items()
        )
        for _order, name, source, item in ordered:
            for bucket, metadata in enumerate(item["manifest"]["buckets"]):
                frames.append(
                    dict(
                        id=len(frames),
                        stream=name,
                        bucket=bucket,
                        owner=owner_rank[source],
                        bytes=metadata["bytes"],
                        recipients=item["members"],
                    )
                )
        for target in targets:
            if len(target["members"]) != len(target["streams"]) or len(target["controls"]) != len(target["members"]):
                raise ValueError("GPU target controls disagree with actual worker topology")
            for member, expected in zip(target["members"], target["streams"], strict=True):
                actual = [name for _order, name, _source, item in ordered if member in item["members"]]
                if actual != expected:
                    raise ValueError("GPU owner stream order disagrees with receiver adapter")
        plan_rounds(frames, members, config.staging_budget_bytes)  # Validate before creating any proxy process.
        connection = getattr(updater, "_gpu_connection", None)
        if connection is None:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            host = ray.util.get_node_ip_address()
            connection = dict(
                init_method=f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}",
                world_size=len(owners) + len(members),
                group_name="relax-gpu-" + secrets.token_hex(12),
            )
            updater._gpu_connection = connection
        return dict(
            frames=frames,
            members=members,
            connection=connection,
        )

    plan = root_call(build_plan)
    receiver_ranks = {member: len(owners) + index for index, member in enumerate(plan["members"])}
    join_refs = None
    if rank == 0:
        join_refs = [
            target["engine"].gpu_memory_proxy.remote(
                "join",
                connection=plan["connection"],
                ranks=[receiver_ranks[m] for m in target["members"]],
                timeout=remaining(deadline),
            )
            for target in targets
        ]
    error = None
    try:
        if rank in owner_rank and getattr(updater, "_gpu_group", None) is None:
            updater._gpu_group = GpuSyncGroup(
                plan["connection"],
                rank=owner_rank[rank],
                device=torch.device("cuda", torch.cuda.current_device()),
                timeout_s=remaining(deadline),
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    gate(error)
    joined = root_call(lambda: ray.get(join_refs, timeout=remaining(deadline)), "proxy_join_result_wait_s")

    def admit():
        requirements = {}
        capacities, available = [], []
        for target, identities in zip(targets, joined, strict=True):
            for member, control, names, identity in zip(
                target["members"], target["controls"], target["streams"], identities, strict=True
            ):
                cache_names = names if control["adapter"] == "hf" else names[-1:]
                cache = sum(
                    bucket["elements"] * {"bfloat16": 2, "float32": 4}[bucket["dtype"]]
                    for name in cache_names
                    for bucket in manifests[name]["buckets"]
                )
                workspace = control["budgets"].get("loader_workspace_budget_bytes", cache)
                scratch = max(
                    (
                        bucket["changed"] * 8 if bucket["encoding"] == "sparse" else 0
                        for name in names
                        for bucket in manifests[name]["buckets"]
                    ),
                    default=0,
                )
                future = (cache if version == 0 else 0) + workspace + scratch
                capacity = identity.get("staging_capacity_bytes", 0)
                capacities.append(capacity)
                available.append(identity["free_bytes"] - future)
                requirements[member] = dict(
                    cache_bytes=cache,
                    workspace_bytes=workspace,
                    scratch_bytes=scratch,
                    physically_free_bytes=identity["free_bytes"],
                    resident_staging_capacity_bytes=capacity,
                )
        resident = {size for size in capacities if size}
        if len(resident) > 1:
            raise ValueError("Fixed proxy arenas disagree on their admitted capacity")
        if resident:
            capacity = resident.pop()
            if capacity > config.staging_budget_bytes or any(size == 0 for size in capacities) or min(available) < 0:
                raise ValueError("Existing staging arena or loader exceeds current physical admission")
        else:
            capacity = min(config.staging_budget_bytes, min(available))
            # Large CUDA allocations round to allocator segments. Keep a
            # physically constrained reservation within the measured margin.
            if capacity < config.staging_budget_bytes and capacity >= 2 << 20:
                capacity = capacity // (2 << 20) * (2 << 20)
        rounds = plan_rounds(plan["frames"], plan["members"], capacity)
        effective, reason = select_overlap(rounds, requested=config.overlap, full=full, budget_bytes=capacity)
        for member, requirement in requirements.items():
            requirement.update(
                largest_round_bytes=max(batch[member]["bytes"] for batch in rounds),
                complete_payload_bytes=sum(batch[member]["bytes"] for batch in rounds),
            )
        return dict(
            effective_overlap=effective,
            overlap_reason=reason,
            admission=requirements,
            staging_capacity_bytes=capacity,
            rounds=rounds,
        )

    admission = root_call(admit)
    plan["rounds"] = admission.pop("rounds")
    effective = admission["effective_overlap"]
    payloads = {
        frame["id"]: streams[frame["stream"]]["buckets"][frame["bucket"]].payload
        for frame in plan["frames"]
        if frame["stream"] in streams
    }
    # Pin the source payloads on ambiguous transport failure. Encoder abort must
    # not recycle storage still referenced by an outstanding NCCL operation.
    updater._gpu_active_payloads = payloads
    acquired = False
    timings = {}

    def engine_call(action, arguments):
        return ray.get(
            [
                target["engine"].gpu_memory_proxy.remote(action, **kwargs, timeout=remaining(deadline))
                for target, kwargs in zip(targets, arguments, strict=True)
            ],
            timeout=remaining(deadline),
        )

    def pause():
        ray.get(
            [target["engine"].pause_generation.remote(timeout=remaining(deadline)) for target in targets],
            timeout=remaining(deadline),
        )

    def begin():
        return engine_call(
            "begin",
            [
                dict(controls=t["controls"], manifests=[[manifests[name] for name in names] for names in t["streams"]])
                for t in targets
            ],
        )

    def consume(round_id, batch):
        arguments = []
        for target in targets:
            entries = []
            for member, names in zip(target["members"], target["streams"], strict=True):
                indices = {name: index for index, name in enumerate(names)}
                entries.append(
                    [
                        [
                            indices[plan["frames"][entry["frame"]]["stream"]],
                            plan["frames"][entry["frame"]]["bucket"],
                            entry["offset"],
                            entry["bytes"],
                        ]
                        for entry in batch[member]["entries"]
                    ]
                )
            arguments.append(dict(round_id=round_id, entries=entries))
        return engine_call("consume", arguments)

    try:

        def acquire():
            nonlocal acquired
            while not ray.get(lock.acquire.remote(), timeout=remaining(deadline)):
                sleep(min(0.05, remaining(deadline)))
            acquired = True

        root_call(acquire, "lock_acquire_s")
        if not effective:
            timings["paused_at"] = monotonic()
            root_call(pause, "pause_calls_s")
            root_call(begin, "begin_and_directories_s")
        for round_id, batch in enumerate(plan["rounds"]):
            transport_started = monotonic()
            receive_refs = None

            def dispatch_receive():
                nonlocal receive_refs
                # NCCL creates peer communicators lazily. Waiting for irecv's
                # posted reply before any source enters isend can deadlock the
                # first pair initialization despite the all-reduce warmup.
                receive_refs = [
                    target["engine"].gpu_memory_proxy.remote(
                        "receive",
                        round_id=round_id,
                        batches=[batch[m] for m in target["members"]],
                        budget=admission["staging_capacity_bytes"],
                        timeout=remaining(deadline),
                    )
                    for target in targets
                ]

            root_call(dispatch_receive)
            error = None
            try:
                if rank in owner_rank:
                    updater._gpu_group.send(plan["frames"], batch, payloads, receiver_ranks, deadline=deadline)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            gate(error)
            root_call(lambda: ray.get(receive_refs, timeout=remaining(deadline)))
            readiness = root_call(lambda: engine_call("ready", [dict(round_id=round_id) for _ in targets]))
            if rank == 0:
                stages["transport_and_ready_wall_s"] = (
                    stages.get("transport_and_ready_wall_s", 0.0) + monotonic() - transport_started
                )
            if not effective:
                root_call(lambda: consume(round_id, batch), "ipc_apply_and_fence_calls_s")
        if effective:
            timings["paused_at"] = monotonic()
            root_call(pause, "pause_calls_s")
            root_call(begin, "begin_and_directories_s")
            for round_id, batch in enumerate(plan["rounds"]):
                root_call(lambda: consume(round_id, batch), "ipc_apply_and_fence_calls_s")
        loaded = root_call(
            lambda: engine_call(
                "finalize",
                [
                    dict(
                        update=manifests[target["streams"][0][0]]["update"]
                        if target["controls"][0]["adapter"] == "hf"
                        else group_update,
                    )
                    for target in targets
                ],
            ),
            "finalize_loader_and_version_calls_s",
        )
        error = None
        try:
            for name, item in streams.items():
                item["encoder"].prepare_commit(set(local[name]["members"]), set(item["members"]))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        gate(error)
        error = None
        try:
            for item in streams.values():
                item["encoder"].commit()
            on_commit()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        gate(error)
        updater._gpu_active_payloads = None

        def resume():
            ray.get(
                [t["engine"].continue_generation.remote(timeout=remaining(deadline)) for t in targets],
                timeout=remaining(deadline),
            )
            ready = monotonic()
            ray.get(
                [t["engine"].memory_weight_generation_probe.remote(remaining(deadline)) for t in targets],
                timeout=remaining(deadline),
            )
            return dict(
                sync_to_ready_s=ready - update_started,
                generation_pause_s=ready - timings["paused_at"],
                install_to_ready_s=ready - started,
                probe_s=monotonic() - ready,
                stages=stages,
            )

        timing_result = root_call(resume)
        return dict(
            transport="nccl",
            **admission,
            receivers=loaded,
            rounds=len(plan["rounds"]),
            frames=len(plan["frames"]),
            proxy_last_round_memory=readiness,
            logical_delivery_bytes=sum(f["bytes"] * len(f["recipients"]) for f in plan["frames"]),
            full_delivery_bytes=sum(
                b["elements"] * {"bfloat16": 2, "float32": 4}[b["dtype"]] * len(item["members"])
                for items in gathered
                for item in items.values()
                for b in item["manifest"]["buckets"]
            ),
            **timing_result,
        )
    finally:
        root_call(lambda: ray.get(lock.release.remote(), timeout=10) if acquired else None)
