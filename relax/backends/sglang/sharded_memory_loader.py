# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-owner expert shards and typed HF buffers for a paused MoE group."""

from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any

from relax.utils.logging_utils import get_logger


SHARDED_MEMORY_LOADER = "relax.backends.sglang.sharded_memory_loader.load_sharded_memory_weights"
_EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(w13_weight|w2_weight)")
_PREFIX = "relax.expert_slice.v1:"
logger = get_logger(__name__)


class ShardedMemoryReceiver:
    """A failed multi-stream install stays poisoned until a new complete FULL.

    Expert buffers are the existing native parameter slices. Only nonexpert HF
    tensors persist separately, under a per-worker GPU budget. The updater owns
    the fixed group, pause lock and cross-source commit gate.
    """

    def __init__(self, model: Any, *, gpu_budget_bytes: int) -> None:
        if type(gpu_budget_bytes) is not int or gpu_budget_bytes <= 0:
            raise ValueError("A positive receiver GPU budget is required")
        self.model, self.gpu_budget_bytes = model, gpu_budget_bytes
        self.version = -1
        self.root: str | None = None
        self._attempted_root: str | None = None
        self._topology: tuple | None = None
        self.poisoned = False
        self.layouts: dict[str, list] = {}
        self.schemas: dict[str, str] = {}
        self.hf: dict[str, Any] = {}
        self.storage: dict[str, tuple] = {}
        self.hf_bytes = 0
        self._seen_hf: set[str] = set()
        self._control: dict | None = None

    def poison(self) -> None:
        self.poisoned = True
        self._control = None

    def apply_group(self, control: dict, receive: Any) -> dict:
        """TCP compatibility path uses the same per-update adapter session."""
        try:
            self.begin(control)
            for index, item in enumerate(control["streams"]):

                def apply(manifest: dict, payloads: list) -> None:
                    if len(manifest["buckets"]) != len(payloads):
                        raise ValueError("Shard payload count mismatch")
                    self.begin_stream(index, manifest)
                    for frame, payload in enumerate(payloads):
                        self.consume(index, frame, payload)

                receive(item["descriptor"], item["member"], apply)
            return self.finalize()
        except BaseException:
            self.poison()
            raise

    def begin(self, control: dict) -> None:
        """Validate group topology once without invoking the ordinary
        loader."""
        root, version, group_update = control["stream"], control["version"], control["group_update"]
        reset = control.get("reset") is True
        tp_rank, tp_size, owners = control["tp_rank"], control["tp_size"], control["owners"]
        ep_size, moe_dp_size = control.get("ep_size", 1), control.get("moe_dp_size", 1)
        if (
            not isinstance(root, str)
            or not root
            or not isinstance(group_update, str)
            or not group_update
            or type(version) is not int
            or version < 0
            or type(tp_rank) is not int
            or type(tp_size) is not int
            or tp_size <= 0
            or not 0 <= tp_rank < tp_size
            or type(owners) is not int
            or owners <= 0
        ):
            raise ValueError("Invalid sharded group identity or topology")
        if type(ep_size) is not int or ep_size != 1 or type(moe_dp_size) is not int or moe_dp_size != 1:
            raise ValueError("Native owner streams require receiver EP1 and MoE DP1")
        topology = (tp_rank, tp_size, owners, ep_size, moe_dp_size)
        if not reset and topology != self._topology:
            raise ValueError("Sharded topology changed; create a new FULL stream")
        expected = [f"{root}:expert:{owner}:tp:{tp_rank}" for owner in range(owners)] + [f"{root}:nonexpert"]
        streams = control["streams"]
        if [item.get("stream") for item in streams] != expected:
            raise ValueError("Sharded request must contain the exact ordered owner stream set")
        if not reset and (self.poisoned or root != self.root or version != self.version + 1):
            raise ValueError("A failed or mismatched group needs a new FULL stream")
        if reset and root in (self.root, self._attempted_root):
            raise ValueError("FULL reset requires a new group stream identity")
        if any(
            value is not None and name not in module._non_persistent_buffers_set
            for module in self.model.modules()
            for name, value in module._buffers.items()
        ):
            raise ValueError("Persistent inference buffers require explicit synchronization rules")
        self._validate_storage()
        self.poisoned = True
        self._attempted_root = root
        if reset:
            self.hf.clear()
            self.hf_bytes = 0
            self.layouts.clear()
            self.schemas.clear()
        self._coverage: dict[str, list[tuple[int, int]]] = {}
        self._seen_hf = set()
        self._control = control
        self._manifests = {}
        self._next_stream = self._next_bucket = 0
        self._started = monotonic()
        self._gpu_apply_s = 0.0

    def begin_stream(self, index: int, manifest: dict) -> None:
        try:
            self._begin_stream(index, manifest)
        except BaseException:
            self.poison()
            raise

    def _begin_stream(self, index: int, manifest: dict) -> None:
        import hashlib

        from relax.distributed.weight_sync.memory_delta import validate_bucket_directory

        control = self._control
        if control is None or index in self._manifests or not 0 <= index < len(control["streams"]):
            raise ValueError("Invalid or duplicate native stream directory")
        stream = control["streams"][index]["stream"]
        if (
            manifest["stream"] != stream
            or manifest.get("group_update") != control["group_update"]
            or manifest["version"] != control["version"]
            or manifest["base"] != control["version"] - 1
            or manifest.get("reset") is not control["reset"]
        ):
            raise ValueError("Mixed source update identities or versions")
        metadata = manifest["buckets"]
        if not metadata:
            raise ValueError("Empty native stream directory")
        layouts = [tuple((name, tuple(shape)) for name, shape in bucket["names"]) for bucket in metadata]
        typed = [(names, bucket["dtype"]) for names, bucket in zip(layouts, metadata, strict=True)]
        schema = hashlib.sha256(json.dumps(typed, separators=(",", ":")).encode()).hexdigest()
        if schema != manifest["schema"]:
            raise ValueError("Native stream schema does not match its directory")
        if not control["reset"] and (layouts != self.layouts.get(stream) or schema != self.schemas.get(stream)):
            raise ValueError("Shard schema or bucket layout changed")
        for bucket in metadata:
            validate_bucket_directory(bucket, max_bytes=self.gpu_budget_bytes)
            if control["reset"] and bucket["encoding"] != "full":
                raise ValueError("Every reset stream must be FULL")
        self._manifests[index] = manifest

    def consume(self, stream_index: int, frame_index: int, payload: Any) -> None:
        import torch

        from relax.distributed.weight_sync.gpu_payload import validate_raw_frame

        started = monotonic()
        try:
            if self._control is None or (stream_index, frame_index) != (self._next_stream, self._next_bucket):
                raise ValueError("Native frames must be consumed exactly once in stream order")
            manifest = self._manifests[stream_index]
            metadata = manifest["buckets"][frame_index]
            if len(payload) != metadata["bytes"]:
                raise ValueError("Shard payload size mismatch")
            if isinstance(payload, torch.Tensor):
                validate_raw_frame(metadata, payload)
            device = next(self.model.parameters()).device
            kind = "nonexpert" if stream_index == self._control["owners"] else "expert"
            self._apply_bucket(
                metadata, payload, kind, self._coverage, device, owner=stream_index, owners=self._control["owners"]
            )
            self._next_bucket += 1
            if self._next_bucket == len(manifest["buckets"]):
                # TCP ACK follows the stream callback, before group finalize.
                # Preserve its existing guarantee that every stream write is fenced.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stream = manifest["stream"]
                self.layouts[stream] = [
                    tuple((n, tuple(shape)) for n, shape in b["names"]) for b in manifest["buckets"]
                ]
                self.schemas[stream] = manifest["schema"]
                self._next_stream += 1
                self._next_bucket = 0
            self._gpu_apply_s += monotonic() - started
        except BaseException:
            self.poison()
            raise

    def finalize(self) -> dict:
        """Refresh nonexpert/derived state once after all owner slices
        arrived."""
        try:
            return self._finalize()
        except BaseException:
            self.poison()
            raise

    def _finalize(self) -> dict:
        import torch

        control = self._control
        if control is None or self._next_stream != len(control["streams"]) or self._next_bucket:
            self.poison()
            raise ValueError("Cannot finalize incomplete native owner streams")
        device = next(self.model.parameters()).device
        self._check_expert_coverage(self._coverage)
        # The actual ordinary TP-aware loader handles dense/GDN/shared tensors.
        # It consumes the persistent, typed HF subset; experts never enter it.
        from relax.backends.sglang.weight_write_coverage import record_weight_write_coverage

        expert_names = {
            name for name, _ in self.model.named_parameters(remove_duplicate=False) if _EXPERT.fullmatch(name)
        }
        load_started = monotonic()
        with (
            torch.no_grad(),
            record_weight_write_coverage(
                self.model, required_hf_names=set(self.hf), excluded_expert_names=expert_names
            ) as recorder,
        ):
            self.model.load_weights(recorder.tag_weights(self.hf.items()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        hf_load_s = monotonic() - load_started
        self._remember_storage()
        self.version, self.root, self.poisoned = control["version"], control["stream"], False
        self._control = None
        self._topology = tuple(control.get(k, 1) for k in ("tp_rank", "tp_size", "owners", "ep_size", "moe_dp_size"))
        self.last_metrics = dict(
            stream=control["stream"],
            group_update=control["group_update"],
            schemas=dict(self.schemas),
            version=control["version"],
            streams=len(control["streams"]),
            hf_bytes=self.hf_bytes,
            receive_and_validate_s=max(0.0, monotonic() - self._started - self._gpu_apply_s - hf_load_s),
            gpu_apply_and_fence_s=self._gpu_apply_s,
            hf_load_and_fence_s=hf_load_s,
            receiver_total_s=monotonic() - self._started,
        )
        if device.type == "cuda":
            self.last_metrics.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        logger.info("Sharded memory receiver stages: %s", json.dumps(self.last_metrics, sort_keys=True))
        return self.last_metrics

    def _apply_bucket(
        self, metadata: dict, payload: bytes, kind: str, coverage: dict, device: Any, *, owner: int, owners: int
    ) -> None:
        import torch

        from relax.distributed.weight_sync.gpu_payload import apply_raw_slice, payload_to_device, sparse_boundaries

        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[metadata["dtype"]]
        if kind == "expert" and dtype != torch.bfloat16:
            raise ValueError("Native expert shards must be BF16")
        width = torch.empty((), dtype=dtype).element_size()
        # Include encoded H2D staging and the worst per-bucket sparse index
        # expansion, in addition to all persistent nonexpert buffers.
        scratch = len(payload) + metadata["changed"] * 8
        if self.hf_bytes + scratch > self.gpu_budget_bytes:
            raise ValueError("Receiver staging exceeds its GPU budget")
        # The transport owns private bytearrays until this stream's GPU fence
        # and ACK; immutable callers still get an independent writable copy.
        raw = payload_to_device(payload, device)
        sparse = metadata["encoding"] == "sparse"
        metadata = {**metadata, "sparse_boundaries": sparse_boundaries(metadata, raw)}
        offset = 0
        for tensor_index, (name, shape) in enumerate(metadata["names"]):
            count = torch.Size(shape).numel()
            if kind == "expert":
                target = self._expert_target(name, shape, coverage, owner=owner, owners=owners)
            else:
                if name.startswith(_PREFIX) or ".experts." in name:
                    raise ValueError("Expert tensor in the nonexpert HF stream")
                if name in self._seen_hf:
                    raise ValueError("Duplicate nonexpert tensor in a group update")
                self._seen_hf.add(name)
                target = self.hf.get(name)
                if target is None:
                    size = count * width
                    if sparse or self.hf_bytes + size + scratch > self.gpu_budget_bytes:
                        raise ValueError("Nonexpert FULL buffers exceed their GPU budget")
                    target = torch.empty(tuple(shape), dtype=dtype, device=device)
                    self.hf[name] = target
                    self.hf_bytes += size
            if target.dtype != dtype or tuple(target.shape) != tuple(shape) or not target.is_contiguous():
                raise ValueError("Shard target shape, dtype or storage layout mismatch")
            with torch.no_grad():
                apply_raw_slice(target, metadata, raw, tensor_index=tensor_index, offset=offset)
            offset += count

    def _expert_target(self, descriptor: str, shape: list, coverage: dict, *, owner: int, owners: int) -> Any:
        import torch

        if not descriptor.startswith(_PREFIX):
            raise ValueError("A native expert shard requires a slice descriptor")
        name, start, full_shape = json.loads(descriptor[len(_PREFIX) :])
        if (
            not isinstance(name, str)
            or _EXPERT.fullmatch(name) is None
            or type(start) is not int
            or len(shape) != 3
            or len(full_shape) != 3
            or any(type(value) is not int or value <= 0 for value in full_shape)
            or full_shape[0] % owners
            or tuple(shape[1:]) != tuple(full_shape[1:])
            or not 0 <= start < start + shape[0] <= full_shape[0]
            or not owner * (full_shape[0] // owners)
            <= start
            < start + shape[0]
            <= (owner + 1) * (full_shape[0] // owners)
        ):
            raise ValueError("Invalid expert shard descriptor")
        parameter = self.model.get_parameter(name)
        if tuple(parameter.shape) != tuple(full_shape) or parameter.dtype != torch.bfloat16:
            raise ValueError("Expert descriptor does not match the native parameter")
        coverage.setdefault(name, []).append((start, start + shape[0]))
        return parameter.narrow(0, start, shape[0])

    def _check_expert_coverage(self, coverage: dict) -> None:
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            if not _EXPERT.fullmatch(name):
                continue
            end = 0
            for start, stop in sorted(coverage.get(name, [])):
                if start != end:
                    raise ValueError("Missing or overlapping native expert owner slices")
                end = stop
            if end != parameter.shape[0]:
                raise ValueError("Native expert owner slices do not cover every local expert")

    def _remember_storage(self) -> None:
        self.storage = {
            name: (p.data_ptr(), tuple(p.shape), p.dtype, p.device)
            for name, p in self.model.named_parameters(remove_duplicate=False)
        }

    def _validate_storage(self) -> None:
        if self.storage:
            current = {
                name: (p.data_ptr(), tuple(p.shape), p.dtype, p.device)
                for name, p in self.model.named_parameters(remove_duplicate=False)
            }
            if current != self.storage:
                raise ValueError("Inference parameter storage changed; restart the entire group")


def load_sharded_memory_weights(model: Any, weights: list[tuple[str, Any]]) -> None:
    """One engine supplies a separate control tensor for each TP worker."""
    from relax.distributed.weight_sync.memory_transport import receive_memory_update

    weights = list(weights)
    if len(weights) != 1 or weights[0][1].numel() or len(weights[0][0]) > 65536:
        raise ValueError("Sharded memory loader requires one bounded empty control tensor")
    control = json.loads(weights[0][0])
    receiver = getattr(model, "_relax_sharded_memory_receiver", None)
    if receiver is None:
        receiver = ShardedMemoryReceiver(model, gpu_budget_bytes=control["gpu_budget_bytes"])
        model._relax_sharded_memory_receiver = receiver
    elif receiver.gpu_budget_bytes != control["gpu_budget_bytes"]:
        raise ValueError("Receiver GPU budget changed")

    timings = {}

    def receive(descriptor: dict, member: str, apply: Any) -> None:
        receive_memory_update(
            descriptor,
            member,
            apply,
            timeout_s=control["timeout_s"],
            host_budget_bytes=control["host_budget_bytes"],
            private_buffers=True,
            timings=timings,
        )

    receiver.apply_group(control, receive)
    receiver.last_metrics["tcp_stages"] = timings
    logger.info("Sharded TCP receive phases: %s", json.dumps(timings, sort_keys=True))
