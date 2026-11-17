# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded IPC controls feeding the existing HF/native receiver sessions."""

from __future__ import annotations

import hashlib
import json
from typing import Any


PROXY_MEMORY_LOADER = "relax.backends.sglang.proxy_weight_loader.load_proxy_memory_weights"
MAX_DOCUMENT_BYTES = 64 << 20


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def split_directory(manifests: list[dict]) -> tuple[dict, list[dict]]:
    """Cache immutable names/shapes once; send only per-update frame
    metadata."""
    directory, updates = [], []
    for manifest in manifests:
        directory.append(
            [{key: bucket[key] for key in ("names", "elements", "dtype")} for bucket in manifest["buckets"]]
        )
        updates.append(
            {
                **{key: value for key, value in manifest.items() if key != "buckets"},
                "buckets": [
                    {key: value for key, value in b.items() if key not in ("names", "elements", "dtype")}
                    for b in manifest["buckets"]
                ],
            }
        )
    return {"streams": directory}, updates


def _document(state: dict, control: dict) -> None:
    kind, identity = control["kind"], control["id"]
    if kind not in ("schema", "update") or type(control["count"]) is not int or not 0 < control["count"] <= 65536:
        raise ValueError("Invalid proxy document identity")
    index, data = control["index"], control["data"]
    if type(index) is not int or not 0 <= index < control["count"] or not isinstance(data, str):
        raise ValueError("Invalid proxy document fragment")
    pending = state.setdefault("fragments", {})
    if index == 0:
        if kind in pending:
            raise ValueError("A proxy document is already being received")
        pending[kind] = dict(id=identity, count=control["count"], next=0, parts=[], bytes=0)
    item = pending[kind]
    if (item["id"], item["count"], item["next"]) != (identity, control["count"], index):
        raise ValueError("Proxy fragments must arrive exactly once in order")
    item["bytes"] += len(data.encode())
    if item["bytes"] > MAX_DOCUMENT_BYTES:
        raise ValueError("Proxy control directory exceeds its memory bound")
    item["parts"].append(data)
    item["next"] += 1
    if item["next"] == item["count"]:
        encoded = "".join(item["parts"]).encode()
        if hashlib.sha256(encoded).hexdigest() != identity:
            raise ValueError("Proxy document checksum mismatch")
        state[kind] = (identity, json.loads(encoded))
        del pending[kind]


def _begin(model: Any, state: dict, control: dict) -> None:
    import torch

    from .hf_memory_loader import HFMemoryReceiver
    from .sharded_memory_loader import ShardedMemoryReceiver

    if state.get("active") is not None or state.get("failed"):
        raise ValueError("Proxy session is active or failed; discard the failed group")
    directory = state["schema"]
    updates = state["update"]
    if directory[0] != control["schema_id"] or updates[0] != control["update_id"]:
        raise ValueError("Proxy begin references an incomplete or stale directory")
    static, dynamic = directory[1]["streams"], updates[1]
    if len(static) != len(dynamic) or not static:
        raise ValueError("Proxy stream directory count mismatch")
    manifests = []
    for schema, update in zip(static, dynamic, strict=True):
        if len(schema) != len(update["buckets"]):
            raise ValueError("Proxy frame directory count mismatch")
        buckets = []
        for immutable, mutable in zip(schema, update["buckets"], strict=True):
            if set(immutable) & set(mutable) or mutable.get("integrity") != "nccl":
                raise ValueError("Proxy frame metadata changes its immutable directory")
            buckets.append(immutable | mutable)
        manifests.append(update | {"buckets": buckets})
    device = next(model.parameters()).device
    if device.type != "cuda" or device.index != control["device"]:
        raise ValueError("Proxy and model worker CUDA ordinals differ")
    actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if actual_uuid != control["uuid"]:
        raise ValueError("Proxy and model worker physical GPU UUIDs differ")
    adapter = control["adapter"]
    if adapter == "hf":
        budgets = control["budgets"]
        receiver = getattr(model, "_relax_hf_memory_receiver", None)
        if receiver is None:
            receiver = HFMemoryReceiver(model, **budgets)
            model._relax_hf_memory_receiver = receiver
        elif any(getattr(receiver, key) != value for key, value in budgets.items()):
            raise ValueError("Proxy HF receiver budgets changed")
        if len(manifests) != 1 or receiver.loader_workspace_budget_bytes <= 0:
            raise ValueError("Proxy HF requires one stream and a positive workspace reserve")
        plan = control["plan_fingerprint"]
        if (
            not isinstance(plan, str)
            or len(plan) != 64
            or any(c not in "0123456789abcdef" for c in plan)
            or manifests[0].get("group_update") != plan
            or (receiver.stream == manifests[0]["stream"] and receiver.plan_fingerprint not in (None, plan))
        ):
            raise ValueError("Proxy HF synchronization plan changed or disagrees with its manifest")
        receiver.begin(manifests[0])
    elif adapter == "native":
        receiver = getattr(model, "_relax_sharded_memory_receiver", None)
        if receiver is None:
            receiver = ShardedMemoryReceiver(model, gpu_budget_bytes=control["budgets"]["gpu_budget_bytes"])
            model._relax_sharded_memory_receiver = receiver
        elif receiver.gpu_budget_bytes != control["budgets"]["gpu_budget_bytes"]:
            raise ValueError("Proxy native receiver budget changed")
        receiver.begin(control["group"])
        for index, manifest in enumerate(manifests):
            receiver.begin_stream(index, manifest)
    else:
        raise ValueError("Unsupported proxy model adapter")
    state["active"] = dict(
        id=control["update_id"],
        adapter=adapter,
        receiver=receiver,
        manifests=manifests,
        batch=0,
        plan_fingerprint=control.get("plan_fingerprint"),
    )


def load_proxy_memory_weights(model: Any, weights: list[tuple[str, Any]]) -> None:
    """Successful consume returns only after all imported slab reads are
    fenced."""
    import torch

    state = getattr(model, "_relax_proxy_install", None)
    if state is None:
        state = {}
        model._relax_proxy_install = state
    try:
        if len(weights) != 1 or len(weights[0][0].encode()) > 65536:
            raise ValueError("Proxy loader requires one bounded control and slab")
        control, raw = json.loads(weights[0][0]), weights[0][1]
        op = control["op"]
        if state.get("failed"):
            raise ValueError("Failed proxy session cannot resume")
        if op == "document":
            if raw.numel() or state.get("active") is not None:
                raise ValueError("Documents cannot mutate an active proxy session")
            _document(state, control)
        elif op == "begin":
            if raw.numel():
                raise ValueError("Proxy begin must not carry a payload")
            _begin(model, state, control)
        elif op in ("consume", "finalize"):
            active = state.get("active")
            if active is None or active["id"] != control["update_id"]:
                raise ValueError("Proxy IPC update identity mismatch")
            receiver = active["receiver"]
            if op == "consume":
                if control["batch"] != active["batch"]:
                    raise ValueError("Proxy batches must arrive exactly once in order")
                end = 0
                for stream, bucket, offset, size in control["frames"]:
                    if any(type(v) is not int or v < 0 for v in (stream, bucket, offset, size)):
                        raise ValueError("Invalid proxy frame slice")
                    if offset % 4 or offset < end or offset + size > raw.numel():
                        raise ValueError("Proxy frame slices overlap or exceed their slab")
                    if size != active["manifests"][stream]["buckets"][bucket]["bytes"]:
                        raise ValueError("Proxy frame slice size disagrees with its manifest")
                    frame = raw[offset : offset + size]
                    if active["adapter"] == "hf":
                        if stream != 0:
                            raise ValueError("HF frame has a native stream index")
                        receiver.consume(bucket, frame)
                    else:
                        receiver.consume(stream, bucket, frame)
                    end = offset + size
                # ModelRunner holds the input list until this function returns.
                # Its caller must also discard the deserialized list before its
                # HTTP success; fixed-image source inspection and IPC tests gate deployment.
                torch.cuda.synchronize(next(model.parameters()).device)
                active["batch"] += 1
            else:
                if raw.numel():
                    raise ValueError("Proxy finalize must not carry a payload")
                receiver.finalize()
                if active["adapter"] == "hf":
                    receiver.plan_fingerprint = active["plan_fingerprint"]
                state["active"] = None
                state.pop("update", None)
        else:
            raise ValueError("Unknown proxy loader operation")
    except BaseException:
        state["failed"] = True
        active = state.get("active")
        if active is not None:
            active["receiver"].poison()
        raise
