# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-owned local TP receivers.

CUDA payloads never enter the parent actor.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from typing import Any

from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup, remaining

from .proxy_weight_loader import MAX_DOCUMENT_BYTES, PROXY_MEMORY_LOADER, json_bytes, split_directory


MAX_HTTP_BYTES = 65536


def _http_bytes(payload: dict) -> int:
    # requests' json= body uses these default separators/ASCII escaping.
    return len(json.dumps(payload, allow_nan=False).encode())


def _process_identity(pid: int) -> tuple[int, int]:
    import psutil

    return pid, int(psutil.Process(pid).create_time() * 1e6)


def _workers_exited(identities: list) -> bool:
    import psutil

    for pid, stamp in identities:
        try:
            process = psutil.Process(pid)
            if int(process.create_time() * 1e6) == stamp and process.status() != psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            pass
    return True


def _serialize(control: dict, tensor: Any = None) -> str:
    import torch
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
        monkey_patch_torch_reductions()
    return MultiprocessingSerializer.serialize(
        [(json_bytes(control).decode(), _EmptyControlTensor() if tensor is None else tensor)], output_str=True
    )


class _EmptyControlTensor:
    """Use Torch's original CPU recipe even in a CUDA-patched parent
    process."""

    def __reduce__(self):
        import torch
        from torch.multiprocessing import reductions

        reducer = getattr(reductions, "_reduce_tensor_original", reductions.reduce_tensor)
        return reducer(torch.empty(0))


class _CudaWireSize:
    """A serialization-only upper bound; it never exports CUDA storage.

    The supported Torch CUDA reducer has three IPC handles and signed-64-bit
    sizes/offsets. Use oversized distinct handles and UUID text so pickle's
    memoization cannot underestimate a real one-dimensional uint8 slab. Never
    deserialize this object. GPU component tests compare the bound to the exact
    reducer in the deployment image.
    """

    def __reduce__(self):
        import torch
        from sglang.srt.utils.patch_torch import _rebuild_cuda_tensor_modified

        upper = (1 << 63) - 1
        return _rebuild_cuda_tensor_modified, (
            torch.Tensor,
            torch.Size([upper]),
            (upper,),
            upper,
            torch.storage.TypedStorage,
            torch.uint8,
            "GPU-" + "x" * 256,
            bytes(range(64)),
            upper,
            upper,
            False,
            b"r" * 512,
            upper,
            bytes(range(64, 128)),
            True,
        )


class _StagingArena:
    """Reuse fenced storage even while NCCL/IPC retain inactive Python
    references."""

    def __init__(self, capacity: int, device: Any) -> None:
        import torch

        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Positive staging capacity required")
        self.storage = torch.empty(capacity, dtype=torch.uint8, device=device)
        self.free = [(0, capacity)]
        self.active: dict[int, tuple[int, int]] = {}

    def reserve(self, key: int, size: int) -> Any:
        if key in self.active or type(size) is not int or size < 0:
            raise ValueError("Invalid or repeated arena reservation")
        if not size:
            self.active[key] = (0, 0)
            return self.storage[:0]
        charged = (size + 3) // 4 * 4
        for index, (offset, available) in enumerate(self.free):
            if size <= available:
                charged = min(charged, available)
                self.active[key] = (offset, charged)
                self.free[index : index + 1] = [(offset + charged, available - charged)] if charged < available else []
                return self.storage[offset : offset + size]
        raise ValueError("Retained round storage exceeds the staging arena")

    def release(self, key: int) -> None:
        offset, size = self.active.pop(key)
        if size:
            blocks = sorted([*self.free, (offset, size)])
            self.free = []
            for start, length in blocks:
                if self.free and self.free[-1][0] + self.free[-1][1] == start:
                    previous, count = self.free.pop()
                    self.free.append((previous, count + length))
                else:
                    self.free.append((start, length))


def _child(connection: Connection) -> None:
    """The allocating process serializes IPC and survives ambiguous
    consumption."""
    import torch

    group, buffers, workers, arena = None, {}, [], None
    try:
        config = connection.recv()
        workers = config["workers"]
        torch.cuda.set_device(config["device"])
        device = torch.device("cuda", config["device"])
        group = GpuSyncGroup(config["connection"], rank=config["rank"], device=device, timeout_s=config["timeout_s"])
        properties = torch.cuda.get_device_properties(device)
        identity = dict(pid=os.getpid(), device=device.index, uuid=str(properties.uuid), rank=config["rank"])
        free, total = torch.cuda.mem_get_info(device)
        connection.send(
            dict(
                success=True,
                **identity,
                free_bytes=free,
                total_bytes=total,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                staging_capacity_bytes=0,
            )
        )
        while True:
            command = connection.recv()
            op = command["op"]
            try:
                if op == "receive":
                    key, batch = command["round"], command["batch"]
                    if key in buffers:
                        raise ValueError("A proxy round is already retained")
                    if arena is None:
                        free, _ = torch.cuda.mem_get_info(device)
                        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                        if command["budget"] > free + reusable:
                            raise ValueError("Proxy arena exceeds physically available memory")
                        arena = _StagingArena(command["budget"], device)
                    if arena.storage.numel() != command["budget"]:
                        raise ValueError("A live proxy arena cannot change capacity")
                    slab = arena.reserve(key, batch["bytes"])
                    slab, works = group.receive(batch, slab)
                    buffers[key] = dict(slab=slab, works=works, exported=False, complete=False)
                    del slab, works
                    result = dict(success=True, posted=True)
                elif op == "ready":
                    item = buffers[command["round"]]
                    group.fence(item["works"], deadline=command["deadline"])
                    item["complete"] = True
                    item["works"] = []
                    del item
                    result = dict(
                        success=True,
                        retained_payload_bytes=sum(v["slab"].numel() for v in buffers.values()),
                        staging_capacity_bytes=arena.storage.numel(),
                        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                        cuda_process_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        cuda_process_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    )
                elif op == "export":
                    item = buffers[command["round"]]
                    if not item["complete"]:
                        raise ValueError("Cannot export an incomplete NCCL receive")
                    # Mark before serialization: even an ambiguous send failure
                    # must not cause early producer destruction.
                    item["exported"] = True
                    result = dict(success=True, serialized=_serialize(command["control"], item["slab"]))
                    del item
                elif op == "release":
                    item = buffers[command["round"]]
                    if item["exported"] and command.get("consumed") is not True:
                        raise ValueError("Exported IPC slab lacks a worker consumption fence")
                    if not item["complete"]:
                        raise ValueError("Cannot release an in-flight NCCL receive")
                    del buffers[command["round"]]
                    del item
                    arena.release(command["round"])
                    torch.cuda.ipc_collect()
                    result = dict(success=True)
                elif op == "memory":
                    free, total = torch.cuda.mem_get_info(device)
                    result = dict(
                        success=True,
                        **identity,
                        free_bytes=free,
                        total_bytes=total,
                        allocated_bytes=torch.cuda.memory_allocated(device),
                        reserved_bytes=torch.cuda.memory_reserved(device),
                        staging_capacity_bytes=arena.storage.numel() if arena is not None else 0,
                        retained_payload_bytes=sum(v["slab"].numel() for v in buffers.values()),
                    )
                elif op == "close":
                    if buffers and not _workers_exited(workers):
                        raise RuntimeError("Retaining unresolved IPC/NCCL buffers until model worker exit")
                    connection.send(dict(success=True))
                    break
                else:
                    raise ValueError("Unknown proxy operation")
                connection.send(result)
            except BaseException as exc:
                connection.send(dict(success=False, error=f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as exc:
        try:
            connection.send(dict(success=False, error=f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        # Parent death/HTTP timeout is not a consumer fence. Keep every pending
        # producer alive until the original server tree actually exits.
        while buffers and not _workers_exited(workers):
            time.sleep(1)
        buffers.clear()
        if group is not None:
            group.close()
        connection.close()


class WeightSyncProxy:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.children: list[tuple[Any, Connection]] = []
        self.connection: dict | None = None
        self.identities: list[dict] = []
        self.schema_ids: tuple | None = None
        self.update_ids: list[str] = []
        self.batch = 0
        self.metrics: dict = {}

    def _receive_replies(self, *, deadline: float) -> list:
        results = []
        for _process, connection in self.children:
            if not connection.poll(remaining(deadline)):
                raise TimeoutError("Proxy subprocess did not complete before its deadline")
            result = connection.recv()
            if result.get("success") is not True:
                raise RuntimeError(f"GPU proxy subprocess failed: {result}")
            results.append(result)
        return results

    def _commands(self, commands: list[dict], *, deadline: float) -> list:
        if len(commands) != len(self.children):
            raise ValueError("Proxy commands must cover every TP process")
        for (_process, connection), command in zip(self.children, commands, strict=True):
            connection.send(command)
        return self._receive_replies(deadline=deadline)

    def join(self, connection: dict, ranks: list[int], *, timeout: float) -> list:
        from multiprocessing import Pipe

        import psutil

        if self.connection is not None:
            if self.connection != connection or [item["rank"] for item in self.identities] != ranks:
                raise ValueError("Proxy process group or member mapping changed")
            return self._commands([dict(op="memory")] * len(ranks), deadline=time.monotonic() + timeout)
        args = self.engine._memory_proxy_server_args
        if args["nnodes"] != 1 or args["tp_size"] != len(ranks) or self.children:
            raise ValueError("GPU proxy requires one fixed local process per actual TP GPU")
        root = psutil.Process(self.engine.process.pid)
        workers = [_process_identity(p.pid) for p in [root, *root.children(recursive=True)]]
        if len(workers) <= 1:
            raise ValueError("Cannot verify the existing SGLang worker process tree")
        deadline = time.monotonic() + timeout
        for tp, rank in enumerate(ranks):
            parent, child = Pipe(duplex=True)
            fd = child.fileno()
            process = subprocess.Popen(
                [sys.executable, "-m", __name__, str(fd)],
                pass_fds=(fd,),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            child.close()
            self.children.append((process, parent))
            parent.send(
                dict(
                    connection=connection,
                    rank=rank,
                    timeout_s=remaining(deadline),
                    workers=workers,
                    device=args["base_gpu_id"] + tp * args["gpu_id_step"],
                )
            )
        self.identities = self._receive_replies(deadline=deadline)
        if len({item["uuid"] for item in self.identities}) != len(ranks):
            raise ValueError("Proxy TP processes do not map to distinct physical GPUs")
        self.connection = connection
        return self.identities

    def receive(self, round_id: int, batches: list[dict], budget: int, *, timeout: float) -> list:
        return self._commands(
            [dict(op="receive", round=round_id, batch=b, budget=budget) for b in batches],
            deadline=time.monotonic() + timeout,
        )

    def ready(self, round_id: int, *, timeout: float) -> list:
        deadline = time.monotonic() + timeout
        return self._commands(
            [dict(op="ready", round=round_id, deadline=deadline)] * len(self.children), deadline=deadline
        )

    def _request(self, endpoint: str, payload: dict, *, deadline: float) -> None:
        size = _http_bytes(payload)
        if size > MAX_HTTP_BYTES:
            raise ValueError("Serialized whole-engine proxy HTTP body exceeds 64KiB")
        started = time.monotonic()
        response = self.engine._make_request(endpoint, payload, timeout=remaining(deadline))
        if not isinstance(response, dict) or response.get("success") is not True:
            raise RuntimeError(f"SGLang {endpoint} failed during GPU proxy installation")
        self.metrics["http_calls"] += 1
        self.metrics["control_bytes"] += size
        self.metrics["http_s"] += time.monotonic() - started

    @staticmethod
    def _payload(serialized: list[str], *, flush: bool = False) -> dict:
        return dict(serialized_named_tensors=serialized, load_format=PROXY_MEMORY_LOADER, flush_cache=flush)

    def _controls(self, controls: list[dict], *, deadline: float, flush: bool = False) -> None:
        self._request(
            "update_weights_from_tensor",
            self._payload([_serialize(c) for c in controls], flush=flush),
            deadline=deadline,
        )

    def _documents(self, kind: str, documents: list, *, deadline: float) -> list[str]:
        encoded = [json_bytes(d).decode() for d in documents]
        if any(len(data.encode()) > MAX_DOCUMENT_BYTES for data in encoded):
            raise ValueError("Proxy document exceeds its bounded worker directory capacity")
        identities = [hashlib.sha256(data.encode()).hexdigest() for data in encoded]
        # Decide the complete fragment set before publishing its first fragment.
        width = max(1, MAX_HTTP_BYTES // (8 * len(encoded)))
        while True:
            count = max((len(data) + width - 1) // width for data in encoded)
            controls = [
                [
                    dict(
                        op="document",
                        kind=kind,
                        id=identity,
                        index=i,
                        count=count,
                        data=data[i * width : (i + 1) * width],
                    )
                    for data, identity in zip(encoded, identities, strict=True)
                ]
                for i in range(count)
            ]
            if all(_http_bytes(self._payload([_serialize(c) for c in part])) <= MAX_HTTP_BYTES for part in controls):
                break
            if width == 1:
                raise ValueError("One proxy control fragment cannot fit the whole-engine HTTP budget")
            width = max(1, width // 2)
        for part in controls:
            self._controls(part, deadline=deadline)
            self.metrics["schema_calls" if kind == "schema" else "update_directory_calls"] += 1
        return identities

    def begin(self, controls: list[dict], manifests: list[list[dict]], *, timeout: float) -> dict:
        from .sglang_engine import _sglang_has_weight_update_session

        deadline = time.monotonic() + timeout
        self.metrics = dict(
            http_calls=0, control_bytes=0, http_s=0.0, schema_calls=0, update_directory_calls=0, consume_calls=0
        )
        if _sglang_has_weight_update_session():
            self._request("begin_weight_update", {}, deadline=deadline)
        split = [split_directory(items) for items in manifests]
        schemas, updates = [item[0] for item in split], [item[1] for item in split]
        ids = tuple(hashlib.sha256(json_bytes(item)).hexdigest() for item in schemas)
        if self.schema_ids != ids:
            self._documents("schema", schemas, deadline=deadline)
            self.schema_ids = ids
        self.update_ids = self._documents("update", updates, deadline=deadline)
        self.batch = 0
        self._controls(
            [
                dict(
                    op="begin",
                    **control,
                    schema_id=schema_id,
                    update_id=update_id,
                    device=identity["device"],
                    uuid=identity["uuid"],
                )
                for control, schema_id, update_id, identity in zip(
                    controls, ids, self.update_ids, self.identities, strict=True
                )
            ],
            deadline=deadline,
        )
        return dict(success=True)

    def consume(self, round_id: int, entries: list[list], *, timeout: float) -> dict:
        deadline = time.monotonic() + timeout

        def partition(parts: list[list]) -> list:
            controls = [
                dict(op="consume", update_id=identity, batch=(1 << 63) - 1, frames=frames)
                for identity, frames in zip(self.update_ids, parts, strict=True)
            ]
            preview = self._payload([_serialize(c, _CudaWireSize()) for c in controls])
            if _http_bytes(preview) <= MAX_HTTP_BYTES:
                return [parts]
            if max(map(len, parts)) <= 1:
                raise ValueError("One IPC frame cannot fit the whole-engine HTTP budget")
            return partition([items[: (len(items) + 1) // 2] for items in parts]) + partition(
                [items[(len(items) + 1) // 2 :] for items in parts]
            )

        # Decide all controls before the first real _share_cuda_ call. Discarded
        # speculative exports would create IPC refcounters with no consumer.
        for parts in partition(entries):
            controls = [
                dict(op="consume", update_id=identity, batch=self.batch, frames=frames)
                for identity, frames in zip(self.update_ids, parts, strict=True)
            ]
            replies = self._commands(
                [dict(op="export", round=round_id, control=control) for control in controls], deadline=deadline
            )
            payload = self._payload([item["serialized"] for item in replies])
            # A changed dependency reducer must fail closed and retain the
            # producer; never re-export or retry an already shared slab.
            self._request("update_weights_from_tensor", payload, deadline=deadline)
            self.metrics["consume_calls"] += 1
            self.batch += 1

        self._commands([dict(op="release", round=round_id, consumed=True)] * len(self.children), deadline=deadline)
        return dict(success=True)

    def finalize(self, update: str, *, timeout: float) -> dict:
        from .sglang_engine import _sglang_has_weight_update_session

        deadline = time.monotonic() + timeout
        self._controls(
            [dict(op="finalize", update_id=identity) for identity in self.update_ids], deadline=deadline, flush=True
        )
        if _sglang_has_weight_update_session():
            self._request("end_weight_update", {}, deadline=deadline)
        self._request("update_weight_version", dict(new_version=update, abort_all_requests=False), deadline=deadline)
        return dict(success=True, metrics=dict(self.metrics), proxies=self.identities)

    def close(self, *, timeout: float = 10) -> None:
        if not self.children:
            return
        deadline = time.monotonic() + timeout
        self._commands([dict(op="close")] * len(self.children), deadline=deadline)
        for process, connection in self.children:
            connection.close()
            process.wait(timeout=remaining(deadline))
        self.children.clear()


if __name__ == "__main__":
    _child(Connection(int(sys.argv[1])))
