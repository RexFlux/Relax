# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in real NCCL peer initialization and same-GPU cross-process CUDA IPC."""

import gc
import json
import os
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from relax.backends.sglang.weight_sync_proxy import (
    WeightSyncProxy,
    _child,
    _CudaWireSize,
    _http_bytes,
    _process_identity,
    _serialize,
)
from relax.distributed.weight_sync.gpu_payload import apply_raw_slice
from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup, plan_rounds
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder
from relax.utils.logging_utils import get_logger


def _source(pipe, connection):
    torch.cuda.set_device(0)
    group = GpuSyncGroup(connection, rank=0, device=torch.device("cuda", 0), timeout_s=30)
    raw = torch.arange(256, dtype=torch.uint8, device="cuda:0")
    pipe.send("joined")
    while True:
        command = pipe.recv()
        if command == "send":
            batch = {"worker": dict(entries=[dict(frame=0, offset=0, bytes=256, owner=0)], bytes=256)}
            group.send([dict(id=0, owner=0)], batch, {0: raw}, {"worker": 1}, deadline=time.monotonic() + 30)
            pipe.send("sent")
        else:
            group.close()
            pipe.send("closed")
            return


def _consumer(pipe):
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.cuda.set_device(1)
    monkey_patch_torch_reductions()
    pipe.send(dict(pid=os.getpid(), uuid=str(torch.cuda.get_device_properties(1).uuid)))
    while True:
        serialized = pipe.recv()
        if serialized is None:
            return
        weights = MultiprocessingSerializer.deserialize(serialized)
        control, raw = weights[0]
        assert raw.device == torch.device("cuda", 1)
        assert torch.equal(raw.cpu(), torch.arange(256, dtype=torch.uint8))
        torch.cuda.synchronize(1)
        del control, raw, weights
        gc.collect()
        torch.cuda.ipc_collect()
        pipe.send("consumed_and_fenced")


def _arena_source(pipe, connection, capacity):
    torch.cuda.set_device(0)
    group = GpuSyncGroup(connection, rank=0, device=torch.device("cuda", 0), timeout_s=60)
    storage = torch.empty(capacity, device="cuda:0", dtype=torch.uint8)
    pipe.send("joined")
    while True:
        command = pipe.recv()
        if command is None:
            group.close()
            return
        size, value = command
        raw = storage[:size].fill_(value)
        batch = dict(entries=[dict(frame=0, offset=0, bytes=size, owner=0)], bytes=size)
        group.send(
            [dict(id=0, owner=0)], {"receiver": batch}, {0: raw}, {"receiver": 1}, deadline=time.monotonic() + 60
        )
        pipe.send("sent")


def _arena_consumer(pipe):
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.cuda.set_device(1)
    monkey_patch_torch_reductions()
    pipe.send(dict(pid=os.getpid(), uuid=str(torch.cuda.get_device_properties(1).uuid)))
    inactive = []
    while True:
        serialized = pipe.recv()
        if serialized is None:
            return
        weights = MultiprocessingSerializer.deserialize(serialized)
        control, raw = weights[0]
        value = json.loads(control)["value"]
        assert bool(torch.all(raw == value).cpu())
        torch.cuda.synchronize(1)
        # These objects deliberately survive their consumption fence. No
        # subsequent operation reads them; a producer may safely reuse storage.
        inactive.append(raw)
        del weights
        pipe.send("consumed_and_fenced")


def _reply(pipe, timeout=40):
    assert pipe.poll(timeout), "component process exceeded its bounded deadline"
    return pipe.recv()


def _cleanup(consumers, others):
    # Consumer exit is a prerequisite for force-destroying an IPC producer.
    # If even SIGKILL cannot finish a consumer, preserve the producer and fail.
    for processes in (consumers, others):
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join(3)
        assert not any(process.is_alive() for process in processes), "Unexited consumer or worker; retain IPC producer"


def _multi_source(pipe, connection, rank, frames, receiver_ranks):
    torch.cuda.set_device(rank)
    group = GpuSyncGroup(connection, rank=rank, device=torch.device("cuda", rank), timeout_s=60)
    payloads = {
        frame["id"]: ((torch.arange(frame["bytes"], device=f"cuda:{rank}") + rank * 37 + frame["id"] * 11) % 256).to(
            torch.uint8
        )
        for frame in frames
        if frame["owner"] == rank
    }
    pipe.send("joined")
    while True:
        batch = pipe.recv()
        if batch is None:
            group.close()
            pipe.send("closed")
            return
        group.send(frames, batch, payloads, receiver_ranks, deadline=time.monotonic() + 60)
        pipe.send("sent")


def _multi_consumer(pipe, device):
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.cuda.set_device(device)
    monkey_patch_torch_reductions()
    pipe.send(dict(pid=os.getpid(), uuid=str(torch.cuda.get_device_properties(device).uuid)))
    while True:
        serialized = pipe.recv()
        if serialized is None:
            return
        weights = MultiprocessingSerializer.deserialize(serialized)
        encoded, raw = weights[0]
        control = json.loads(encoded)
        assert raw.device == torch.device("cuda", device)
        for frame, owner, offset, size in control["frames"]:
            expected = bytes((index + owner * 37 + frame * 11) % 256 for index in range(size))
            assert raw[offset : offset + size].cpu().numpy().tobytes() == expected
        torch.cuda.synchronize(device)
        del encoded, control, raw, weights
        gc.collect()
        torch.cuda.ipc_collect()
        pipe.send("consumed_and_fenced")


@pytest.mark.skipif(os.environ.get("RELAX_RUN_GPU_PROXY_COMPONENT") != "1", reason="requires explicit GPU ownership")
def test_first_nccl_pair_and_cuda_ipc_producer_wait_for_real_consumer_fence():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two real GPUs, with proxy and consumer sharing GPU1")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    connection = dict(init_method=f"tcp://127.0.0.1:{port}", world_size=2, group_name=f"proxy-component-{os.getpid()}")
    context = mp.get_context("spawn")
    consumer_parent, consumer_child = context.Pipe()
    consumer = context.Process(target=_consumer, args=(consumer_child,))
    source_parent, source_child = context.Pipe()
    proxy_parent, proxy_child = context.Pipe()
    source = context.Process(target=_source, args=(source_child, connection))
    proxy = context.Process(target=_child, args=(proxy_child,))
    processes = []
    try:
        consumer.start()
        processes.append(consumer)
        identity = _reply(consumer_parent)
        source.start()
        processes.append(source)
        proxy.start()
        processes.append(proxy)
        proxy_parent.send(
            dict(workers=[_process_identity(consumer.pid)], device=1, connection=connection, rank=1, timeout_s=30)
        )
        assert _reply(source_parent) == "joined"
        joined = _reply(proxy_parent)
        assert joined["success"] and joined["uuid"] == identity["uuid"]
        batch = dict(bytes=256, entries=[dict(frame=0, offset=0, bytes=256, owner=0)])
        # First peer initialization must overlap, rather than waiting for irecv
        # to return before the source enters isend.
        proxy_parent.send(dict(op="receive", round=0, batch=batch, budget=256))
        source_parent.send("send")
        assert _reply(source_parent) == "sent"
        assert _reply(proxy_parent)["posted"]
        proxy_parent.send(dict(op="ready", round=0, deadline=time.monotonic() + 30))
        assert _reply(proxy_parent)["success"]
        control = dict(op="consume", update_id="x", batch=0, frames=[[0, 0, 0, 256]])
        preview = _http_bytes(WeightSyncProxy._payload([_serialize(control, _CudaWireSize())]))
        proxy_parent.send(dict(op="export", round=0, control=control))
        exported = _reply(proxy_parent)
        assert exported["success"]
        actual = _http_bytes(WeightSyncProxy._payload([exported["serialized"]]))
        assert actual <= preview <= 65536
        proxy_parent.send(dict(op="release", round=0, consumed=False))
        assert _reply(proxy_parent)["success"] is False
        assert proxy.is_alive()
        consumer_parent.send(exported["serialized"])
        assert _reply(consumer_parent) == "consumed_and_fenced"
        proxy_parent.send(dict(op="release", round=0, consumed=True))
        assert _reply(proxy_parent)["success"]
        source_parent.send("close")
        proxy_parent.send(dict(op="close"))
        assert _reply(proxy_parent)["success"]
        assert _reply(source_parent) == "closed"
        consumer_parent.send(None)
        for process in (source, proxy, consumer):
            process.join(10)
            assert process.exitcode == 0
    finally:
        # Terminate only these isolated test processes, never shared Ray jobs.
        _cleanup([p for p in processes if p is consumer], [p for p in processes if p is not consumer])


@pytest.mark.skipif(os.environ.get("RELAX_RUN_GPU_PROXY_COMPONENT") != "1", reason="requires explicit GPU ownership")
def test_two_sources_route_multiple_rounds_to_two_ipc_consumers_including_empty_batches():
    if torch.cuda.device_count() < 4:
        pytest.skip("requires two source GPUs and two receiver GPUs")
    context = mp.get_context("spawn")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    connection = dict(init_method=f"tcp://127.0.0.1:{port}", world_size=4, group_name=f"proxy-multi-{os.getpid()}")
    members = ("r0", "r1")
    receiver_ranks = {"r0": 2, "r1": 3}
    frames = [
        dict(id=0, owner=0, bytes=256, recipients=list(members)),
        dict(id=1, owner=0, bytes=22, recipients=["r1"]),
        dict(id=2, owner=1, bytes=26, recipients=["r0"]),
        dict(id=3, owner=1, bytes=258, recipients=list(members)),
        dict(id=4, owner=1, bytes=16, recipients=["r1"]),
        dict(id=5, owner=1, bytes=0, recipients=list(members)),
        dict(id=6, owner=0, bytes=390, recipients=["r1"]),
    ]
    rounds = plan_rounds(frames, members, 400)
    assert len(rounds) == 3 and not rounds[-1]["r0"]["entries"]
    assert {entry["owner"] for entry in rounds[0]["r0"]["entries"] if entry["bytes"]} == {0, 1}
    processes, consumer_processes, sources, proxies, consumers = [], [], [], [], []
    try:
        for index in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_multi_consumer, args=(child, index + 2))
            process.start()
            processes.append(process)
            consumer_processes.append(process)
            consumers.append((parent, process, _reply(parent, 60)))
        for rank in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_multi_source, args=(child, connection, rank, frames, receiver_ranks))
            process.start()
            processes.append(process)
            sources.append(parent)
        for index, (_consumer_pipe, consumer, identity) in enumerate(consumers):
            parent, child = context.Pipe()
            process = context.Process(target=_child, args=(child,))
            process.start()
            processes.append(process)
            parent.send(
                dict(
                    workers=[_process_identity(consumer.pid)],
                    device=index + 2,
                    connection=connection,
                    rank=index + 2,
                    timeout_s=60,
                )
            )
            proxies.append(parent)
        for source in sources:
            assert _reply(source, 70) == "joined"
        for proxy, (_consumer_pipe, _consumer, identity) in zip(proxies, consumers, strict=True):
            joined = _reply(proxy, 70)
            assert joined["success"] and joined["uuid"] == identity["uuid"]
        for round_id, batches in enumerate(rounds):
            # Dispatch every receive before entering any source send. Lazy peer
            # communicator initialization must make progress on both sides.
            for member, proxy in zip(members, proxies, strict=True):
                proxy.send(dict(op="receive", round=round_id, batch=batches[member], budget=400))
            for source in sources:
                source.send(batches)
            for source in sources:
                assert _reply(source, 70) == "sent"
            for proxy in proxies:
                assert _reply(proxy, 70)["posted"]
                proxy.send(dict(op="ready", round=round_id, deadline=time.monotonic() + 60))
            for proxy in proxies:
                assert _reply(proxy, 70)["success"]
            for member, proxy, (consumer_pipe, _consumer, _identity) in zip(members, proxies, consumers, strict=True):
                entries = batches[member]["entries"]
                control = dict(
                    op="consume",
                    batch=round_id,
                    frames=[[entry["frame"], entry["owner"], entry["offset"], entry["bytes"]] for entry in entries],
                )
                proxy.send(dict(op="export", round=round_id, control=control))
                exported = _reply(proxy, 70)
                assert exported["success"]
                consumer_pipe.send(exported["serialized"])
            for consumer_pipe, _consumer, _identity in consumers:
                assert _reply(consumer_pipe, 70) == "consumed_and_fenced"
            for proxy in proxies:
                proxy.send(dict(op="release", round=round_id, consumed=True))
                assert _reply(proxy, 70)["success"]
        for source in sources:
            source.send(None)
        for proxy in proxies:
            proxy.send(dict(op="close"))
        for source in sources:
            assert _reply(source, 70) == "closed"
        for proxy in proxies:
            assert _reply(proxy, 70)["success"]
        for consumer_pipe, _consumer, _identity in consumers:
            consumer_pipe.send(None)
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
    finally:
        _cleanup(consumer_processes, [p for p in processes if p not in consumer_processes])


@pytest.mark.skipif(os.environ.get("RELAX_RUN_GPU_PROXY_COMPONENT") != "1", reason="requires explicit GPU ownership")
def test_staging_arena_gpu_peak_is_bounded_with_old_ipc_references_retained():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires one source and a same-GPU proxy/consumer pair")
    capacity = 64 << 20
    context = mp.get_context("spawn")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    connection = dict(init_method=f"tcp://127.0.0.1:{port}", world_size=2, group_name=f"proxy-arena-{os.getpid()}")
    consumer_parent, consumer_child = context.Pipe()
    source_parent, source_child = context.Pipe()
    proxy_parent, proxy_child = context.Pipe()
    consumer = context.Process(target=_arena_consumer, args=(consumer_child,))
    source = context.Process(target=_arena_source, args=(source_child, connection, capacity))
    proxy = context.Process(target=_child, args=(proxy_child,))
    processes = []
    samples = []
    try:
        consumer.start()
        processes.append(consumer)
        identity = _reply(consumer_parent)
        source.start()
        processes.append(source)
        proxy.start()
        processes.append(proxy)
        proxy_parent.send(
            dict(connection=connection, rank=1, device=1, timeout_s=60, workers=[_process_identity(identity["pid"])])
        )
        assert _reply(source_parent) == "joined"
        assert _reply(proxy_parent)["uuid"] == identity["uuid"]
        for round_id in range(12):
            size, value = (48 << 20 if round_id % 3 != 2 else 16 << 20), round_id + 1
            batch = dict(bytes=size, entries=[dict(frame=0, offset=0, bytes=size, owner=0)])
            proxy_parent.send(dict(op="receive", round=round_id, batch=batch, budget=capacity))
            source_parent.send((size, value))
            assert _reply(proxy_parent)["success"]
            assert _reply(source_parent) == "sent"
            proxy_parent.send(dict(op="ready", round=round_id, deadline=time.monotonic() + 60))
            ready = _reply(proxy_parent)
            assert ready["retained_payload_bytes"] == size
            assert ready["staging_capacity_bytes"] == capacity
            assert ready["cuda_process_peak_allocated_bytes"] <= capacity + (4 << 20)
            proxy_parent.send(dict(op="export", round=round_id, control=dict(value=value)))
            consumer_parent.send(_reply(proxy_parent)["serialized"])
            assert _reply(consumer_parent) == "consumed_and_fenced"
            proxy_parent.send(dict(op="release", round=round_id, consumed=True))
            assert _reply(proxy_parent)["success"]
            proxy_parent.send(dict(op="memory"))
            released = _reply(proxy_parent)
            assert released["retained_payload_bytes"] == 0
            assert released["allocated_bytes"] <= capacity + (4 << 20)
            samples.append(dict(round=round_id, ready=ready, released=released))
        get_logger(__name__).info("staging_arena_gpu_samples=%s", json.dumps(samples))
        consumer_parent.send(None)
        consumer.join(10)
        assert consumer.exitcode == 0
        proxy_parent.send(dict(op="close"))
        assert _reply(proxy_parent)["success"]
        source_parent.send(None)
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
    finally:
        _cleanup([p for p in processes if p is consumer], [p for p in processes if p is not consumer])


@pytest.mark.skipif(os.environ.get("RELAX_RUN_GPU_PROXY_COMPONENT") != "1", reason="requires explicit GPU ownership")
@pytest.mark.parametrize(
    "dtype,word,bits",
    [
        (torch.bfloat16, torch.int16, [-32768, 0x7FC1, 0x7FC2]),
        (torch.float32, torch.int32, [-(2**31), 0x7F800001, 0x7FC00002]),
    ],
)
def test_cuda_raw_codec_sparse_edges_nan_payload_and_full_refresh(dtype, word, bits):
    if torch.cuda.device_count() < 1:
        pytest.skip("requires one owned CUDA GPU")
    device = torch.device("cuda", 0)
    budget = SnapshotBudget(1 << 20)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=512, gpu_payload=True)
    values = [("a", torch.zeros(64, dtype=dtype, device=device)), ("b", torch.zeros(64, dtype=dtype, device=device))]
    targets = [torch.empty_like(tensor) for _, tensor in values]
    try:
        for step in range(4):
            if step == 1:
                values[0][1].view(word)[-1] = bits[0]
                values[1][1].view(word)[:2] = torch.tensor(bits[1:], dtype=word, device=device)
            encoder.begin(full=step in (0, 3))
            for name, tensor in values:
                encoder.add(name, tensor)
            bucket = encoder.finish()[0]
            metadata = bucket.metadata()
            assert bucket.payload.device == device
            assert bucket.encoding == ("full" if step in (0, 3) else "sparse")
            if step == 1:
                assert metadata["sparse_boundaries"] == [0, 1, 3]
            if step == 2:
                assert bucket.changed == len(bucket.payload) == 0
            for index, target in enumerate(targets):
                apply_raw_slice(target, metadata, bucket.payload, tensor_index=index, offset=64 * index)
            torch.cuda.synchronize(device)
            for target, (_name, expected) in zip(targets, values, strict=True):
                assert torch.equal(target.view(torch.uint8).cpu(), expected.view(torch.uint8).cpu())
            encoder.prepare_commit({"consumer"}, {"consumer"})
            encoder.commit()
    finally:
        encoder.close()
    assert budget.used_bytes == 0
