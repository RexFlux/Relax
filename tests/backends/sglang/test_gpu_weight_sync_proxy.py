# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in real NCCL peer initialization and same-GPU cross-process CUDA IPC."""

import gc
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
from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup


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


def _reply(pipe, timeout=40):
    assert pipe.poll(timeout), "component process exceeded its bounded deadline"
    return pipe.recv()


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
    consumer.start()
    identity = _reply(consumer_parent)
    source_parent, source_child = context.Pipe()
    proxy_parent, proxy_child = context.Pipe()
    source = context.Process(target=_source, args=(source_child, connection))
    proxy = context.Process(target=_child, args=(proxy_child,))
    source.start()
    proxy.start()
    proxy_parent.send(
        dict(workers=[_process_identity(consumer.pid)], device=1, connection=connection, rank=1, timeout_s=30)
    )
    try:
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
        for process in (consumer, source, proxy):
            if process.is_alive():
                process.terminate()
            process.join(5)
