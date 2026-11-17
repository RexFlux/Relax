# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU raw-bit and admission regressions; these do not prove CUDA IPC/NCCL."""

import copy
import hashlib

import pytest
import torch

from relax.backends.sglang.proxy_weight_loader import _document, json_bytes, split_directory
from relax.distributed.weight_sync.gpu_payload import apply_raw_slice, validate_raw_frame
from relax.distributed.weight_sync.gpu_transport import plan_rounds, select_overlap
from relax.distributed.weight_sync.streaming_delta import (
    SnapshotBudget,
    StreamingDeltaEncoder,
    steady_state_snapshot_bytes,
)


def _commit(encoder):
    encoder.prepare_commit({"worker"}, {"worker"})
    encoder.commit()


@pytest.mark.parametrize(
    "dtype,word,bits",
    [
        (torch.bfloat16, torch.int16, [-32768, 0x7FC1, 0x7FC2]),
        (torch.float32, torch.int32, [-(2**31), 0x7F800001, 0x7FC00002]),
    ],
)
def test_raw_frames_preserve_nan_payload_signed_zero_cross_tensor_and_empty_delta(dtype, word, bits):
    encoder = StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=1024, gpu_payload=True)
    values = [("a", torch.zeros(32, dtype=dtype)), ("b", torch.zeros(32, dtype=dtype))]
    targets = [torch.empty_like(value) for _, value in values]
    for step in range(4):
        if step == 1:
            values[0][1].view(word)[-1] = bits[0]
            values[1][1].view(word)[:2] = torch.tensor(bits[1:], dtype=word)
        encoder.begin(full=step in (0, 3))
        for name, value in values:
            encoder.add(name, value)
        bucket = encoder.finish()[0]
        metadata = bucket.metadata()
        assert metadata["integrity"] == "nccl" and "sha256" not in metadata
        assert bucket.encoding == ("full" if step in (0, 3) else "sparse")
        if step == 2:
            assert bucket.changed == len(bucket.payload) == 0
        if step in (0, 3):
            assert bucket.payload.untyped_storage().data_ptr() == encoder._pending_snapshots[0][2].data_ptr()
        for index, target in enumerate(targets):
            apply_raw_slice(target, metadata, bucket.payload, tensor_index=index, offset=32 * index)
            assert torch.equal(target.view(torch.uint8), values[index][1].view(torch.uint8))
        _commit(encoder)
    encoder.close()
    assert encoder.budget.used_bytes == 0


def test_retained_sparse_budget_survives_scratch_and_steady_admission_covers_next_update():
    budget = SnapshotBudget(1 << 20)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=128, gpu_payload=True)
    values = [(str(i), torch.zeros(64, dtype=torch.bfloat16)) for i in range(10)]
    encoder.begin()
    for name, value in values:
        encoder.add(name, value)
    full = encoder.finish()
    estimate = steady_state_snapshot_bytes(iter(full))
    assert estimate == 3 * 1280 + 64 * 21
    _commit(encoder)
    encoder.begin()
    for name, value in values:
        value[:10] = 1
        encoder.add(name, value)
    sparse = encoder.finish()
    assert sum(len(bucket.payload) for bucket in sparse) == 600
    assert budget.used_bytes == 2 * 1280 + 600
    assert budget.peak_bytes <= estimate
    encoder.abort()
    assert budget.used_bytes == 1280
    encoder.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("positions", [[-1, 2], [2, 2], [3, 2], [2, 64]])
def test_invalid_gpu_sparse_positions_fail_before_any_write(positions):
    raw = torch.cat(
        [
            torch.tensor(positions, dtype=torch.int32).view(torch.uint8),
            torch.ones(2, dtype=torch.int16).view(torch.uint8),
        ]
    )
    metadata = dict(names=[("a", [64])], elements=64, changed=2, dtype="bfloat16", encoding="sparse", bytes=12)
    with pytest.raises(ValueError, match="Sparse positions"):
        validate_raw_frame(metadata, raw)


def test_rounds_preserve_recipient_order_alignment_and_full_fallback_before_transfer():
    frames = [
        dict(id=0, bytes=6, owner=0, recipients=["a", "b"]),
        dict(id=1, bytes=4, owner=1, recipients=["a"]),
        dict(id=2, bytes=0, owner=1, recipients=["a", "b"]),
    ]
    rounds = plan_rounds(frames, ("a", "b"), 10)
    assert len(rounds) == 2
    assert rounds[0]["a"]["bytes"] == 6
    assert [e["frame"] for r in rounds for e in r["a"]["entries"]] == [0, 1, 2]
    assert [e["frame"] for r in rounds for e in r["b"]["entries"]] == [0, 2]
    assert all(e["offset"] % 4 == 0 for r in rounds for b in r.values() for e in b["entries"])
    assert select_overlap(rounds, requested=True, full=True, budget_bytes=10)[0] is False
    assert select_overlap(rounds, requested=True, full=False, budget_bytes=10)[0] is True
    assert select_overlap(rounds, requested=True, full=False, budget_bytes=9)[0] is False
    with pytest.raises(ValueError, match="single-frame"):
        plan_rounds(frames, ("a", "b"), 5)


def test_schema_cache_and_dynamic_document_fragments_reject_corruption_and_reordering():
    manifest = dict(
        stream="x",
        version=0,
        base=-1,
        reset=True,
        buckets=[
            dict(
                names=[("a", [32])],
                dtype="bfloat16",
                elements=32,
                encoding="full",
                changed=32,
                bytes=64,
                integrity="nccl",
            )
        ],
    )
    schema, update = split_directory([manifest])
    assert "names" not in update[0]["buckets"][0]
    assert schema["streams"][0][0]["names"] == [("a", [32])]
    state = {}
    data = json_bytes(schema).decode()
    controls = [
        dict(kind="schema", id=hashlib.sha256(data.encode()).hexdigest(), index=i, count=2, data=part)
        for i, part in enumerate((data[:10], data[10:]))
    ]
    _document(state, controls[0])
    with pytest.raises(ValueError, match="already being"):
        _document(state, controls[0])
    _document(state, controls[1])
    assert json_bytes(state["schema"][1]) == json_bytes(schema)
    damaged = copy.deepcopy(controls)
    damaged[1]["data"] += "x"
    state = {}
    _document(state, damaged[0])
    with pytest.raises(ValueError, match="checksum"):
        _document(state, damaged[1])
