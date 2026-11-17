# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real serializer/control integration without creating a CUDA context."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_fenced_staging_arena_reuses_storage_despite_inactive_retained_views():
    import pytest
    import torch

    from relax.backends.sglang.weight_sync_proxy import _StagingArena

    arena = _StagingArena(12, "cpu")
    first = arena.reserve(0, 6)
    first.fill_(17)
    second = arena.reserve(1, 4)
    assert second.storage_offset() == 8
    with pytest.raises(ValueError, match="exceeds"):
        arena.reserve(2, 1)
    empty = arena.reserve(3, 0)
    assert empty.numel() == 0
    arena.release(3)
    arena.release(0)
    replacement = arena.reserve(4, 6)
    assert replacement.data_ptr() == first.data_ptr()
    replacement.fill_(29)
    assert torch.equal(first, replacement)  # Old references stay valid but have no live consumer after its fence.
    arena.release(1)
    arena.release(4)
    assert arena.free == [(0, 12)] and not arena.active
    odd = _StagingArena(7, "cpu")
    assert odd.reserve(0, 7).numel() == 7
    odd.release(0)
    assert odd.free == [(0, 7)]


def test_fragmented_controls_round_trip_before_and_after_sglang_cuda_reducer_patch():
    # SGLang patches the global Torch reducer. One isolated CPU process is
    # necessary to exercise that real dependency interaction without changing
    # reduction behavior for unrelated tests in this pytest process.
    code = r"""
import hashlib,json,requests
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from relax.backends.sglang.proxy_weight_loader import _document,json_bytes
from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy,_serialize,_CudaWireSize,_http_bytes

class Engine:
    def __init__(self):
        self.states=[{},{}]
        self.bodies=[]
    def _make_request(self,endpoint,payload,timeout):
        assert endpoint=='update_weights_from_tensor'
        prepared=requests.Request('POST','http://localhost/control',json=payload).prepare()
        assert len(prepared.body)==_http_bytes(payload)<=65536
        self.bodies.append(len(prepared.body))
        for state,serialized in zip(self.states,payload['serialized_named_tensors'],strict=True):
            weights=MultiprocessingSerializer.deserialize(serialized)
            control,empty=weights[0]
            assert empty.numel()==0 and empty.device.type=='cpu'
            _document(state,json.loads(control))
        return {'success':True}

for patched in [False,True]:
    if patched:monkey_patch_torch_reductions()
    serialized=_serialize({'op':'noop'})
    assert MultiprocessingSerializer.deserialize(serialized)[0][1].numel()==0
    # Preview consists only of reducer metadata. It must never be deserialized.
    assert _http_bytes(WeightSyncProxy._payload([_serialize({'op':'consume'},_CudaWireSize())]))<65536
    engine=Engine();proxy=WeightSyncProxy(engine)
    proxy.metrics=dict(http_calls=0,control_bytes=0,http_s=0.,schema_calls=0,update_directory_calls=0)
    documents=[{'names':['layer.'+str(i)+'x'*40 for i in range(1400)]}, {'names':['unicode_\u4e2d'*8]*600}]
    identities=proxy._documents('schema',documents,deadline=__import__('time').monotonic()+30)
    assert len(engine.bodies)>1 and proxy.metrics['schema_calls']==len(engine.bodies)
    for state,document,identity in zip(engine.states,documents,identities,strict=True):
        assert state['schema'][0]==identity==hashlib.sha256(json_bytes(document)).hexdigest()
        assert state['schema'][1]==document
print(json.dumps({'status':'PASS','patched_and_unpatched':True,'gpu_context_created':False}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "PASS"
