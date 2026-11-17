# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Model-independent raw-bit frames shared by GPU IPC and TCP adapters."""

from __future__ import annotations

import math
from typing import Any

from .memory_delta import validate_bucket_directory


def sparse_boundaries(metadata: dict, raw: Any) -> list[int]:
    """Copy only O(tensor-count) offsets to control metadata, never
    payloads."""
    import torch

    if metadata["encoding"] != "sparse":
        return []
    offsets = [0]
    for _name, shape in metadata["names"]:
        offsets.append(offsets[-1] + math.prod(shape))
    positions = raw[: metadata["changed"] * 4].view(torch.int32)
    boundaries = torch.tensor(offsets, dtype=torch.int32, device=raw.device)
    # This readiness boundary deliberately synchronizes small metadata. Dynamic
    # nonzero also synchronizes; this path does not claim zero host syncs.
    return torch.searchsorted(positions, boundaries).cpu().tolist()


def validate_raw_frame(metadata: dict, raw: Any) -> None:
    """Validate at the receive readiness fence before exposing any GPU
    writes."""
    import torch

    validate_bucket_directory(metadata, max_bytes=metadata["bytes"])
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous() or raw.numel() != metadata["bytes"]:
        raise ValueError("Raw frame must be a contiguous uint8 payload of the declared size")
    if metadata["encoding"] == "sparse":
        count = metadata["changed"]
        positions = raw[: count * 4].view(torch.int32)
        if count:
            invalid = (positions[0] < 0) | (positions[-1] >= metadata["elements"])
            invalid = invalid | torch.any(positions[1:] <= positions[:-1])
            if bool(invalid.cpu()):
                raise ValueError("Sparse positions must be unique, sorted and in bounds")
        if "sparse_boundaries" in metadata and metadata["sparse_boundaries"] != sparse_boundaries(metadata, raw):
            raise ValueError("Sparse tensor boundaries disagree with the payload")


def payload_to_device(payload: Any, device: Any) -> Any:
    import torch

    if isinstance(payload, torch.Tensor):
        if payload.device != torch.device(device):
            raise ValueError("IPC payload and target must be on the same device")
        return payload
    if not payload:
        return torch.empty(0, dtype=torch.uint8, device=device)
    host = payload if isinstance(payload, bytearray) else bytearray(payload)
    return torch.frombuffer(host, dtype=torch.uint8).to(device)


def apply_raw_slice(target: Any, metadata: dict, raw: Any, *, tensor_index: int, offset: int) -> None:
    """Patch an already validated target view without changing floating
    bits."""
    import torch

    width = target.element_size()
    words = torch.int16 if width == 2 else torch.int32
    flat = target.view(words).reshape(-1)
    if metadata["encoding"] == "full":
        flat.copy_(raw[offset * width : (offset + target.numel()) * width].view(words))
    elif metadata["changed"]:
        count = metadata["changed"]
        left, right = metadata["sparse_boundaries"][tensor_index : tensor_index + 2]
        if left != right:
            indices = raw[: count * 4].view(torch.int32)[left:right].to(torch.int64) - offset
            values = raw[count * 4 :].view(words)[left:right]
            flat.index_copy_(0, indices, values)
