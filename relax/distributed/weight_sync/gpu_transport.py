# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-plan NCCL P2P and bounded receiver slabs; no model or Ray
ownership."""

from __future__ import annotations

from datetime import timedelta
from time import monotonic
from typing import Any


def remaining(deadline: float) -> float:
    value = deadline - monotonic()
    if value <= 0:
        raise TimeoutError("GPU weight synchronization deadline exceeded")
    return value


def plan_rounds(frames: list[dict], members: tuple[str, ...], budget_bytes: int) -> list[dict]:
    """Greedily pack each recipient's ordered frames into four-byte aligned
    slabs."""
    if type(budget_bytes) is not int or budget_bytes <= 0 or len(set(members)) != len(members) or not members:
        raise ValueError("A positive staging budget and distinct recipients are required")
    pending = {member: [] for member in members}
    for index, frame in enumerate(frames):
        if frame["id"] != index or type(frame["bytes"]) is not int or frame["bytes"] < 0:
            raise ValueError("Invalid fixed GPU frame directory")
        if not frame["recipients"] or len(set(frame["recipients"])) != len(frame["recipients"]):
            raise ValueError("Frames require distinct recipients")
        for member in frame["recipients"]:
            if member not in pending or frame["bytes"] > budget_bytes:
                raise ValueError("Frame recipient or single-frame staging budget mismatch")
            pending[member].append(frame)
    rounds = []
    cursors = dict.fromkeys(members, 0)
    while any(cursors[m] < len(pending[m]) for m in members):
        batch = {}
        for member in members:
            entries, size = [], 0
            for frame in pending[member][cursors[member] :]:
                offset = (size + 3) // 4 * 4
                if offset + frame["bytes"] > budget_bytes:
                    break
                entries.append(dict(frame=frame["id"], offset=offset, bytes=frame["bytes"], owner=frame["owner"]))
                size = offset + frame["bytes"]
                cursors[member] += 1
            batch[member] = dict(bytes=size, entries=entries)
        rounds.append(batch)
    return rounds


def select_overlap(rounds: list[dict], *, requested: bool, full: bool, budget_bytes: int) -> tuple[bool, str]:
    if not requested:
        return False, "disabled"
    if full:
        return False, "initial_or_periodic_FULL"
    members = tuple(rounds[0]) if rounds else ()
    for member in members:
        retained = 0
        for batch in rounds:
            if batch[member]["bytes"]:
                retained = (retained + 3) // 4 * 4 + batch[member]["bytes"]
        if retained > budget_bytes:
            return False, "complete_payload_exceeds_staging_budget"
    return True, "complete_payload_admitted"


class GpuSyncGroup:
    """A dedicated group, initialized by all sources and proxies
    concurrently."""

    def __init__(self, connection: dict, *, rank: int, device: Any, timeout_s: float) -> None:
        import torch
        import torch.distributed as dist

        from relax.utils.distributed_utils import init_process_group

        self.device = torch.device(device)
        self.rank = rank
        self.group = init_process_group(
            backend="nccl",
            init_method=connection["init_method"],
            world_size=connection["world_size"],
            rank=rank,
            group_name=connection["group_name"],
            timeout=timedelta(seconds=timeout_s),
        )
        # Warm every communicator member before P2P subsets participate.
        warmup = torch.zeros(1, dtype=torch.int32, device=self.device)
        dist.all_reduce(warmup, group=self.group)
        torch.cuda.synchronize(self.device)

    def receive(self, batch: dict, slab: Any = None) -> tuple[Any, list]:
        import torch
        import torch.distributed as dist

        if slab is None:
            slab = torch.empty(batch["bytes"], dtype=torch.uint8, device=self.device)
        if (
            slab.device != self.device
            or slab.dtype != torch.uint8
            or slab.ndim != 1
            or not slab.is_contiguous()
            or slab.numel() != batch["bytes"]
        ):
            raise ValueError("Receive storage must match the admitted byte slab")
        works = [
            dist.irecv(slab[e["offset"] : e["offset"] + e["bytes"]], src=e["owner"], group=self.group)
            for e in batch["entries"]
            if e["bytes"]
        ]
        return slab, works

    def send(
        self, frames: list[dict], batches: dict, payloads: dict[int, Any], receiver_ranks: dict, *, deadline: float
    ) -> None:
        import torch.distributed as dist

        works = []
        for member, batch in batches.items():
            for entry in batch["entries"]:
                frame = frames[entry["frame"]]
                if frame["owner"] != self.rank or not entry["bytes"]:
                    continue
                raw = payloads[frame["id"]]
                if raw.device != self.device or len(raw) != entry["bytes"]:
                    raise ValueError("Source payload disagrees with the admitted GPU plan")
                works.append(dist.isend(raw, dst=receiver_ranks[member], group=self.group))
        self.fence(works, deadline=deadline)

    def fence(self, works: list, *, deadline: float) -> None:
        import torch

        for work in works:
            work.wait(timeout=timedelta(seconds=remaining(deadline)))
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        import torch.distributed as dist

        dist.destroy_process_group(self.group)
