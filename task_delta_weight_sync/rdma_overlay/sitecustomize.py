# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Initialize only the explicitly selected, immutable RDMA measurement process.

Ray's environment merge prepends the core checkout and supplies
NCCL_DEBUG=WARN. Run before any torch/NCCL import so the task overlay and
transport logs survive that merge. This module does nothing in processes
without the harness selector.
"""

import os
import sys
from pathlib import Path


if os.environ.get("DWS_HARNESS_SHA"):
    overlay = str(Path(__file__).resolve().parent)
    sys.path[:] = [overlay, *(entry for entry in sys.path if entry != overlay)]
    os.environ["NCCL_DEBUG"] = "INFO"
