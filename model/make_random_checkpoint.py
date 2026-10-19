"""Writes the random stand-in for model 1 ($NPD_RUNS/models/random_model1_fp16.safetensors): NVIDIA's model 1 with
its default initialisation under a fixed seed, saved in NVIDIA's SafeTensors format. It exercises the whole pipeline
(tests, packing, kernel C simulation) without the gated pretrained weights; its decoding accuracy means nothing.

Author: Nasir Ali, C-DAC Noida.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import nvidia_bridge as nb  # noqa: E402  (puts NVIDIA's code on the path)
import torch  # noqa: E402
from export.safetensors_utils import _build_minimal_cfg, save_safetensors  # noqa: E402
from model.factory import ModelFactory  # noqa: E402

SEED = 20261019


def main() -> None:
    torch.manual_seed(SEED)
    m = ModelFactory.create_model(_build_minimal_cfg(1)).eval()
    out = nb.RUNS / "models" / "random_model1_fp16.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(m, str(out), 1, dtype="fp16", extra_metadata={"note": "random initialisation, pipeline test only"})
    print(f"wrote {out}: {sum(p.numel() for p in m.parameters())} parameters, seed {SEED}")


if __name__ == "__main__":
    main()
