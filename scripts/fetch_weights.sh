#!/bin/bash
# Downloads NVIDIA's pretrained model 1 (Fast) from Hugging Face into $NPD_RUNS/models/nvidia and checks its SHA256.
# The repository is gated: accept the NVIDIA Open Model License on its page first, then pass a read token in HF_TOKEN.
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=${NPD_RUNS:-$R/runs}/models/nvidia; mkdir -p "$O"
F=ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors
SHA=22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token with access to nvidia/Ising-Decoder-SurfaceCode-1-Fast}"
python - "$O" "$F" <<'PY'
import sys
from huggingface_hub import hf_hub_download
for name in (sys.argv[2], "README.md"):
    hf_hub_download("nvidia/Ising-Decoder-SurfaceCode-1-Fast", name, local_dir=sys.argv[1])
PY
echo "$SHA  $O/$F" | sha256sum -c -
