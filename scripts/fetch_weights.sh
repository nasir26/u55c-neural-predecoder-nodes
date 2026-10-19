#!/bin/bash
# Downloads NVIDIA's pretrained surface code pre-decoders from Hugging Face into $NPD_RUNS/models/nvidia and checks
# their SHA256: model 1 (Fast) and model 4 (Accurate). Both repositories are gated: accept the NVIDIA Open Model
# License on each page first, then pass a read token in HF_TOKEN. usage: fetch_weights.sh [fast|accurate|all]
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=${NPD_RUNS:-$R/runs}/models/nvidia; mkdir -p "$O"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token with access to the NVIDIA Ising decoder repositories}"
fetch() {  # repository, file, sha256, name of the model card copy
  python - "$O" "$1" "$2" "$4" <<'PY'
import os, shutil, sys, tempfile
from huggingface_hub import hf_hub_download
out, repo, name, card = sys.argv[1:]
hf_hub_download(repo, name, local_dir=out)
with tempfile.TemporaryDirectory() as t:
    shutil.copy(hf_hub_download(repo, "README.md", local_dir=t), os.path.join(out, card))
PY
  echo "$3  $O/$2" | sha256sum -c -
}
case "${1:-all}" in fast|all)
  fetch nvidia/Ising-Decoder-SurfaceCode-1-Fast ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors \
        22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05 README.md ;; esac
case "${1:-all}" in accurate|all)
  fetch nvidia/Ising-Decoder-SurfaceCode-1-Accurate ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors \
        a78bc059543aca37a56ca6c7c4ebb791df7830f274a3bfc773e6605ce07e925b README_accurate.md ;; esac
