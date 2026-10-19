#!/bin/bash
# Clones NVIDIA's Ising-Decoding code (Apache-2.0) at the commit this work was built and measured against, into
# third_party/Ising-Decoding (or ISING_DECODING_DIR), and installs its inference requirements.
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D=${ISING_DECODING_DIR:-$R/third_party/Ising-Decoding}
COMMIT=33acb152e403bc189f2effdb07f1a87b34c745f1
if [ ! -d "$D/.git" ]; then git clone https://github.com/NVIDIA/Ising-Decoding.git "$D"; fi
git -C "$D" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$D" checkout -q "$COMMIT"
pip install -r "$D/code/requirements_public_inference.txt"
echo "NVIDIA code at $D ($(git -C "$D" rev-parse --short HEAD))"
