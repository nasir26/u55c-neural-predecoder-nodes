#!/bin/bash
# v++ build of the pre-decoder kernel (hls/predecoder.cpp) for the Alveo U55C.
# Author: Nasir Ali, C-DAC Noida.
# usage: build_hw.sh [clock_mhz] [out_name]        output in build/<out_name>/ (default hw)
# Two kernel instances, predecoder_1 in SLR1 and predecoder_2 in SLR2 (SLR0 holds the platform's HBM and DMA logic);
# instance n reads weights and parameters from HBM pseudo channel 2n - 2 and windows from, and writes bits to, 2n - 1.
# A hw build is a multi hour Vivado run.
set -euo pipefail
MHZ="${1:-300}"
PLATFORM="xilinx_u55c_gen3x16_xdma_3_202210_1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"
OUT="$HERE/${2:-hw}"; mkdir -p "$OUT"; cd "$OUT"
{
  echo "[connectivity]"
  echo "nk=predecoder:2:predecoder_1.predecoder_2"
  for n in 1 2; do
    echo "sp=predecoder_$n.wmem:HBM[$((2 * n - 2))]"
    echo "sp=predecoder_$n.params:HBM[$((2 * n - 2))]"
    echo "sp=predecoder_$n.x_in:HBM[$((2 * n - 1))]"
    echo "sp=predecoder_$n.bits_out:HBM[$((2 * n - 1))]"
    echo "slr=predecoder_$n:SLR$n"
  done
  echo
  echo "[vivado]"
  echo "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore"
  echo "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=${PLACE_DIRECTIVE:-ExtraTimingOpt}"
  echo "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true"
  echo "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore"
  echo "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation"
  echo "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true"
  echo "prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore"
} > connectivity.cfg
# sources the bitstream is built from
mkdir -p src_snapshot; cp "$ROOT"/hls/predecoder.cpp "$ROOT"/hls/predecoder_config.hpp src_snapshot/
echo "== [$(date -Is)] v++ compile at $MHZ MHz =="
v++ -c -t hw --platform "$PLATFORM" -k predecoder -I"$ROOT/hls" --kernel_frequency "$MHZ" \
    -o predecoder.xo "$ROOT/hls/predecoder.cpp"
echo "== [$(date -Is)] v++ link =="
v++ -l -t hw --platform "$PLATFORM" --config connectivity.cfg --kernel_frequency "$MHZ" -o predecoder.xclbin predecoder.xo
sha256sum predecoder.xclbin > predecoder.xclbin.sha256
xclbinutil --info --input predecoder.xclbin > predecoder.xclbin.info 2>&1 || true
echo "== [$(date -Is)] BUILD_EXIT 0 =="
