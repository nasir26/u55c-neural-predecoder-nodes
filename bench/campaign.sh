#!/bin/bash
# Measurement campaign after a hardware build, run inside tmux.
# Author: Nasir Ali, C-DAC Noida.
#   1. routed timing must be met (worst negative slack >= 0, or v++ scaled the kernel clock down), else stop
#   2. cards: bit exact check of every instance, throughput on 1, 2 and 4 cards, latency (bench/run_cards.py)
#   3. host CPU baseline of both networks with the cards idle (bench/cpu_torch_throughput.py)
# usage: campaign.sh [build_name]   (build/<build_name>, default hw)
set -uo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNS=${NPD_RUNS:-$R/runs}
PY=${PY:-python3}
source /opt/xilinx/xrt/setup.sh > /dev/null
B=$R/build/${1:-hw}
X=$B/predecoder.xclbin
[ -f "$X" ] || { echo "no bitstream at $X"; exit 1; }
T=$(find $B/_x -name "*timing_summary_postroute_physopted.rpt" 2>/dev/null | head -1); [ -n "$T" ] || T=$(find $B/_x -name "*timing_summary_routed.rpt" 2>/dev/null | head -1)
echo "== $(date '+%F %T') timing: $T"
WNS=$(awk '/WNS\(ns\)/{getline; getline; print $1; exit}' "$T")
MHZ=$(xclbinutil --info --input $X 2>/dev/null | awk '/Name:/{n=$2} /Type:/{t=$2} /Frequency:/{if (t=="DATA") {print $2; exit}}')
echo "WNS $WNS ns, kernel clock in the xclbin: $MHZ MHz"
awk -v w="$WNS" -v f="$MHZ" 'BEGIN{exit !(w+0 >= 0 || (f+0 > 0 && f+0 < 300))}' \
  || { echo "timing not met and clock not scaled; stopping"; exit 1; }
cat $B/predecoder.xclbin.sha256
mkdir -p $RUNS/hw $RUNS/cpu
echo "== $(date '+%F %T') cards"
$PY $R/bench/run_cards.py --xclbin $X --out $RUNS/hw/cards_$(date +%Y%m%d_%H%M).json || { echo "run_cards failed"; exit 1; }
while pgrep -f harness/eval_ler.py > /dev/null; do sleep 60; done   # host otherwise idle
echo "== $(date '+%F %T') CPU baselines (cards idle)"
M=$RUNS/models/nvidia
$PY $R/bench/cpu_torch_throughput.py --checkpoint $M/ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors \
    --d 9 --t 9 --dtypes float32 --out $RUNS/cpu/cpu_fast_d9.json
$PY $R/bench/cpu_torch_throughput.py --checkpoint $M/ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors \
    --d 13 --t 13 --dtypes float32 --out $RUNS/cpu/cpu_accurate_d13.json
echo "== $(date '+%F %T') CAMPAIGN_DONE"
