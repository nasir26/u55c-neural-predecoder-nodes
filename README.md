# Neural pre-decoder for the surface code on four Alveo U55C FPGA nodes

[![tests](https://github.com/nasir26/u55c-neural-predecoder-nodes/actions/workflows/tests.yml/badge.svg)](https://github.com/nasir26/u55c-neural-predecoder-nodes/actions/workflows/tests.yml)

Code, raw measurement records and manuscript sources of "Neural Pre-Decoding for the Surface Code on Four FPGA Nodes:
Integer Inference of Convolutional Pre-Decoders on Alveo U55C Accelerators" (Nasir Ali, C-DAC Noida).

The two surface code pre-decoders that NVIDIA released with pretrained weights (Ising Decoder, Chamberland et al.,
arXiv:2604.12841): model 1 "Fast" (four 3D convolution layers, 4 to 128 to 128 to 128 to 4 channels) and model 4
"Accurate" (six layers, 128 hidden channels), run as 8 bit integer networks on four AMD Alveo U55C cards. Each card is
an independent node with two instances of one HLS kernel; the layer count is a launch argument, so one bitstream runs
both networks. A pre-decoder removes local error chains from a syndrome window; PyMatching decodes what is left.

## Results

All numbers below are produced by `paper/scripts/make_numbers.py` from the records in `results/` and `build/hw/`.

| Item | Result | Record |
|---|---|---|
| Bitstream | two instances (SLR1, SLR2); worst negative slack -0.013 ns at 300 MHz after post route physical optimization, kernel clock scaled by Vitis to 298 MHz | `build/hw/reports/`, `build/hw/predecoder.xclbin.info` |
| Bit exactness on the cards | output bits of both instances of all four cards equal the integer model, for both networks (8 of 8 checks) | `results/hw/cards_20261019_2040.json` |
| Throughput, Fast, d = 9, 9 rounds | 623 windows/s on one card, 1,245 on two, 2,490 on four; 98% of the cycle model of the kernel | same |
| Throughput, Accurate, d = 13, 13 rounds | 127 windows/s on one card, 255 on two, 509 on four; 99% of the cycle model | same |
| Latency, one window, one launch in flight | 3.77 ms (Fast, d = 9), 16.56 ms (Accurate, d = 13), including transfers | same |
| Logical error rates through NVIDIA's evaluation code (X basis factor, ours against NVIDIA's published) | Fast: 1.08 / 1.10 (p = 0.003, d = 9), 0.90 / 0.91 (p = 0.003, d = 13), 1.17 / 1.24 (p = 0.006, d = 9); Accurate, d = 13: 1.29 / 1.32 (p = 0.003), 1.72 / 1.76 (p = 0.006) | `results/ler/`, `paper/generated/tab_ler.tex` |
| Integer (INT8) model | keeps the gain of the Fast network (1.17 against 1.18 over both bases, p = 0.006, d = 9) and about two thirds of that of the Accurate network (1.48 against 1.72, p = 0.006, d = 13) | `results/ler/*int8*` |
| CPU baseline, PyTorch float32, 32 to 256 threads, best of two repetitions | Fast, d = 9: 2,038 windows/s (64 threads, batch 16), 3.89 ms per window at batch 1; Accurate, d = 13: 318 windows/s, 8.77 ms at batch 1. Four cards are 1.2 and 1.6 times faster; one card is slower than the host | `results/cpu/` |

## Layout

| Path | Content |
|---|---|
| `harness/nvidia_bridge.py` | NVIDIA's configuration, syndrome windows and evaluation, used unchanged |
| `harness/eval_ler.py` | logical error rate of the FP16, FP32 or INT8 network through NVIDIA's evaluation, at any physical error rate (`--p`) |
| `model/qmodel.py` | INT8 model: per channel weights, int32 accumulation, integer requantization, GELU table, integer threshold |
| `model/make_random_checkpoint.py` | random stand-in for model 1 (fixed seed), for tests without the gated weights |
| `host/pack.py` | weight, parameter and window layouts of the kernel |
| `host/xrt_predecoder.cpp` | XRT runtime: both kernel instances of one card, two launches in flight each, start barrier for multi card runs |
| `hls/predecoder.cpp` | HLS kernel: one convolution engine, 16 input by 64 output channels per cycle, four or six layers, activations on chip |
| `hls/tb_predecoder.cpp`, `hls/synth/run_hls.tcl` | C simulation testbench; C synthesis script |
| `build/build_hw.sh` | v++ build of the two instance bitstream; `build/hw/` holds its configuration, source snapshot and routed reports |
| `bench/run_cards.py`, `bench/campaign.sh` | card campaign: bit exact checks, throughput on 1, 2 and 4 cards, latency |
| `bench/cpu_torch_throughput.py` | CPU baseline |
| `tests/` | integer model and kernel C simulation tests (both networks) |
| `results/` | raw records: `ler/`, `hw/`, `cpu/`, `hls/`, `logs/`, `host_platform.json` |
| `paper/` | manuscript sources; `paper/scripts/` regenerates every number, table and figure from `results/` and `build/hw/` |
| `docs/DESIGN.md` | design notes |

## Setup

```
pip install -r requirements.txt
scripts/setup_reference.sh                  # NVIDIA's code at the pinned commit, into third_party/Ising-Decoding
python model/make_random_checkpoint.py      # random stand-in, enough for the tests
pytest -v -rs                               # kernel tests need the Vitis HLS headers (XILINX_HLS), else skipped
```

The pretrained weights are gated by NVIDIA (NVIDIA Open Model License) and are not part of this repository. After
accepting the licence on Hugging Face (`nvidia/Ising-Decoder-SurfaceCode-1-Fast`, `nvidia/Ising-Decoder-SurfaceCode-1-Accurate`):

```
HF_TOKEN=<your token> scripts/fetch_weights.sh   # both networks, into runs/models/nvidia, SHA256 checked
```

Environment variables: `ISING_DECODING_DIR` (NVIDIA's code, default `third_party/Ising-Decoding`), `NPD_RUNS`
(checkpoints and outputs, default `runs/`), `QMODEL_CHECKPOINT` (checkpoint used by the tests), `XILINX_HLS`.

## Reproducing the results

Tables and figures from the stored records (no FPGA needed):

```
cd paper && python scripts/make_numbers.py && python scripts/make_figures.py && latexmk -pdf main.tex
```

Logical error rates (one run per row of the table; `--p` sets the physical error rate):

```
F=runs/models/nvidia/ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors
A=runs/models/nvidia/ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors
python harness/eval_ler.py --checkpoint $F --variant fp32 --d 9 --t 9 --p 0.006 --shots 1048576 --out runs/ler/fp32_d9_p006.json
python harness/eval_ler.py --checkpoint $A --variant fp32 --d 13 --t 13 --p 0.006 --shots 262144 --out runs/ler/acc_fp32_d13_p006.json
python harness/eval_ler.py --checkpoint $A --variant int8 --d 13 --t 13 --p 0.006 --shots 65536 --out runs/ler/acc_int8_d13_p006.json
```

Bitstream and card measurements (Vitis 2023.2, XRT 2.16, platform `xilinx_u55c_gen3x16_xdma_3_202210_1`):

```
scripts/fetch_bitstreams.sh                        # bitstream from the release bitstreams-v1, SHA256 checked
# or rebuild it: build/build_hw.sh (multi hour Vivado run)
mkdir -p runs/bin && g++ -std=c++17 -O2 -Wall -I$XILINX_XRT/include -o runs/bin/xrt_predecoder host/xrt_predecoder.cpp \
    -L$XILINX_XRT/lib -lxrt_coreutil -luuid -pthread
bench/campaign.sh                                  # bit exact checks, throughput on 1, 2 and 4 cards, latency, CPU baseline
```

NVIDIA's pipeline runs on the CPU here with `torch.compile` disabled (`TORCHDYNAMO_DISABLE=1`); eager execution
computes the same function.

## Related work by the author

- Multi FPGA statevector simulation on four Alveo U55C cards (the cards as nodes model used here):
  [u55c-multi-fpga-statevector](https://github.com/nasir26/u55c-multi-fpga-statevector)
- Pauli frame sampler on four FPGA nodes: [u55c-pauli-frame-nodes](https://github.com/nasir26/u55c-pauli-frame-nodes)

## Author

Nasir Ali, Centre for Development of Advanced Computing (C-DAC), Noida, India. nasirali2607@gmail.com

## License

Apache License 2.0 for the code in this repository (`LICENSE`). NVIDIA's code and weights are not included; see
`NOTICE`.
