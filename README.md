# Neural pre-decoder for the surface code on four Alveo U55C FPGA nodes

[![tests](https://github.com/nasir26/u55c-neural-predecoder-nodes/actions/workflows/tests.yml/badge.svg)](https://github.com/nasir26/u55c-neural-predecoder-nodes/actions/workflows/tests.yml)

An FPGA implementation of NVIDIA's convolutional pre-decoder for the rotated surface code (Ising Decoder, model 1
"Fast", Chamberland et al., arXiv:2604.12841), in 8 bit integer arithmetic, for four AMD Alveo U55C cards used as
independent compute nodes. The pre-decoder removes most local error chains from a syndrome window; PyMatching
decodes what is left. The goal is the highest window throughput on four cards first, then the lowest latency per
window.

**Status: work in progress.** The integer model, the HLS kernel and its C simulation, the host runtime and the
evaluation harness are complete and tested. HLS synthesis of the first kernel, the hardware builds and the card
measurements are not yet done; no FPGA number is reported here until it has been measured.

## What is verified so far

| Item | Result | Record |
|---|---|---|
| NVIDIA pipeline on the host CPU, d = 9, 9 rounds, 65,536 shots per basis | logical error rate 1.312e-3 with PyMatching alone, 1.282e-3 after the pre-decoder (average of X and Z) | `results/logs/nvidia_fast_d9_65536.log` |
| INT8 model, fast path against its reference integer path; weight, multiplier and table ranges | bit identical; all invariants hold | `tests/test_qmodel.py` |
| INT8 network through NVIDIA's evaluation, d = 9, 65,536 shots per basis, 2,048 calibration windows | 154 logical errors after the pre-decoder, 154 with PyMatching alone on the same shots (X 88 against 70, Z 66 against 84); standard error about 1.3e-4 per rate | `results/ler/int8_d9_65536.json` |
| HLS kernel (C simulation) against the INT8 model | bit identical at d = 9 (X and Z basis) and d = 11 | `tests/test_kernel_csim.py` |
| Same network on the host CPU, PyTorch float32, 256 threads, d = 9 | 1,359 windows per second at batch 16; 47 per second at batch 1 | `bench/cpu_torch_d9.json` |

At d = 9 the effect of the pre-decoder on the logical error rate is within the statistical error, for the FP16 and
the INT8 network alike (the two runs drew different syndrome samples). NVIDIA reports larger gains at larger
distances, which are evaluated next (d = 13), together with the agreement of INT8 and FP16 decisions on the same
windows.

## Layout

| Path | Content |
|---|---|
| `harness/nvidia_bridge.py` | NVIDIA's configuration, syndrome windows and evaluation, used unchanged |
| `harness/eval_ler.py` | logical error rate of the FP16, FP32 or INT8 network through NVIDIA's evaluation |
| `model/qmodel.py` | INT8 model: per channel weights, int32 accumulation, integer requantization, GELU table, integer threshold |
| `model/make_random_checkpoint.py` | random stand-in for model 1 (fixed seed), for tests without the gated weights |
| `host/pack.py` | weight, parameter and window layouts of the kernel |
| `host/xrt_predecoder.cpp` | XRT runtime: all kernel instances of one card, two launches in flight each, start barrier for multi card runs |
| `hls/predecoder.cpp` | HLS kernel: one convolution engine, 16 input by 64 output channels per cycle, activations on chip |
| `hls/tb_predecoder.cpp`, `hls/synth/run_hls.tcl` | C simulation and cosimulation testbench; C synthesis script |
| `tests/` | integer model and kernel C simulation tests |
| `bench/` | CPU throughput of the same network |
| `docs/DESIGN.md` | design notes: compute budget, kernel sizing, four card schemes, verification ladder |
| `results/` | raw records of the measurements above |

## Setup

```
pip install -r requirements.txt
scripts/setup_reference.sh                  # NVIDIA's code at the pinned commit, into third_party/Ising-Decoding
python model/make_random_checkpoint.py      # random stand-in, enough for the tests
pytest -v -rs                               # kernel tests need the Vitis HLS headers (XILINX_HLS), else skipped
```

The pretrained weights are gated by NVIDIA (NVIDIA Open Model License) and are not part of this repository. After
accepting the licence on Hugging Face (`nvidia/Ising-Decoder-SurfaceCode-1-Fast`):

```
HF_TOKEN=<your token> scripts/fetch_weights.sh   # downloads into runs/models/nvidia and checks the SHA256
```

Environment variables: `ISING_DECODING_DIR` (NVIDIA's code, default `third_party/Ising-Decoding`), `NPD_RUNS`
(checkpoints and outputs, default `runs/`), `QMODEL_CHECKPOINT` (checkpoint used by the tests), `XILINX_HLS`.

## Reproducing the results

```
W=runs/models/nvidia/ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors
python harness/eval_ler.py --checkpoint $W --variant fp16 --d 9 --t 9 --shots 65536 --out runs/ler/fp16_d9.json
python harness/eval_ler.py --checkpoint $W --variant int8 --d 9 --t 9 --shots 65536 --cal-windows 2048 --out runs/ler/int8_d9.json
python bench/cpu_torch_throughput.py
cd hls/synth && vitis_hls -f run_hls.tcl            # C synthesis, xcu55c, 300 MHz target
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
