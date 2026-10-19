# Four card neural pre-decoder on Alveo U55C: design

Author: Nasir Ali, C-DAC Noida.

## Goal

Run NVIDIA's pretrained surface code pre-decoder (Ising-Decoder-SurfaceCode-1-Fast; Chamberland et al.,
arXiv:2604.12841; NVIDIA Open Model License) on four Alveo U55C cards, verified bit for bit against a software model
of the quantized network, and measured against the same model on the host CPU and against PyMatching alone. First
target: throughput, with every card decoding its own shots. Second target: latency per round, with the space time
volume of one decoding window split across the four cards.

## Reference model (model 1 of NVIDIA's registry)

| Layer | Operation | Channels | Kernel |
|---|---|---|---|
| 1 | Conv3d + bias, GELU (tanh) | 4 to 128 | 3 x 3 x 3, same padding |
| 2 | Conv3d + bias, GELU (tanh) | 128 to 128 | 3 x 3 x 3 |
| 3 | Conv3d + bias, GELU (tanh) | 128 to 128 | 3 x 3 x 3 |
| 4 | Conv3d + bias | 128 to 4 | 3 x 3 x 3 |

Receptive field 9 (in rounds and lattice units); about 913,000 parameters; trained at distance d = 9 with T = 9
rounds; input (B, 4, T, d, d) binary; output (B, 4, T, d, d) logits of spacelike corrections and timelike stabilizer
flips. The host maps detectors to the input grid and applies the predicted corrections before PyMatching (NVIDIA's
`qec/surface_code/detector_input.py` and `evaluation/logical_error_rate.py` are the reference for both).

## Compute budget

Multiply accumulates per site: 4 x 128 x 27 + 2 x (128 x 128 x 27) + 128 x 4 x 27 = 912,384 (the weights are the
same count plus 388 biases). One window of d = T = 9 has 729 sites, so about 665 million multiply accumulates per
shot; d = 13 with T = 13 has 2,197 sites, about 2.00 billion.

A U55C has 9,024 DSP blocks. With two INT8 products per DSP block at 300 MHz the peak is about 5.4 x 10^12 multiply
accumulates per second per card, that is about 8,100 windows of d = 9 per second per card if the datapath is kept
busy, or about 123 us per window. These are bounds from the DSP count, not predictions; layer shapes, the halo of
same padding and on chip bandwidth decide how close a design gets.

## Hardware sketch (to be refined after the quantization study)

- **Precision.** INT8 weights per output channel, INT8 activations with per layer scales, 32 bit accumulators; the
  quantized network is defined by a NumPy model that the kernel must match bit for bit. GELU as a 256 entry lookup
  table on INT8 inputs. Quantization error is measured as logical error rate against the FP16 checkpoint, not as
  tensor error.
- **Weights on chip.** About 0.9 MB in INT8, held in URAM and BRAM for the whole run.
- **Layer pipeline.** One engine per layer (four engines), connected by streams; each engine is a line buffer over
  the (T, d, d) volume that forms the 27 neighbours of a site and a multiply accumulate array sized to the layer's
  share of the work (layers 2 and 3 carry 97.0 % of the multiply accumulates).
- **Card level.** Several instances per card as the DSP budget allows, each on its own windows; HBM holds input
  windows and output logits.

## Four cards

1. **Shot parallel (first).** Every card decodes its own windows; no data moves between cards. Expected to scale
   linearly, as the Pauli frame sampler did.
2. **Block parallel (second).** One window is split along the lattice into four blocks with a halo of R // 2 = 4
   sites, the FPGA counterpart of NVIDIA's multi GPU block scheme. The halo travels through host memory (no
   peer to peer transfers on this host), which the latency measurement must include.

## Verification ladder

1. NumPy model of the INT8 network against the FP16 checkpoint (PyTorch): logical error rate after PyMatching at
   d = 9, 11, 13 and several error rates, against PyMatching alone and against the FP16 pre-decoder.
2. C simulation of the HLS kernel bit exact against the NumPy model.
3. RTL cosimulation bit exact, and cycles per window against the model of the design.
4. Every instance of every card bit exact on hardware.
5. Logical error rate of the full pipeline on hardware samples against the FP16 reference.

## Measurements

Windows per second on one, two and four cards; latency per window and per round; the same network on the host CPU
(PyTorch, all cores); PyMatching alone; board power. No claim against GPUs beyond citing NVIDIA's published numbers,
since no GPU of that class is available here.

## Measured so far (2026-10-19)

- NVIDIA's inference pipeline runs on this host's CPU with `torch.compile` disabled (the host compiler is too old for
  it). PyMatching alone at d = 9, 9 rounds, 25 parameter noise (p about 0.003): logical error rate about 1.1e-3
  (4,096 shots per basis; to be repeated with more shots).
- Model inputs (`harness/nvidia_bridge.py`): channels 0 and 1 are syndrome bits (about 2 % ones), channels 2 and 3
  fixed geometry masks with values 0, 0.5 and 1; doubled they are the integers 0, 1, 2.
- Integer model (`model/qmodel.py`): INT8 weights per output channel, int32 accumulators, integer requantization, GELU
  table on an 8 bit pre activation, integer thresholds in the last layer. The float64 PyTorch path is bit identical
  to the NumPy reference and about 18 times faster (`tests/test_qmodel.py`).
- Host CPU, PyTorch float32, 256 threads, d = 9 and 9 rounds (`bench/cpu_torch_d9.json`): 1,359 windows per second
  at batch 16 (best), 47 at batch 1 (21 ms per window). bfloat16 is not accelerated on this CPU.

## Kernel sizing

A weight stationary engine: for each block of Po output channels, partial sums of all sites of the window stay on
chip; for each of the 27 taps and each block of Pi input channels the Po x Pi weight block is held in registers while
all sites stream through, one site per cycle (activations of the neighbour site at the tap offset, zero outside the
volume). Layer 2 at d = 9 then takes 128 / Po x 27 x 128 / Pi x 729 cycles. Two lessons of the Pauli frame sampler
apply: keep the DSP blocks of an instance well below the DSP columns of its region, and keep instances out of SLR0.
Starting point: Pi x Po = 32 x 64 (2,048 products per cycle, about 1,024 DSP blocks with two INT8 products per block,
about a third of one SLR), two instances in SLR1 and SLR2. Estimate at 300 MHz, before overheads: about 325,000
cycles per window, about 920 windows per second per instance, 1,850 per card, 7,400 on four cards, and about
1.1 ms per window (120 us per round) for one instance.

Layers 1 and 4 do not fill a 32 x 64 array (4 input channels in layer 1, 4 output channels in layer 4): run on the
same engine they would take 39,366 and 78,732 cycles instead of about 4,900 each, about 690 windows per second per
instance in total. They get their own small units in the dataflow: layer 1 adds the weight vectors of the few
nonzero syndrome bits plus a precomputed geometry term (the geometry channels are the same for every window), and
layer 4 reduces 128 channels to 4 with a 4 x 128 array, both running concurrently with the main engine.
