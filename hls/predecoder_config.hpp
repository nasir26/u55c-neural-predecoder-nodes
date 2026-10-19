// Four card neural pre-decoder: sizing of the HLS kernel (predecoder.cpp).
// Author: Nasir Ali, C-DAC Noida.
// Mirrored by host/pack.py, which packs weights and windows in this layout; tests/test_kernel_csim.py checks
// that the two agree.
#pragma once

namespace npd {

constexpr int D_MAX = 13;                    // code distance (window is T x D x D sites)
constexpr int T_MAX = 13;                    // rounds per window
constexpr int SITES_MAX = T_MAX * D_MAX * D_MAX;
constexpr int C_IN0 = 4;                     // input channels of layer 1
constexpr int C = 128;                       // hidden channels
constexpr int C_OUT = 4;                     // output channels of the last layer
constexpr int LAYERS = 6;                    // maximum; per launch L = 4 (NVIDIA model 1, Fast) or 6 (model 4, Accurate)
constexpr int TAPS = 27;                     // 3 x 3 x 3 kernel
constexpr int PI = 16;                       // input channels per cycle
constexpr int PO = 64;                       // output channels per cycle
constexpr int CB_IN = C / PI;                // input channel blocks of a hidden layer
constexpr int CB_OUT = C / PO;               // output channel blocks of a hidden layer
constexpr int PRE_BITS = 8;                  // requantized pre activation (GELU table index)
constexpr int LUT_SIZE = 1 << PRE_BITS;
constexpr int ACC_SHIFT = 24;                // fixed point precision of the requantization multipliers

// Weight memory: one word of PI int8 weights per (layer, tap, input block, output channel); layers 1 and 4 are
// zero padded to full blocks (layer 1 has 4 input channels, layer 4 has 4 output channels).
constexpr int W_WORDS_PER_LAYER = TAPS * CB_IN * C;
constexpr int W_WORDS = LAYERS * W_WORDS_PER_LAYER;

}  // namespace npd
