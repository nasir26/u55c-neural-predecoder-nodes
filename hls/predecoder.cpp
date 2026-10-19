// Four card neural pre-decoder: HLS kernel of NVIDIA's surface code pre-decoder (model 1: four 3x3x3
// convolutions, 4 -> 128 -> 128 -> 128 -> 4, GELU), in the integer arithmetic of model/qmodel.py.
// Author: Nasir Ali, C-DAC Noida.
//
// Per window (T x D x D sites, runtime D <= D_MAX, T <= T_MAX) the four layers run one after the other on one
// convolution engine. Activations stay on chip in two buffers that alternate between layers; for each block of PO
// output channels the partial sums of all sites stay on chip; for each tap of the 3x3x3 kernel and each block of PI
// input channels a PO x PI weight block is held in registers while all sites stream through, one site per cycle,
// reading the activations of the neighbour at the tap offset (zero outside the window). The epilogue of a hidden
// layer adds the bias, requantizes with an integer multiplier and shift, and maps the result through the GELU
// table; the last layer compares with an integer threshold and emits one bit per output channel.
//
// Interface (layouts written by host/pack.py):
//   wmem    W_WORDS words of PI int8 weights: word ((l * TAPS + tap) * CB_IN + cib) * C + o holds the weights of
//           output channel o from input channels cib * PI .. cib * PI + PI - 1 at kernel offset tap of layer l
//   params  int32: bias[LAYERS][C], mult[LAYERS - 1][C], thr[C_OUT], lut[LAYERS - 1][LUT_SIZE] (int8 values)
//   x_in    one byte per site: input channel k in bits 2k, 2k + 1 (values 0, 1, 2)
//   bits    one byte per site: output channel k in bit k
#include <ap_int.h>
#include <stdint.h>

#include "predecoder_config.hpp"

using namespace npd;

typedef ap_uint<8 * PI> aword_t;   // PI int8 activations or weights
typedef ap_int<32> acc_t;

namespace {

constexpr int P_BIAS = 0;
constexpr int P_MULT = P_BIAS + LAYERS * C;
constexpr int P_THR = P_MULT + (LAYERS - 1) * C;
constexpr int P_LUT = P_THR + C_OUT;
constexpr int P_WORDS = P_LUT + (LAYERS - 1) * LUT_SIZE;

inline ap_int<8> lane(const aword_t &w, int i) {
#pragma HLS inline
    return ap_int<8>(w.range(8 * i + 7, 8 * i));
}

// One layer of one window: act_in -> act_out (hidden layers) or -> bits (last layer).
void layer(int l, int D, int T, const aword_t W[W_WORDS], const int32_t P[P_WORDS],
           aword_t act_in[SITES_MAX][CB_IN], aword_t act_out[SITES_MAX][CB_IN], ap_uint<8> bits[SITES_MAX]) {
    const int sites = T * D * D;
    const int cin_blocks = (l == 0) ? 1 : CB_IN;
    const int cout_blocks = (l == LAYERS - 1) ? 1 : CB_OUT;
    const int cout = (l == LAYERS - 1) ? C_OUT : C;
    acc_t psum[SITES_MAX][PO];
#pragma HLS array_partition variable = psum dim = 2 complete
    ap_int<8> wblk[PO][PI];
#pragma HLS array_partition variable = wblk dim = 0 complete

COUT_BLOCKS:
    for (int cob = 0; cob < cout_blocks; cob++) {
    ZERO:
        for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
            for (int o = 0; o < PO; o++) {
#pragma HLS unroll
                psum[s][o] = 0;
            }
        }
    TAP_BLOCKS:
        for (int tb = 0; tb < TAPS * cin_blocks; tb++) {
#pragma HLS loop_tripcount max = 216
            const int tap = tb / cin_blocks, cib = tb % cin_blocks;
            const int dt = tap / 9 - 1, dy = (tap / 3) % 3 - 1, dx = tap % 3 - 1;
        LOAD_W:
            for (int o = 0; o < PO; o++) {
#pragma HLS pipeline II = 1
                const aword_t w = W[((l * TAPS + tap) * CB_IN + cib) * C + cob * PO + o];
                for (int i = 0; i < PI; i++) {
#pragma HLS unroll
                    wblk[o][i] = lane(w, i);
                }
            }
            int t = 0, y = 0, x = 0;
        SITES:
            for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
#pragma HLS dependence variable = psum inter false
                const int nt = t + dt, ny = y + dy, nx = x + dx;
                const bool valid = nt >= 0 && nt < T && ny >= 0 && ny < D && nx >= 0 && nx < D;
                const aword_t a = valid ? act_in[(nt * D + ny) * D + nx][cib] : aword_t(0);
                for (int o = 0; o < PO; o++) {
#pragma HLS unroll
                    acc_t dot = 0;
                    for (int i = 0; i < PI; i++) {
#pragma HLS unroll
                        dot += wblk[o][i] * lane(a, i);
                    }
                    psum[s][o] += dot;
                }
                if (++x == D) {
                    x = 0;
                    if (++y == D) {
                        y = 0;
                        ++t;
                    }
                }
            }
        }
    EPILOGUE:
        for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
            ap_uint<8> b = 0;
            aword_t outw[PO / PI];
#pragma HLS array_partition variable = outw complete
            for (int o = 0; o < PO; o++) {
#pragma HLS unroll
                const int oc = cob * PO + o;
                const acc_t acc = psum[s][o] + (oc < cout ? acc_t(P[P_BIAS + l * C + oc]) : acc_t(0));
                if (l == LAYERS - 1) {
                    if (o < C_OUT) b[o] = acc >= acc_t(P[P_THR + o]) ? 1 : 0;
                } else {
                    ap_int<64> v = ap_int<64>(acc) * ap_int<64>(P[P_MULT + l * C + oc]);
                    v = (v + (ap_int<64>(1) << (ACC_SHIFT - 1))) >> ACC_SHIFT;
                    const int lo = -(LUT_SIZE / 2), hi = LUT_SIZE / 2 - 1;
                    const int p = v < lo ? lo : (v > hi ? hi : int(v));
                    const ap_int<8> g = ap_int<8>(P[P_LUT + l * LUT_SIZE + p - lo]);
                    outw[o / PI].range(8 * (o % PI) + 7, 8 * (o % PI)) = ap_uint<8>(g);
                }
            }
            if (l == LAYERS - 1) {
                bits[s] = b;
            } else {
                for (int k = 0; k < PO / PI; k++) {
#pragma HLS unroll
                    act_out[s][cob * (PO / PI) + k] = outw[k];
                }
            }
        }
    }
}

}  // namespace

extern "C" void predecoder(const aword_t *wmem, const int32_t *params, const ap_uint<8> *x_in, ap_uint<8> *bits_out,
                           int num_windows, int D, int T) {
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem0 depth = 110592
#pragma HLS INTERFACE m_axi port = params offset = slave bundle = gmem0 depth = 1924
#pragma HLS INTERFACE m_axi port = x_in offset = slave bundle = gmem1 depth = 2197
#pragma HLS INTERFACE m_axi port = bits_out offset = slave bundle = gmem2 depth = 2197
#pragma HLS INTERFACE s_axilite port = num_windows
#pragma HLS INTERFACE s_axilite port = D
#pragma HLS INTERFACE s_axilite port = T
#pragma HLS INTERFACE s_axilite port = return
    static aword_t W[W_WORDS];
    static int32_t P[P_WORDS];
#pragma HLS bind_storage variable = W type = ram_2p impl = uram
    static aword_t actA[SITES_MAX][CB_IN], actB[SITES_MAX][CB_IN];
#pragma HLS array_partition variable = actA dim = 2 complete
#pragma HLS array_partition variable = actB dim = 2 complete
    static ap_uint<8> bits[SITES_MAX];
LOAD_WEIGHTS:
    for (int i = 0; i < W_WORDS; i++) {
#pragma HLS pipeline II = 1
        W[i] = wmem[i];
    }
LOAD_PARAMS:
    for (int i = 0; i < P_WORDS; i++) {
#pragma HLS pipeline II = 1
        P[i] = params[i];
    }
    const int sites = T * D * D;
WINDOWS:
    for (int w = 0; w < num_windows; w++) {
    READ_X:
        for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
            const ap_uint<8> xb = x_in[(long)w * sites + s];
            aword_t v = 0;
            for (int k = 0; k < C_IN0; k++) {
#pragma HLS unroll
                v.range(8 * k + 7, 8 * k) = ap_uint<8>(xb.range(2 * k + 1, 2 * k));
            }
            actA[s][0] = v;
            for (int k = 1; k < CB_IN; k++) {
#pragma HLS unroll
                actA[s][k] = 0;
            }
        }
        layer(0, D, T, W, P, actA, actB, bits);
        layer(1, D, T, W, P, actB, actA, bits);
        layer(2, D, T, W, P, actA, actB, bits);
        layer(3, D, T, W, P, actB, actA, bits);
    WRITE_BITS:
        for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
            bits_out[(long)w * sites + s] = bits[s];
        }
    }
}
