// Four card neural pre-decoder: HLS kernel of NVIDIA's surface code pre-decoders (model 1, Fast: four 3x3x3
// convolutions 4 -> 128 -> 128 -> 128 -> 4; model 4, Accurate: six, 4 -> 128 x 5 -> 4; GELU), in the integer
// arithmetic of model/qmodel.py. The layer count L (4 or 6) is a launch argument, so one bitstream runs both.
// Author: Nasir Ali, C-DAC Noida.
//
// Per window (T x D x D sites, runtime D <= D_MAX, T <= T_MAX) the L layers run one after the other on one
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
//   params  int32: bias[L][C], mult[L - 1][C], thr[C_OUT], lut[L - 1][LUT_SIZE] (int8 values); 512 L - 380 words
//   x_in    one byte per site: input channel k in bits 2k, 2k + 1 (values 0, 1, 2)
//   bits    one byte per site: output channel k in bit k
#include <ap_int.h>
#include <stdint.h>

#include "predecoder_config.hpp"

using namespace npd;

typedef ap_uint<8 * PI> aword_t;   // PI int8 activations or weights
typedef ap_int<32> acc_t;

namespace {

// parameter layout for L layers (offsets computed per launch); P_WORDS is the size for the maximum LAYERS
inline int p_mult(int L) { return L * C; }
inline int p_thr(int L) { return p_mult(L) + (L - 1) * C; }
inline int p_lut(int L) { return p_thr(L) + C_OUT; }
inline int p_words(int L) { return p_lut(L) + (L - 1) * LUT_SIZE; }
constexpr int P_WORDS = LAYERS * C + (LAYERS - 1) * C + C_OUT + (LAYERS - 1) * LUT_SIZE;

inline ap_int<8> lane(const aword_t &w, int i) {
#pragma HLS inline
    return ap_int<8>(w.range(8 * i + 7, 8 * i));
}

// One layer of one window: act_in -> act_out (hidden layers) or -> bits (last layer).
void layer(int l, int L, int D, int T, const aword_t W[W_WORDS], const int32_t P[P_WORDS],
           aword_t act_in[SITES_MAX][CB_IN], aword_t act_out[SITES_MAX][CB_IN], ap_uint<8> bits[SITES_MAX]) {
    const int sites = T * D * D;
    const int cin_blocks = (l == 0) ? 1 : CB_IN;
    const bool last = l == L - 1;
    const int cout_blocks = last ? 1 : CB_OUT;
    const int cout = last ? C_OUT : C;
    acc_t psum[SITES_MAX][PO];
#pragma HLS array_partition variable = psum dim = 2 complete
    ap_int<8> wblk[PO][PI];
#pragma HLS array_partition variable = wblk dim = 0 complete

    // per output channel copies of this layer's epilogue parameters: the epilogue reads them at fixed lanes, so no
    // wide multiplexer over the parameter memory is needed in its pipeline
    acc_t bias_r[PO], mult_r[PO], thr_r[PO];
    ap_int<8> lut_r[PO][LUT_SIZE];
#pragma HLS array_partition variable = bias_r complete
#pragma HLS array_partition variable = mult_r complete
#pragma HLS array_partition variable = thr_r complete
#pragma HLS array_partition variable = lut_r dim = 1 complete

COUT_BLOCKS:
    for (int cob = 0; cob < cout_blocks; cob++) {
    LOAD_EPI:
        for (int o = 0; o < PO; o++) {
#pragma HLS pipeline II = 1
            const int oc = cob * PO + o;
            bias_r[o] = oc < cout ? acc_t(P[l * C + oc]) : acc_t(0);
            mult_r[o] = !last ? acc_t(P[p_mult(L) + l * C + oc]) : acc_t(0);
            thr_r[o] = (last && o < C_OUT) ? acc_t(P[p_thr(L) + o]) : acc_t(0);
        }
        if (!last && cob == 0) {
        LOAD_LUT:
            for (int j = 0; j < LUT_SIZE; j++) {
#pragma HLS pipeline II = 1
                const ap_int<8> g = ap_int<8>(P[p_lut(L) + l * LUT_SIZE + j]);
                for (int o = 0; o < PO; o++) {
#pragma HLS unroll
                    lut_r[o][j] = g;
                }
            }
        }
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
                const acc_t acc = psum[s][o] + bias_r[o];
                if (last) {
                    if (o < C_OUT) b[o] = acc >= thr_r[o] ? 1 : 0;
                } else {
                    ap_int<64> v = ap_int<64>(acc) * ap_int<64>(mult_r[o]);
                    v = (v + (ap_int<64>(1) << (ACC_SHIFT - 1))) >> ACC_SHIFT;
                    const int lo = -(LUT_SIZE / 2), hi = LUT_SIZE / 2 - 1;
                    const int p = v < lo ? lo : (v > hi ? hi : int(v));
                    outw[o / PI].range(8 * (o % PI) + 7, 8 * (o % PI)) = ap_uint<8>(lut_r[o][p - lo]);
                }
            }
            if (last) {
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
                           int num_windows, int D, int T, int L) {
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem0 depth = 165888
#pragma HLS INTERFACE m_axi port = params offset = slave bundle = gmem0 depth = 2692
#pragma HLS INTERFACE m_axi port = x_in offset = slave bundle = gmem1 depth = 2197
#pragma HLS INTERFACE m_axi port = bits_out offset = slave bundle = gmem2 depth = 2197
#pragma HLS INTERFACE s_axilite port = num_windows
#pragma HLS INTERFACE s_axilite port = D
#pragma HLS INTERFACE s_axilite port = T
#pragma HLS INTERFACE s_axilite port = L
#pragma HLS INTERFACE s_axilite port = return
    static aword_t W[W_WORDS];
    static int32_t P[P_WORDS];
#pragma HLS bind_storage variable = W type = ram_2p impl = uram
#pragma HLS bind_storage variable = P type = ram_1p impl = bram
    static aword_t actA[SITES_MAX][CB_IN], actB[SITES_MAX][CB_IN];
#pragma HLS array_partition variable = actA dim = 2 complete
#pragma HLS array_partition variable = actB dim = 2 complete
    static ap_uint<8> bits[SITES_MAX];
LOAD_WEIGHTS:
    for (int i = 0; i < L * W_WORDS_PER_LAYER; i++) {   // only the L layers the host supplied
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 165888
        W[i] = wmem[i];
    }
LOAD_PARAMS:
    for (int i = 0; i < p_words(L); i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2692
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
        layer(0, L, D, T, W, P, actA, actB, bits);
        layer(1, L, D, T, W, P, actB, actA, bits);
        layer(2, L, D, T, W, P, actA, actB, bits);
        layer(3, L, D, T, W, P, actB, actA, bits);
        if (L == 6) {
            layer(4, L, D, T, W, P, actA, actB, bits);
            layer(5, L, D, T, W, P, actB, actA, bits);
        }
    WRITE_BITS:
        for (int s = 0; s < sites; s++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 2197
            bits_out[(long)w * sites + s] = bits[s];
        }
    }
}
