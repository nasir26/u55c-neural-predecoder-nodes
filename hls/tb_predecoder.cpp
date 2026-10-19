// C simulation testbench of the pre-decoder kernel (also its cosimulation testbench).
// Author: Nasir Ali, C-DAC Noida.
// usage: tb_predecoder <wmem.bin> <params.bin> <x.bin> <num_windows> <D> <T> <bits.bin>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <ap_int.h>
#include "predecoder_config.hpp"

using namespace npd;

extern "C" void predecoder(const ap_uint<8 * PI> *wmem, const int32_t *params, const ap_uint<8> *x_in,
                           ap_uint<8> *bits_out, int num_windows, int D, int T, int L);

template <typename T>
static std::vector<T> read_all(const char *path) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    std::vector<T> v;
    T x;
    while (std::fread(&x, sizeof(T), 1, f) == 1) v.push_back(x);
    std::fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s <wmem.bin> <params.bin> <x.bin> <num_windows> <D> <T> <bits.bin>\n", argv[0]);
        return 2;
    }
    std::vector<uint8_t> wb = read_all<uint8_t>(argv[1]);
    std::vector<int32_t> params = read_all<int32_t>(argv[2]);
    std::vector<uint8_t> xb = read_all<uint8_t>(argv[3]);
    const int nw = std::atoi(argv[4]), D = std::atoi(argv[5]), T = std::atoi(argv[6]);
    const int L = (int(params.size()) + 380) / 512;   // params hold 512 L - 380 words (host/pack.py)
    if (512 * L - 380 != int(params.size()) || (L != 4 && L != 6)) {
        std::fprintf(stderr, "params.bin does not hold 4 or 6 layers\n");
        return 1;
    }
    std::vector<ap_uint<8 * PI>> wmem(W_WORDS);
    for (int i = 0; i < int(wb.size() / PI) && i < W_WORDS; i++)
        for (int k = 0; k < PI; k++) wmem[i].range(8 * k + 7, 8 * k) = wb[(size_t)i * PI + k];
    const size_t n = (size_t)nw * T * D * D;
    std::vector<ap_uint<8>> x(std::max(n, (size_t)SITES_MAX)), bits(std::max(n, (size_t)SITES_MAX));
    for (size_t i = 0; i < n; i++) x[i] = xb[i];
    params.resize(std::max<size_t>(params.size(), 512 * LAYERS - 380));
    predecoder(wmem.data(), params.data(), x.data(), bits.data(), nw, D, T, L);
    std::FILE *o = std::fopen(argv[7], "wb");
    for (size_t i = 0; i < n; i++) {
        uint8_t v = bits[i];
        std::fwrite(&v, 1, 1, o);
    }
    std::fclose(o);
    return 0;
}
