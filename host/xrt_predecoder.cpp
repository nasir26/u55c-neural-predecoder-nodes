// Host runtime of the pre-decoder kernel (hls/predecoder.cpp) on one card.
// Author: Nasir Ali, C-DAC Noida.
//
// Loads the bitstream, copies the packed weights and parameters into every kernel instance once, then launches
// every instance `launches` times on `windows` windows each, two launches per instance in flight (one with
// NPD_INFLIGHT=1, for latency). The input windows
// of a launch are taken cyclically from the input file; every launch's output bits are copied back to host memory
// before it counts. Instances are named predecoder_1, predecoder_2, ... (build script).
//
// usage: xrt_predecoder <xclbin> <wmem.bin> <params.bin> <x.bin> <D> <T> <windows> <launches> <device_index>
//                       [<bits_out.bin>]
// With bits_out.bin, the output of the first launch of every instance is written to it (instance after instance)
// for checking against the integer model. As in the other runtimes of this project: "ready" on stderr after setup,
// optional STIM_GO_FILE start barrier, the wall clock window of the timed loop on stderr, windows/s on stdout.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>
#include <experimental/xrt_xclbin.h>

namespace {

[[noreturn]] void fail(const char *msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

std::vector<uint8_t> read_file(const char *path) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) fail("cannot open an input file");
    std::vector<uint8_t> v;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    return v;
}

long long epoch_ns(std::chrono::system_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 10 && argc != 11) {
        std::fprintf(stderr, "usage: %s <xclbin> <wmem.bin> <params.bin> <x.bin> <D> <T> <windows> <launches> "
                             "<device_index> [<bits_out.bin>]\n", argv[0]);
        return 2;
    }
    const char *xclbin = argv[1];
    std::vector<uint8_t> wmem = read_file(argv[2]), params = read_file(argv[3]), x = read_file(argv[4]);
    const int D = std::atoi(argv[5]), T = std::atoi(argv[6]), windows = std::atoi(argv[7]);
    const long launches = std::atol(argv[8]);
    const unsigned dev_index = std::strtoul(argv[9], nullptr, 10);
    const size_t sites = (size_t)T * D * D, in_bytes = (size_t)windows * sites;
    if (x.size() < sites || x.size() % sites) fail("x.bin is not a whole number of windows");
    if (windows < 1 || launches < 1) fail("windows and launches must be positive");
    const int L = (int(params.size() / 4) + 380) / 512;   // params hold 512 L - 380 int32 words (host/pack.py)
    if (512 * L - 380 != int(params.size() / 4) || (L != 4 && L != 6)) fail("params.bin does not hold 4 or 6 layers");

    xrt::device device(dev_index);
    xrt::uuid uuid = device.load_xclbin(xclbin);
    const size_t n_cu = xrt::xclbin(xclbin).get_kernel("predecoder").get_cus().size();
    struct Instance {
        xrt::kernel k;
        xrt::bo w, p, in[2], out[2];
        xrt::run run[2];
    };
    std::vector<Instance> inst;
    for (size_t c = 1; c <= n_cu; c++) {
        xrt::kernel k(device, uuid, "predecoder:{predecoder_" + std::to_string(c) + "}");
        Instance I{k, xrt::bo(device, wmem.size(), k.group_id(0)), xrt::bo(device, params.size(), k.group_id(1)),
                   {xrt::bo(device, in_bytes, k.group_id(2)), xrt::bo(device, in_bytes, k.group_id(2))},
                   {xrt::bo(device, in_bytes, k.group_id(3)), xrt::bo(device, in_bytes, k.group_id(3))},
                   {xrt::run(k), xrt::run(k)}};
        std::memcpy(I.w.map<uint8_t *>(), wmem.data(), wmem.size());
        std::memcpy(I.p.map<uint8_t *>(), params.data(), params.size());
        I.w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        I.p.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int s = 0; s < 2; s++) {           // input windows, cyclic over the file; fixed for the run
            uint8_t *dst = I.in[s].map<uint8_t *>();
            for (size_t i = 0; i < in_bytes; i++) dst[i] = x[i % x.size()];
            I.in[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        inst.push_back(I);
    }
    const bool one_in_flight = std::getenv("NPD_INFLIGHT") && std::atoi(std::getenv("NPD_INFLIGHT")) == 1;
    std::FILE *dump = argc == 11 ? std::fopen(argv[10], "wb") : nullptr;
    auto launch = [&](long k) {
        for (Instance &I : inst) I.run[k % 2] = I.k(I.w, I.p, I.in[k % 2], I.out[k % 2], windows, D, T, L);
    };
    auto finish = [&](long k) {
        for (Instance &I : inst) {
            I.run[k % 2].wait();
            I.out[k % 2].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (dump && k == 0) std::fwrite(I.out[0].map<const uint8_t *>(), 1, in_bytes, dump);
        }
    };
    std::fprintf(stderr, "instances %zu layers %d\n", inst.size(), L);
    std::fprintf(stderr, "ready\n");
    std::fflush(stderr);
    if (const char *go = std::getenv("STIM_GO_FILE")) {
        struct stat st;
        while (stat(go, &st) != 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    const auto w0 = std::chrono::system_clock::now();
    const auto t0 = std::chrono::steady_clock::now();
    if (one_in_flight) {                    // latency: each launch waits for the previous one
        for (long k = 0; k < launches; k++) {
            launch(k);
            finish(k);
        }
    } else {                                // throughput: two launches per instance in flight
        launch(0);
        for (long k = 1; k < launches; k++) {
            launch(k);
            finish(k - 1);
        }
        finish(launches - 1);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto w1 = std::chrono::system_clock::now();
    if (dump) std::fclose(dump);
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double n = (double)launches * windows * inst.size();
    std::fprintf(stderr, "completed %ld launches, %.0f windows in %.6f s\n", launches, n, sec);
    std::fprintf(stderr, "window_ns %lld %lld device %u bdf %s\n", epoch_ns(w0), epoch_ns(w1), dev_index,
                 device.get_info<xrt::info::device::bdf>().c_str());
    std::printf("%.3f\n", n / sec);
    return 0;
}
