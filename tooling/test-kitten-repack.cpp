// Standalone native check: link ggml-base from the pinned builder image.
// Compares every repacked weight using GGML's independent dequantizers.
#include "kitten_repack.h"
#include <cstring>
#include <iostream>
#include <random>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void verify(const std::string & source, const std::string & output) {
    using Meta = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    Meta a(gguf_init_from_file(source.c_str(), {true, nullptr}), gguf_free);
    Meta b(gguf_init_from_file(output.c_str(), {true, nullptr}), gguf_free);
    require(a && b, "Invalid GGUF");
    const auto count = gguf_get_n_tensors(a.get());
    require(count == gguf_get_n_tensors(b.get()), "Tensor count changed");
    std::unique_ptr<FILE, decltype(&std::fclose)> fa(std::fopen(source.c_str(), "rb"), std::fclose);
    std::unique_ptr<FILE, decltype(&std::fclose)> fb(std::fopen(output.c_str(), "rb"), std::fclose);
    uint64_t weights = 0;
    for (int64_t i = 0; i < count; ++i) {
        require(std::strcmp(gguf_get_tensor_name(a.get(), i), gguf_get_tensor_name(b.get(), i)) == 0,
                "Tensor name changed");
        require(std::memcmp(gguf_get_tensor_ne(a.get(), i), gguf_get_tensor_ne(b.get(), i),
                            sizeof(int64_t)*GGML_MAX_DIMS) == 0, "Tensor shape changed");
        fseeko(fa.get(), gguf_get_data_offset(a.get()) + gguf_get_tensor_offset(a.get(), i), SEEK_SET);
        fseeko(fb.get(), gguf_get_data_offset(b.get()) + gguf_get_tensor_offset(b.get(), i), SEEK_SET);
        const bool ternary = gguf_get_tensor_type(a.get(), i) == GGML_TYPE_TQ2_1;
        require(gguf_get_tensor_type(b.get(), i) == (ternary ? GGML_TYPE_Q4_0 : gguf_get_tensor_type(a.get(), i)),
                "Wrong output type");
        std::vector<uint8_t> packed(68*4096), expanded(144*4096);
        std::vector<float> original(256*4096), restored(256*4096);
        size_t remaining = gguf_get_tensor_size(a.get(), i);
        while (remaining) {
            const auto n = std::min(remaining, packed.size());
            const auto m = ternary ? n/68*144 : n;
            require(std::fread(packed.data(), 1, n, fa.get()) == n, "Truncated source");
            require(std::fread(expanded.data(), 1, m, fb.get()) == m, "Truncated output");
            if (ternary) {
                const auto elements = n/68*256;
                ggml_get_type_traits(GGML_TYPE_TQ2_1)->to_float(packed.data(), original.data(), elements);
                ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(expanded.data(), restored.data(), elements);
                require(std::memcmp(original.data(), restored.data(), elements*sizeof(float)) == 0,
                        "Repacking changed weight values");
                weights += elements;
            } else require(std::memcmp(packed.data(), expanded.data(), n) == 0, "Unconverted tensor changed");
            remaining -= n;
        }
    }
    std::cout << "Verified " << count << " tensors, " << weights << " ternary weights bit-exact\n";
}

int main(int argc, char ** argv) try {
    require(argc == 3, "usage: test-kitten-repack SOURCE.gguf OUTPUT.gguf");
    std::mt19937 rng(42);
    // Independently decode random blocks, including unequal scales and zero.
    for (int i = 0; i < 1000; ++i) {
        std::array<uint8_t, 68> src{};
        std::array<uint8_t, 144> dst{};
        for (int j = 0; j < 64; ++j) for (int k = 0; k < 4; ++k) src[j] |= (rng()%3) << (k*2);
        src[65] = i%2 ? 0x3c : 0; src[67] = 0x38;
        kitten_android::expand_block(src.data(), dst.data());
        std::array<float, 256> a{}, b{};
        ggml_get_type_traits(GGML_TYPE_TQ2_1)->to_float(src.data(), a.data(), a.size());
        ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(dst.data(), b.data(), b.size());
        require(std::memcmp(a.data(), b.data(), sizeof(a)) == 0, "Random block differs");
    }
    std::array<uint8_t, 68> bad{};
    std::array<uint8_t, 144> dst{};
    bad[0] = 3;
    bool rejected = false;
    try { kitten_android::expand_block(bad.data(), dst.data()); } catch (...) { rejected = true; }
    require(rejected, "Invalid ternary code accepted");
    std::atomic<bool> canceled{true};
    rejected = false;
    try { kitten_android::prepare_gpu_model(argv[1], argv[2], canceled); } catch (...) { rejected = true; }
    require(rejected, "Cancellation ignored");
    canceled = false;
    kitten_android::prepare_gpu_model(argv[1], argv[2], canceled);
    verify(argv[1], argv[2]);
    struct stat before{}, after{};
    stat(argv[2], &before);
    kitten_android::prepare_gpu_model(argv[1], argv[2], canceled);
    stat(argv[2], &after);
    require(before.st_ino == after.st_ino, "Cached output was rebuilt");
    // Simulate a truncated cache; preparation must rebuild it atomically.
    require(truncate(argv[2], 128) == 0, "Cannot truncate test cache");
    kitten_android::prepare_gpu_model(argv[1], argv[2], canceled);
    verify(argv[1], argv[2]);
    std::cout << "Random blocks, invalid codes, cancellation, cache reuse and repair passed\n";
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
