#pragma once

#include "gguf.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

namespace kitten_android {
// TQ2_1 holds two 128-weight groups per 68-byte block. Q4_0 uses eight
// 32-weight blocks (144 bytes). Copy the FP16 scales and map -1/0/+1 to
// nibbles 7/8/9: no floating-point requantization or weight rounding.
inline void expand_block(const uint8_t * src, uint8_t * dst) {
    for (int g = 0; g < 2; ++g) {
        if ((src[65 + 2*g] & 0x7c) == 0x7c)
            throw std::runtime_error("Invalid Kitten weight scale");
        for (int lane = 0; lane < 4; ++lane) {
            auto out = dst + (g*4 + lane)*18;
            out[0] = src[64 + 2*g]; out[1] = src[65 + 2*g];
            for (int j = 0; j < 16; ++j) {
                const auto lo = (src[g*32 + j] >> (2*lane)) & 3;
                const auto hi = (src[g*32 + j + 16] >> (2*lane)) & 3;
                if (lo == 3 || hi == 3) throw std::runtime_error("Invalid Kitten ternary code");
                out[2 + j] = (lo + 7) | ((hi + 7) << 4);
            }
        }
    }
}

/** Stream the pinned CPU model into a GPU-compatible GGUF, retaining all
 *  metadata and non-ternary tensors. Publishes only a complete, synced file. */
inline void prepare_gpu_model(const std::string & source, const std::string & dest,
                              const std::atomic<bool> & canceled) {
    if (source == dest) throw std::runtime_error("Kitten GPU output must differ from its source");
    auto check_cancel = [&] { if (canceled.load()) throw std::runtime_error("Canceled"); };
    check_cancel();
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> in(
        gguf_init_from_file(source.c_str(), {true, &raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw, ggml_free);
    if (!in || !tensors) throw std::runtime_error("Cannot read Kitten model for GPU preparation");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> out(gguf_init_empty(), gguf_free);
    gguf_set_kv(out.get(), in.get());
    gguf_set_val_u32(out.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_Q4_0);
    gguf_set_val_u32(out.get(), "ttsrunner.kitten.repack_version", 1);
    const auto count = gguf_get_n_tensors(in.get());
    int converted = 0;
    for (int64_t i = 0; i < count; ++i) {
        const auto name = gguf_get_tensor_name(in.get(), i);
        gguf_add_tensor(out.get(), ggml_get_tensor(tensors.get(), name));
        if (gguf_get_tensor_type(in.get(), i) == GGML_TYPE_TQ2_1) {
            gguf_set_tensor_type(out.get(), name, GGML_TYPE_Q4_0);
            ++converted;
        }
    }
    if (!converted) throw std::runtime_error("Kitten model contains no TQ2_1 weights");
    const size_t alignment = gguf_get_alignment(out.get());
    const size_t last_size = gguf_get_tensor_size(out.get(), count - 1);
    const size_t padded_last = (last_size + alignment - 1) / alignment * alignment;
    const size_t expected = gguf_get_meta_size(out.get()) +
        gguf_get_tensor_offset(out.get(), count - 1) + padded_last;
    struct stat cached{};
    if (stat(dest.c_str(), &cached) == 0 && uint64_t(cached.st_size) == expected) return;
    const auto tmp = dest + ".part";
    // A killed process may leave a partial file; it is never treated as ready.
    std::remove(tmp.c_str());
    const auto parent = dest.substr(0, dest.find_last_of('/'));
    struct statvfs space{};
    if (statvfs(parent.c_str(), &space) == 0 &&
        uint64_t(space.f_bavail)*space.f_frsize < expected + 16*1024*1024)
        throw std::runtime_error("Kitten GPU preparation needs about 1.5 GB of free storage");
    using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
    try {
        File input(std::fopen(source.c_str(), "rb"), std::fclose);
        File output(std::fopen(tmp.c_str(), "wb"), std::fclose);
        if (!input || !output) throw std::runtime_error("Cannot open Kitten GPU model files");
        if (!gguf_write_to_file_ptr(out.get(), output.get(), true))
            throw std::runtime_error("Cannot write Kitten GPU metadata");
        std::vector<uint8_t> packed(68*4096), expanded(144*4096), zeros(alignment, 0);
        for (int64_t i = 0; i < count; ++i) {
            check_cancel();
            if (fseeko(input.get(), gguf_get_data_offset(in.get()) + gguf_get_tensor_offset(in.get(), i), SEEK_SET))
                throw std::runtime_error("Cannot seek Kitten weights");
            const bool ternary = gguf_get_tensor_type(in.get(), i) == GGML_TYPE_TQ2_1;
            size_t remaining = gguf_get_tensor_size(in.get(), i);
            if (ternary && remaining % 68) throw std::runtime_error("Invalid Kitten tensor size");
            while (remaining) {
                check_cancel();
                const auto n = std::min(remaining, packed.size());
                if (std::fread(packed.data(), 1, n, input.get()) != n)
                    throw std::runtime_error("Kitten model is truncated; download it again");
                if (ternary) for (size_t b = 0; b < n/68; ++b)
                    expand_block(packed.data() + b*68, expanded.data() + b*144);
                const auto bytes = ternary ? n/68*144 : n;
                if (std::fwrite(ternary ? expanded.data() : packed.data(), 1, bytes, output.get()) != bytes)
                    throw std::runtime_error("Cannot write Kitten GPU weights; check free storage");
                remaining -= n;
            }
            const auto size = gguf_get_tensor_size(out.get(), i);
            const auto pad = (alignment - size % alignment) % alignment;
            if (std::fwrite(zeros.data(), 1, pad, output.get()) != pad)
                throw std::runtime_error("Cannot pad Kitten GPU weights");
        }
        check_cancel();
        if (uint64_t(ftello(output.get())) != expected || std::fflush(output.get()) || fsync(fileno(output.get())))
            throw std::runtime_error("Cannot finish Kitten GPU model");
        if (std::fclose(output.release())) throw std::runtime_error("Cannot close Kitten GPU model");
        if (std::rename(tmp.c_str(), dest.c_str())) throw std::runtime_error("Cannot publish Kitten GPU model");
    } catch (...) {
        std::remove(tmp.c_str());
        throw;
    }
}
} // namespace kitten_android
