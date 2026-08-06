// JNI wrapper around llama.cpp's Qwen3-TTS support (tools/tts/tts.cpp turned
// into a persistent engine): load talker + codec mmproj once, then generate
// WAV per utterance with a progress callback and a cancel flag.
#include <jni.h>
#include <android/log.h>

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#define TAG "TtsRunnerNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static void llama_log_to_android(ggml_log_level level, const char * text, void *) {
    int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
             : level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN
                                             : ANDROID_LOG_INFO;
    __android_log_write(prio, "llama.cpp", text);
}

namespace {

struct Engine {
    common_init_result_ptr init;
    llama_model   * model = nullptr;
    llama_context * lctx  = nullptr;
    mtmd::context_ptr mctx;
    common_params params;
    std::string backend_used;
};

std::unique_ptr<Engine> g_engine;
std::atomic<bool> g_cancel{false};
std::string g_last_error;

std::string jstr(JNIEnv * env, jstring s) {
    if (!s) return "";
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    env->ReleaseStringUTFChars(s, c);
    return out;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *, void *) {
    llama_log_set(llama_log_to_android, nullptr);
    return JNI_VERSION_1_6;
}

// backend: "cpu" keeps everything on the CPU; "gpu" offloads the talker
// layers to OpenCL (Adreno). The codec ("mmproj") always runs on CPU: the
// Adreno OpenCL kernels abort on its graph (they cover LLM shapes only).
extern "C" JNIEXPORT jboolean JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nLoad(JNIEnv * env, jclass,
        jstring jmodel, jstring jmmproj, jstring jbackend, jint nThreads) try {
    g_engine.reset();
    g_last_error.clear();

    auto eng = std::make_unique<Engine>();
    common_params & params = eng->params;

    params.model.path  = jstr(env, jmodel);
    params.mmproj.path = jstr(env, jmmproj);
    params.embedding   = true;  // gen_audio needs hidden states
    params.n_batch     = 512;
    // per-utterance use peaks ~1.3k positions (speaker ref + text + frames);
    // 4096 halves KV-cache RAM vs 8192 (~450 MB on the 1.7B) — phones OOM-kill
    // silently under memory pressure, so every MB counts
    params.n_ctx       = 4096;
    params.cpuparams.n_threads = nThreads > 0 ? nThreads : 4;
    params.warmup      = false;

    const std::string backend = jstr(env, jbackend);   // "cpu" | "opencl" | "vulkan"
    params.mmproj_use_gpu = false;  // codec on CPU always, see nLoad comment
    if (backend == "cpu") {
        params.n_gpu_layers = 0;
    } else {
        params.n_gpu_layers = 999;
        // both Vulkan and OpenCL are compiled in; pin the requested one
        // explicitly instead of trusting default device selection
        ggml_backend_load_all();
        const char * want = backend == "vulkan" ? "Vulkan" : "OpenCL";
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const std::string name = ggml_backend_dev_name(dev);
            if (name.find(want) != std::string::npos) {
                params.devices.push_back(dev);
                LOGI("pinned device: %s (%s)", name.c_str(), ggml_backend_dev_description(dev));
                break;
            }
        }
        if (params.devices.empty()) {
            g_last_error = "no " + backend + " device available on this phone";
            LOGE("%s", g_last_error.c_str());
            return JNI_FALSE;
        }
    }

    common_init();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    const int64_t t_load_start = ggml_time_us();
    LOGI("nLoad: model=%s mmproj=%s backend=%s threads=%d ngl=%d",
         params.model.path.c_str(), params.mmproj.path.c_str(),
         backend.c_str(), params.cpuparams.n_threads, params.n_gpu_layers);

    eng->init = common_init_from_params(params);
    if (!eng->init || !eng->init->model() || !eng->init->context()) {
        g_last_error = "failed to load model " + params.model.path;
        LOGE("%s", g_last_error.c_str());
        return JNI_FALSE;
    }
    eng->model = eng->init->model();
    eng->lctx  = eng->init->context();

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    eng->mctx.reset(mtmd_init_from_file(params.mmproj.path.c_str(), eng->model, mparams));
    if (!eng->mctx) {
        g_last_error = "failed to load mmproj " + params.mmproj.path;
        LOGE("%s", g_last_error.c_str());
        return JNI_FALSE;
    }
    if (mtmd_gen_audio_get_info(eng->mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        g_last_error = "mmproj does not support audio generation";
        LOGE("%s", g_last_error.c_str());
        return JNI_FALSE;
    }

    eng->backend_used = backend;
    g_engine = std::move(eng);
    LOGI("engine loaded in %.1fs (backend=%s threads=%d)",
         (ggml_time_us() - t_load_start) / 1e6, backend.c_str(), params.cpuparams.n_threads);
    return JNI_TRUE;
} catch (const std::exception & e) {
    // llama.cpp and driver wrappers throw (e.g. vk::SystemError on Adreno);
    // surface as a load failure the Kotlin side can fall back from
    g_engine.reset();
    g_last_error = std::string("native exception during load: ") + e.what();
    LOGE("%s", g_last_error.c_str());
    return JNI_FALSE;
} catch (...) {
    g_engine.reset();
    g_last_error = "unknown native exception during load";
    LOGE("%s", g_last_error.c_str());
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nUnload(JNIEnv *, jclass) {
    g_engine.reset();
}

// Local requantization: nobody hosts a Q4_0 of this model, but the Adreno
// OpenCL kernels are tuned for Q4_0 specifically, so the app derives it from
// the downloaded Q8_0 (streaming, low memory, a few minutes on-device).
extern "C" JNIEXPORT jboolean JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nQuantize(JNIEnv * env, jclass,
        jstring jsrc, jstring jdst, jstring jtype) try {
    g_last_error.clear();
    const std::string src  = jstr(env, jsrc);
    const std::string dst  = jstr(env, jdst);
    const std::string type = jstr(env, jtype);

    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.nthread = 4;
    qp.allow_requantize = true;
    if (type == "Q4_0") {
        qp.ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
    } else if (type == "Q4_K_M") {
        qp.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    } else {
        g_last_error = "unsupported quant type " + type;
        return JNI_FALSE;
    }

    llama_backend_init();
    LOGI("nQuantize: %s -> %s (%s)", src.c_str(), dst.c_str(), type.c_str());
    const int64_t t0 = ggml_time_us();
    if (llama_model_quantize(src.c_str(), dst.c_str(), &qp) != 0) {
        g_last_error = "llama_model_quantize failed";
        LOGE("%s", g_last_error.c_str());
        return JNI_FALSE;
    }
    LOGI("nQuantize done in %.1fs", (ggml_time_us() - t0) / 1e6);
    return JNI_TRUE;
} catch (const std::exception & e) {
    g_last_error = std::string("native exception during quantize: ") + e.what();
    LOGE("%s", g_last_error.c_str());
    return JNI_FALSE;
} catch (...) {
    g_last_error = "unknown native exception during quantize";
    LOGE("%s", g_last_error.c_str());
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nCancel(JNIEnv *, jclass) {
    g_cancel = true;
}

// The cancel flag is sticky: nGenerate does NOT clear it on entry, so a
// cancel that lands between two nGenerate calls still kills the next one.
// The service clears it explicitly when a new job takes over.
extern "C" JNIEXPORT void JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nResetCancel(JNIEnv *, jclass) {
    g_cancel = false;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nLastError(JNIEnv * env, jclass) {
    return env->NewStringUTF(g_last_error.c_str());
}

// One line per compute device ggml can see (CPU + any usable GPU backends),
// with free/total memory — for the in-app status panel and debug log.
extern "C" JNIEXPORT jstring JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nDeviceInfo(JNIEnv * env, jclass) {
    ggml_backend_load_all();
    std::string out;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        const char * type = "CPU";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "iGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "ACCEL"; break;
            default: break;
        }
        out += std::string(type) + " " + ggml_backend_dev_name(dev)
             + " — " + ggml_backend_dev_description(dev)
             + " (" + std::to_string(free_mem / (1024*1024)) + "/"
             + std::to_string(total_mem / (1024*1024)) + " MB free/total)\n";
    }
    if (out.empty()) out = "no ggml devices found\n";
    return env->NewStringUTF(out.c_str());
}

// Returns a complete WAV file as bytes, or null on failure/cancel.
// progressCb (may be null) receives (framesDone, framesMax); one frame is
// 1/12.5 s of audio, so the callback doubles as a live duration estimate.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_maxfridbe_ttsrunner_TtsEngine_nGenerate(JNIEnv * env, jclass,
        jstring jtext, jstring jspeaker, jstring jlang, jstring jinstruct,
        jint maxFrames, jint seed, jfloat temp, jfloat topP, jobject progressCb) try {
    if (!g_engine) {
        g_last_error = "engine not loaded";
        return nullptr;
    }
    if (g_cancel) return nullptr;   // sticky cancel, see nResetCancel
    g_last_error.clear();
    Engine & eng = *g_engine;

    jmethodID onProgress = nullptr;
    if (progressCb) {
        jclass cbCls = env->GetObjectClass(progressCb);
        onProgress = env->GetMethodID(cbCls, "onProgress", "(II)V");
    }

    const std::string text     = jstr(env, jtext);
    const std::string speaker  = jstr(env, jspeaker);
    const std::string lang     = jstr(env, jlang);
    const std::string instruct = jstr(env, jinstruct);  // VoiceDesign description

    // fresh KV cache per utterance; the model is prompt + autoregressive codes
    llama_memory_clear(llama_get_memory(eng.lctx), true);

    mtmd::bitmap_ptr speaker_bitmap;
    if (!speaker.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(eng.mctx.get(), speaker.c_str(), false);
        if (!wrapper.bitmap) {
            g_last_error = "failed to load speaker file " + speaker;
            LOGE("%s", g_last_error.c_str());
            return nullptr;
        }
        speaker_bitmap.reset(wrapper.bitmap);
        // cap the reference length: the speaker-encoder graph's compute
        // buffer scales with it, and phones get lmkd-killed at the peak
        // (a 16 s ref contributed to a 5.7 GB RSS kill on an 11 GB phone);
        // 8 s is plenty for voice cloning
        if (mtmd_bitmap_is_audio(speaker_bitmap.get())) {
            int sr = mtmd_get_audio_sample_rate(eng.mctx.get());
            if (sr <= 0) sr = 24000;
            const size_t max_samples = (size_t) sr * 8;
            const size_t n_samples   = mtmd_bitmap_get_n_bytes(speaker_bitmap.get()) / sizeof(float);
            if (n_samples > max_samples) {
                const float * pcm = (const float *) mtmd_bitmap_get_data(speaker_bitmap.get());
                mtmd_bitmap * cut = mtmd_bitmap_init_from_audio(max_samples, pcm);
                if (cut) {
                    LOGI("speaker ref capped: %zu -> %zu samples (%d Hz)", n_samples, max_samples, sr);
                    speaker_bitmap.reset(cut);
                }
            }
        }
    }

    // official qwen-tts talker generation config: top_k 50, top_p 1.0,
    // temp 0.9, repetition_penalty 1.05, and all special/control tokens
    // (last 1024 vocab ids) suppressed except codec-EOS — without the
    // suppression the talker can sample control tokens mid-stream
    common_params_sampling sparams;
    sparams.seed  = seed;
    sparams.temp  = temp;
    sparams.top_p = topP;
    sparams.top_k = 50;
    sparams.penalty_repeat = 1.05f;
    sparams.penalty_last_n = 1024;
    {
        const llama_vocab * v = llama_model_get_vocab(eng.model);
        const int n_vocab = llama_vocab_n_tokens(v);
        for (llama_token t = n_vocab - 1024; t < n_vocab; t++) {
            if (t >= 0 && !llama_vocab_is_eog(v, t)) {
                sparams.logit_bias.push_back({t, -INFINITY});
            }
        }
    }
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)>
        smpl(common_sampler_init(eng.model, sparams), common_sampler_free);
    if (!smpl) {
        g_last_error = "sampler init failed";
        return nullptr;
    }

    mtmd_helper::gen_audio gen(eng.lctx, eng.mctx.get());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id      = 0;
    inp.prompt      = text.c_str();
    inp.prompt_len  = text.size();
    inp.speaker_ref = speaker_bitmap.get();
    inp.lang        = lang.c_str();
    inp.instruct    = instruct.empty() ? nullptr : instruct.c_str();
    inp.top_k       = 50;    // official subtalker config: top_k 50, top_p 1.0
    inp.top_p       = 1.0f;
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    if (gen.set_input(&inp) != 0) {
        g_last_error = "set_input failed";
        LOGE("%s", g_last_error.c_str());
        return nullptr;
    }

    for (;;) {
        // framesDone=0 heartbeat: the speaker-ref encode + prompt decode take
        // tens of seconds on phones; without this the UI looks hung
        if (onProgress) {
            env->CallVoidMethod(progressCb, onProgress, 0, maxFrames > 0 ? maxFrames : 512);
            if (env->ExceptionCheck()) { env->ExceptionClear(); }
        }
        int32_t ret = gen.step_prompt(eng.params.n_batch);
        if (ret < 0) {
            g_last_error = "prompt processing failed";
            LOGE("%s", g_last_error.c_str());
            return nullptr;
        }
        if (ret == 0) break;
        if (g_cancel) return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(eng.model);
    auto sample_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(smpl.get(), eng.lctx, -1);
        common_sampler_accept(smpl.get(), t, true);
        return t;
    };

    const int max_new = maxFrames > 0 ? maxFrames : 512;
    LOGI("nGenerate: %zu chars, maxFrames=%d seed=%d temp=%.2f", text.size(), max_new, seed, temp);
    const int64_t t_gen_start = ggml_time_us();
    int n_frames = 0;
    llama_token sampled = sample_code();
    const float * h_state = llama_get_embeddings_ith(eng.lctx, -1);

    for (; n_frames < max_new && !llama_vocab_is_eog(vocab, sampled); n_frames++) {
        if (g_cancel) return nullptr;
        const float * h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next) != 0) {
            g_last_error = "step_gen failed at frame " + std::to_string(n_frames);
            LOGE("%s", g_last_error.c_str());
            return nullptr;
        }
        h_state = h_next;
        sampled = sample_code();
        // every frame (~0.5-1 s on phone CPUs); the Kotlin side throttles
        if (onProgress) {
            env->CallVoidMethod(progressCb, onProgress, n_frames + 1, max_new);
            if (env->ExceptionCheck()) { env->ExceptionClear(); }
        }
    }

    int32_t      sample_rate = 0;
    const char * data        = nullptr;
    size_t       data_len    = 0;
    int64_t      n_samples   = 0;
    if (gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0 || !data || data_len == 0) {
        g_last_error = "get_output failed";
        LOGE("%s", g_last_error.c_str());
        return nullptr;
    }

    const double gen_s = (ggml_time_us() - t_gen_start) / 1e6;
    const double audio_s = sample_rate > 0 ? (double) n_samples / sample_rate : 0.0;
    LOGI("generated %d frames, %zu WAV bytes (%d Hz): %.1fs audio in %.1fs (RTF %.2f)",
         n_frames, data_len, sample_rate, audio_s, gen_s, audio_s > 0 ? gen_s / audio_s : 0.0);
    jbyteArray out = env->NewByteArray((jsize) data_len);
    if (!out) return nullptr;
    env->SetByteArrayRegion(out, 0, (jsize) data_len, (const jbyte *) data);
    return out;
} catch (const std::exception & e) {
    // engine state is suspect after a backend blew up mid-graph — drop it so
    // the next ensureLoaded starts clean (Kotlin retries the chunk on CPU)
    g_engine.reset();
    g_last_error = std::string("native exception during generate: ") + e.what();
    LOGE("%s", g_last_error.c_str());
    return nullptr;
} catch (...) {
    g_engine.reset();
    g_last_error = "unknown native exception during generate";
    LOGE("%s", g_last_error.c_str());
    return nullptr;
}
