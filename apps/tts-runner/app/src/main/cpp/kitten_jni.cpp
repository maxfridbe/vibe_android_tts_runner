#include <jni.h>
#include "llama.h"
#include "tools/kitten-tts/sampling.h"
#include <atomic>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
using json = kitten::json;
struct Engine {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    json config;
    int threads;
};
std::unique_ptr<Engine> engine;
std::atomic<bool> canceled{false};

std::string utf8(JNIEnv * env, jbyteArray bytes) {
    std::string s(env->GetArrayLength(bytes), '\0');
    env->GetByteArrayRegion(bytes, 0, s.size(), reinterpret_cast<jbyte *>(s.data()));
    return s;
}
void error(JNIEnv * env, const std::exception & e) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}
struct Batch {
    llama_batch b;
    Batch(int n, int dim = 0) : b(llama_batch_init(n, dim, 1)) { b.n_tokens = n; }
    ~Batch() { llama_batch_free(b); }
    void position(int i, int pos, bool logits) {
        b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = logits;
    }
};
std::vector<int> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    std::vector<int> ids(n);
    n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, true);
    if (n < 0) throw std::runtime_error("Kitten tokenization failed");
    ids.resize(n);
    return ids;
}
}

extern "C" JNIEXPORT void JNICALL
Java_com_techhurts_ttsrunner_KittenNative_load(JNIEnv * env, jobject, jbyteArray path,
        jbyteArray config, jint threads) try {
    engine.reset();
    auto next = std::make_unique<Engine>();
    next->config = json::parse(utf8(env, config));
    if (next->config.at("type") != "KITTEN2") throw std::runtime_error("Not a Kitten TTS 2 model");
    next->threads = std::max(1, int(threads));
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t no_offload[] = {nullptr};
    mp.devices = no_offload;
    mp.use_extra_bufts = false;
    next->model.reset(llama_model_load_from_file(utf8(env, path).c_str(), mp));
    if (!next->model) throw std::runtime_error("Cannot load Kitten language model");
    engine = std::move(next);
} catch (const std::exception & e) { error(env, e); }

extern "C" JNIEXPORT void JNICALL
Java_com_techhurts_ttsrunner_KittenNative_unload(JNIEnv *, jobject) { engine.reset(); }

extern "C" JNIEXPORT void JNICALL
Java_com_techhurts_ttsrunner_KittenNative_cancel(JNIEnv *, jobject) { canceled = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_techhurts_ttsrunner_KittenNative_resetCancel(JNIEnv *, jobject) { canceled = false; }

extern "C" JNIEXPORT jintArray JNICALL
Java_com_techhurts_ttsrunner_KittenNative_generate(JNIEnv * env, jobject, jbyteArray text,
        jbyteArray voice, jint seed, jboolean expression) try {
    if (!engine) throw std::runtime_error("Kitten engine is not loaded");
    if (canceled) throw std::runtime_error("Canceled");
    const auto & config = engine->config;
    const auto ref = json::parse(utf8(env, voice));
    const kitten::token_map tm(config.at("token_map"));
    const auto vocab = llama_model_get_vocab(engine->model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (tm.base + tm.count > n_vocab) throw std::runtime_error("Kitten token map exceeds vocabulary");
    auto speaker = ref.at("speaker").get<std::vector<float>>();
    if (speaker.size() != size_t(llama_model_n_embd(engine->model.get())))
        throw std::runtime_error("Kitten speaker projection dimension mismatch");
    const auto gen = config.at("generation");
    const auto emotion = expression ? tokenize(vocab, gen.at("emotion_control")) : std::vector<int>{};
    auto ids = kitten::prompt(tm, tokenize(vocab, utf8(env, text)),
        tokenize(vocab, ref.at("transcript")), ref.at("reference_tokens").get<std::vector<int>>(),
        emotion, gen.value("use_reference_prompt", true));
    const int budget = 1000;
    auto cp = llama_context_default_params();
    cp.n_ctx = ids.size() + budget; cp.n_batch = 256; cp.n_ubatch = 256;
    cp.n_threads = engine->threads; cp.n_threads_batch = engine->threads;
    cp.abort_callback = [](void *) { return canceled.load(); };
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(engine->model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("Cannot initialize Kitten context");
    {
        Batch b(1, speaker.size());
        std::copy(speaker.begin(), speaker.end(), b.b.embd); b.position(0, 0, false);
        if (llama_decode(ctx.get(), b.b)) throw std::runtime_error("Kitten speaker prefill failed");
    }
    for (size_t offset = 1; offset < ids.size();) {
        int n = std::min(size_t(256), ids.size() - offset); Batch b(n);
        for (int i = 0; i < n; ++i) {
            b.b.token[i] = ids[offset + i];
            b.position(i, offset + i, offset + i + 1 == ids.size());
        }
        if (llama_decode(ctx.get(), b.b)) throw std::runtime_error("Kitten prompt prefill failed");
        offset += n;
    }
    kitten::sampling settings;
    const auto preset = config.at("decode_presets").at(expression ? "expressive" : "stable");
    settings.temperature = preset.at("temperature"); settings.top_p = preset.at("top_p");
    settings.top_k = preset.at("top_k"); settings.min_p = preset.value("min_p", 0.f);
    settings.window = gen.value("repetition_window", 50);
    std::mt19937 rng(seed);
    auto history = ids;
    std::vector<int> audio;
    bool ended = false;
    for (int step = 0; step < budget; ++step) {
        if (canceled) throw std::runtime_error("Canceled");
        const float * raw = llama_get_logits_ith(ctx.get(), -1);
        std::vector<float> scores(raw, raw + n_vocab);
        kitten::process(scores, history, ids.size(), tm, settings);
        int token = kitten::sample(scores, rng, false);
        history.push_back(token);
        if (token == tm.speech_end || token == tm.stop) { ended = true; break; }
        if (token >= tm.base && token < tm.base + tm.count) audio.push_back(token - tm.base);
        if (step + 1 < budget) {
            Batch b(1); b.b.token[0] = token; b.position(0, history.size() - 1, true);
            if (llama_decode(ctx.get(), b.b)) throw std::runtime_error("Kitten generation failed");
        }
    }
    if (!ended) throw std::runtime_error("Kitten reached its token limit; try a shorter sentence");
    if (audio.empty()) throw std::runtime_error("Kitten generated no audio tokens");
    auto out = env->NewIntArray(audio.size());
    if (out) env->SetIntArrayRegion(out, 0, audio.size(), audio.data());
    return out;
} catch (const std::exception & e) { error(env, e); return nullptr; }
