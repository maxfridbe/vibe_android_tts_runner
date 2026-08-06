# TTS Runner — on-device Qwen3-TTS for Android

Share any text or article link to **TTS Runner** from any app and it reads it
aloud in a cloned voice, fully on-device: no server, no account, no audio
leaving the phone. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
Qwen3-TTS support (12 Hz codec, 24 kHz audio) with speaker-embedding voice
cloning from a 10–20 s reference clip.

Cloned from [AndroidBase](https://github.com/maxfridbe/AndroidBase) — the
containerized build system is unchanged: the host needs only podman/docker.

## Building

```sh
./build.sh tts-runner
adb install -r output/tts-runner/tts-runner-*-debug.apk
```

First build is slow: the builder image fetches the Android SDK/NDK, a pinned
llama.cpp, Vulkan headers, and a current LunarG `glslc`, then compiles
llama.cpp (CPU + Vulkan + OpenCL backends) for arm64.

## Using it

1. Open TTS Runner once: download a model (1.7B Q4_K_M, ~1.5 GB) and import a
   voice (any wav/mp3/flac with 10–20 s of clean speech from one speaker).
2. From any app, share text or an article URL → "Read aloud (TTS Runner)".
   Selected text also works via the text-selection menu.
3. Pick a voice in the popup, hit Speak. Generation streams: playback starts
   after the first chunk (~25 s of audio), a notification shows progress and
   a Stop button.

Shared URLs are fetched and run through a small readability pass (densest
`<article>`/`<p>` container); markdown noise, inline URLs, and `[17]`-style
citations are stripped before speaking.

## Architecture

```
apps/tts-runner/
├── app/src/main/cpp/
│   ├── CMakeLists.txt      # llama.cpp (CPU+Vulkan+OpenCL, Adreno kernels)
│   ├── tts_jni.cpp         # persistent engine: load once, WAV per utterance
│   ├── android_posix_shim.h, opencl-headers/, opencl-stub/
│   │                       # from android_builder/diffusion: posix_madvise
│   │                       # stub + Qualcomm CL headers + dlopen loader shim
└── app/src/main/kotlin/com/maxfridbe/ttsrunner/
    ├── TtsService.kt       # foreground service in the separate :engine
    │                       # process (native crash can't kill the UI);
    │                       # chunk → generate → AudioTrack pipeline;
    │                       # model unloaded after 5 min idle
    ├── ShareActivity.kt    # share target: clean text → voice popup → speak
    ├── MainActivity.kt     # model download, backend choice, voice library
    ├── TextCleaner.kt      # jsoup readability + plain-text cleanup
    ├── Chunker.kt          # natural-break splitting (~400 chars ≈ 25 s)
    ├── ModelManager.kt     # resumable 2-file (talker + mmproj) downloads
    └── VoiceStore.kt / TtsEngine.kt
```

Generation quality guard (ported from vibe_audiobook_maker): a chunk whose
audio is implausibly short (instant EOS) or pegged at the token cap (runaway)
is re-rolled with a fresh seed, up to twice.

## Backends and the llama.cpp patch

`docker/patches/qwen3tts-fixes.patch` carries two fixes (validated on a
desktop RTX 5070, 2026-08-05) that upstream does not have yet:

- **CPU** (`tools/mtmd/clip.cpp`): the multi-stage TTS generator leaves
  stage-unused graph inputs uninitialized; stale allocator memory then trips
  `GGML_ASSERT(i01 >= 0 && i01 < ne01)` in CPU `get_rows`
  (upstream issue #26632). Fixed by zero-filling all graph inputs per eval.
- **Vulkan** (`tools/mtmd/models/qwen3tts-gen.cpp`): code-index views into
  the acoustic-code cache are 4-byte offset, and the Vulkan `get_rows`
  kernels reject misaligned buffer offsets. Fixed with `ggml_cont` on those
  views.

With both patches, desktop results for the 1.7B Q8 model: CPU (8 threads)
RTF ≈ 1.25, Vulkan RTF ≈ 0.86. On-phone expectations: CPU on a big-core
arm64 with Q4_K_M should be near real time; **GPU is exposed as
"experimental"** — the OpenCL Adreno backend and mobile Vulkan drivers are
untested against this model's codec graph, but ggml falls back per-op to the
(fixed) CPU path.

Peak memory for 1.7B Q4_K_M + Q8 mmproj is ~2.5–3 GB — comfortably inside an
8 GB phone. A 0.6B variant would halve that, but nobody has published a 0.6B
**mmproj** GGUF yet; converting one from the original checkpoint is the
obvious follow-up (the safetensors are on brainiac).

To bump llama.cpp: change `LLAMA_CPP_COMMIT` in `docker/Dockerfile`, and drop
the patch once upstream fixes land.
