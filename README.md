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

First build is slow: the builder image fetches the Android SDK/NDK and a
pinned llama.cpp, then compiles llama.cpp (CPU + OpenCL backends) for arm64.

## Using it

1. Open TTS Runner once: download a model (1.7B Q4_K_M, ~1.5 GB) and import a
   voice (any wav/mp3/flac with 10–20 s of clean speech from one speaker).
2. From any app, share text or an article URL → "Read aloud (TTS Runner)".
   Selected text also works via the text-selection menu.
3. Pick a voice in the popup, then **Speak** (streamed playback: audio starts
   after the first chunk, ~25 s) or **Save** (renders the whole text in the
   background to `Music/TTS Runner/<title>.m4a` — AAC, since Android has no
   MP3 encoder). A notification shows a real progress bar and a Stop button;
   generation keeps running with the screen off.

Shared URLs are fetched and run through a small readability pass (densest
`<article>`/`<p>` container); markdown noise, inline URLs, and `[17]`-style
citations are stripped before speaking.

## Architecture

```
apps/tts-runner/
├── app/src/main/cpp/
│   ├── CMakeLists.txt      # llama.cpp (CPU + OpenCL w/ Adreno kernels)
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

`docker/patches/qwen3tts-fixes.patch` carries three fixes (validated on a
desktop RTX 5070, 2026-08-05/06) that upstream does not have yet:

- **Double-read** (`tools/mtmd/mtmd-helper-gen.cpp`): the generation loop
  overlaid the text embeddings onto every generated frame (a streaming-mode
  leftover), feeding the utterance to the model a second time — it then read
  the text twice (~2.3x duration, whisper-confirmed, most seeds). The
  official non-streaming pipeline adds only `tts_pad`; fixed to match, and
  the talker sampling now mirrors the official generation config (top_k 50,
  top_p 1.0, temp 0.9, repetition_penalty 1.05, control tokens suppressed).

- **CPU** (`tools/mtmd/clip.cpp`): the multi-stage TTS generator leaves
  stage-unused graph inputs uninitialized; stale allocator memory then trips
  `GGML_ASSERT(i01 >= 0 && i01 < ne01)` in CPU `get_rows`
  (upstream issue #26632). Fixed by zero-filling all graph inputs per eval.
- **Vulkan** (`tools/mtmd/models/qwen3tts-gen.cpp`): code-index views into
  the acoustic-code cache are 4-byte offset, and the Vulkan `get_rows`
  kernels reject misaligned buffer offsets. Fixed with `ggml_cont` on those
  views.

With both patches, desktop results for the 1.7B Q8 model: CPU (8 threads)
RTF ≈ 1.25, Vulkan RTF ≈ 0.86.

On-phone findings (Galaxy Z Fold5, SD 8 Gen 2 / Adreno 740, Android 16):

- **Adreno Vulkan driver**: cannot compile llama.cpp's compute shaders at
  all (`vk::Device::createComputePipeline: ErrorUnknown` on the first talker
  decode) — the Vulkan backend is compiled out of the Android build.
- **Adreno OpenCL**: Qualcomm's kernels run the talker LLM but SIGABRT on
  the codec graph, so "GPU (experimental)" mode offloads **talker layers
  only**; the codec always runs on CPU (`mmproj_use_gpu=false`).
- Native C++ exceptions from a backend are caught at the JNI boundary and
  the chunk is retried on CPU in-process; a hard native abort kills the
  `:engine` process, and a crash-marker file makes the next run start on
  CPU automatically.

Peak memory for 1.7B Q4_K_M + Q8 mmproj is ~2.5–3 GB — comfortably inside an
8 GB phone. A 0.6B variant would halve that, but nobody has published a 0.6B
**mmproj** GGUF yet; converting one from the original checkpoint is the
obvious follow-up (the safetensors are on brainiac).

To bump llama.cpp: change `LLAMA_CPP_COMMIT` in `docker/Dockerfile`, and drop
the patch once upstream fixes land.
