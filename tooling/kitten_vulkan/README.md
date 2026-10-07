# Kitten decoder Vulkan prototype

This is an offline porting harness for the exact `student_w4` decoder downloaded
by the Android app. It is **not integrated into the APK**. The app's existing
Kitten GPU setting accelerates the speech language model; waveform decoding
still uses LibTorch on CPU.

## Galaxy S10 target

Target Vulkan **1.1**, FP32, on both Snapdragon 855/Adreno 640 and
Exynos 9820/Mali-G76 variants. Do not require FP16 arithmetic, integer dot-product
extensions, cooperative matrices, or Vulkan 1.2. Actual driver capabilities,
shader correctness, memory use, and speed still need measurement on both phones.
Export requests conservative 256×256×256 texture limits and a 128 MiB buffer
limit; these settings alone do not certify compatibility.

ExecuTorch's [Vulkan backend](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html)
supports a Vulkan 1.1 baseline. This is separate from the app's ggml Vulkan
language-model backend, whose pinned implementation requires Vulkan **1.2**.
An S10 firmware exposing only 1.1 cannot use that existing language-model path.

The Android CMake build now uses the ARMv8-A CPU baseline, with GGML native
architecture detection disabled. The previous global `+i8mm` requirement was
newer than S10 hardware. This may reduce CPU performance on newer phones;
architecture-specific optimizations need runtime dispatch before reintroduction.

## Implemented and checked

- Convert the frozen TorchScript graph while preserving its trained weights.
- Restore 137 shape/scalar round trips that otherwise block `torch.export`.
- Make random noise explicit inputs so GPU parity can use identical noise.
- Replace the 16-point STFT/ISTFT with real-valued convolution and overlap-add.
- Lift 1D transposed convolutions into equivalent 2D operations.
- Compare the transformed decoder with the original before export.

On 2026-10-07, CPU graph parity passed at **4, 17, 53, and 101 tokens**, using
Bruno conditioning, seed 123, and `atol=rtol=2e-4`. Fourier and transposed
convolution unit tests passed. The static 17-token `decoder.pt2` was produced.
This is a numerical transformation check, not an audio quality or GPU speed test.

Android debug/release builds passed for arm64-v8a and x86_64, along with the five
existing unit tests and lint (0 errors, 166 warnings). All 418 ARM compile commands
were inspected: no global i8mm/SVE requirement remained. Local APK version:
`26.1007.1727`. This build contains the CPU baseline fix, not a Vulkan decoder.

## Reproduce

Run inside a Linux development container with Python 3.13, CMake, a C++ compiler,
`glslc`, and a Vulkan loader/driver. Use the repository's container workflow for
the Android APK. Generated files, environments and model weights stay in ignored
`output/kitten-vulkan/`.

```sh
python3.13 -m venv output/kitten-vulkan/venv
output/kitten-vulkan/venv/bin/pip install \
  torch==2.14.1 executorch==1.5.1 \
  --extra-index-url https://download.pytorch.org/whl/cpu
output/kitten-vulkan/venv/bin/python tooling/kitten_vulkan/fetch_assets.py \
  output/kitten-vulkan/assets
output/kitten-vulkan/venv/bin/python -m unittest discover \
  -s tooling/kitten_vulkan -p 'test_*.py'
output/kitten-vulkan/venv/bin/python tooling/kitten_vulkan/export_decoder.py \
  --decoder output/kitten-vulkan/assets/decoder.pt \
  --voices output/kitten-vulkan/assets/voices.json \
  --output output/kitten-vulkan/export
```

Use `--audit-only` for the original decoder/operator inventory, or `--export-only`
to check graph parity and produce `.pt2` without waiting for Vulkan lowering.
The exporter records progress, numerical errors, input shapes, and, on successful
lowering, Vulkan delegate counts and remaining CPU operators in `report.json`.
It prepares `decoder.bpte` with real inputs and reference output for the native
runner. The bundled-test/report additions have syntax checks but await rerunning
with the development container available.

The Python wheel does not include the Vulkan runtime. Build the native runner:

```sh
output/kitten-vulkan/venv/bin/python tooling/kitten_vulkan/fetch_runtime.py \
  output/kitten-vulkan/executorch
bash tooling/kitten_vulkan/build_runtime.sh
output/kitten-vulkan/runtime-build/executor_runner \
  --model_path=output/kitten-vulkan/export/decoder.bpte \
  --bundleio_atol=0.0002 --bundleio_rtol=0.0002
```

ExecuTorch is pinned to `3b60683923245cf472b7323426920e15623ba361` (1.5.1);
the fetcher resolves dependencies at that commit and records their revisions.
The source directory must be named `executorch`. The first native configuration
identified missing FXdiv and the directory-name requirement; the scripts now
address both, but the corrected build has not run because the session's sandbox
blocked restarting the development container. Native GPU execution remains
unvalidated.

## Remaining work

The first Vulkan partition attempt identifies unsupported `reciprocal`, `log1p`,
attention `any`/scalar multiplication, and casts from frozen FP64 scalar
constants. These create CPU fallbacks; a successful export alone is not evidence
of a useful GPU port. Fold constant casts and introduce numerically checked
decompositions before measuring delegate boundaries and transfer overhead.

Then validate native Vulkan output, make token/noise shapes dynamic (the current
export is fixed at 17 tokens), and benchmark peak memory and sustained inference
on actual S10 variants. Android runtime/asset packaging, capability checks,
cancellation, and a separate decoder setting come after those checks. Test that
integration through `android_screen_runner` before publishing a GPU-decoder APK.
