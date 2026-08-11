# Play Store: build format, 16 KB pages, targetSdk — what's done and what to check

## App Bundle (AAB), not APK

Play requires an `.aab`. The build system already parameterizes the Gradle task:

```sh
BUILD_TASK=bundleRelease ./build.sh tts-runner
# -> output/tts-runner/tts-runner-<version>-release.aab   (signed with the shared key)
```

`build.sh` collects `*.aab` alongside `*.apk`. Upload the `.aab` to the console;
Play generates per-device APKs from it. The sideloadable APKs stay useful for
`adb install` and the GitHub releases.

## targetSdk 35 (Android 15)

`compileSdk`/`targetSdk` are 35 and AGP is 8.6.0 (Gradle 8.7). Android 15 forces
edge-to-edge for apps targeting 35; these screens set their own padding and do
not yet consume window insets, so the theme opts out via
`android:windowOptOutEdgeToEdgeEnforcement` (a no-op below API 35). That flag is
a transition measure — before a future Android removes it, the activities should
apply `WindowInsets` padding to their roots and the opt-out should be dropped.

## 16 KB page size

Apps targeting Android 15+ must load with 16 KB memory pages. Status:

- **App's own native code** (`libttsrunner_jni.so` and the llama.cpp libraries
  built from source): aligned. CMake is invoked with
  `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON` under NDK r27, which links with
  `max-page-size=16384`.
- **Prebuilt dependency `libonnxruntime.so`** (ONNX Runtime Android AAR): its
  alignment is set by the upstream build, not by us. Verify before submitting:

  ```sh
  # per .so inside the AAB/APK:
  ${ANDROID_NDK}/toolchains/llvm/prebuilt/*/bin/llvm-readelf -l lib/arm64-v8a/libonnxruntime.so \
    | grep LOAD    # every LOAD segment's Align must be >= 0x4000 (16384)
  ```

  If ORT is not 16 KB-aligned, bump `onnxruntime-android` to a release built for
  16 KB (1.20.x+ ships aligned libs) — it's the only third-party native lib in
  the app.

- The debug build shows Android's "16 KB compatibility" warning dialog because
  `assembleDebug` marks the app debuggable and the OS runs the check eagerly;
  the release AAB is what Play evaluates. Confirm with Play's **pre-launch
  report** (it runs the app on 16 KB devices) after the first upload.

## Signing

Enroll the app in **Play App Signing**. The upload key is the repository's shared
release keystore (`keystore/release.keystore`, password in
`keystore/keystore.properties`; the same key is set as the CI secrets). Play
re-signs with its own app key for distribution.
