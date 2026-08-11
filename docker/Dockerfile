# Shared Android builder image for every app under apps/.
# All compilation happens inside this container; the host never needs
# a JDK, Gradle, or the Android SDK.
#
# On top of the AndroidBase image this adds everything the tts-runner
# native layer needs: NDK + CMake, llama.cpp sources pre-fetched to
# /opt/llama_cpp, Vulkan/SPIRV headers patched into the NDK sysroot,
# and a LunarG glslc new enough for ggml's Vulkan feature probes.
FROM gradle:8.7.0-jdk17

ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=/usr/local/bin:$PATH:${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${ANDROID_SDK_ROOT}/platform-tools

RUN apt-get update && \
    apt-get install -y wget unzip git g++ cmake ninja-build patch xz-utils --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# https://developer.android.com/studio#command-tools
ARG CMDLINE_TOOLS_VERSION="11076708"
RUN wget -q "https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_TOOLS_VERSION}_latest.zip" -O cmdline-tools.zip && \
    mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    unzip -q cmdline-tools.zip -d ${ANDROID_SDK_ROOT}/cmdline-tools && \
    mv ${ANDROID_SDK_ROOT}/cmdline-tools/cmdline-tools ${ANDROID_SDK_ROOT}/cmdline-tools/latest && \
    rm cmdline-tools.zip

RUN yes | sdkmanager --licenses >/dev/null

ARG ANDROID_PLATFORM="android-35"
ARG BUILD_TOOLS="35.0.0"
ARG NDK_VERSION="27.2.12479018"
RUN sdkmanager \
    "platform-tools" \
    "platforms;${ANDROID_PLATFORM}" \
    "build-tools;${BUILD_TOOLS}" \
    "ndk;${NDK_VERSION}" \
    "cmake;3.22.1"

# llama.cpp sources, pinned. Qwen3-TTS support landed 2026-08-04 (PR #26254);
# this pin is the same commit validated against the desktop CUDA build on
# Bump LLAMA_CPP_COMMIT to pick up upstream Vulkan/CPU fixes.
ARG LLAMA_CPP_COMMIT="f9e832c10e9444cb168ddcb579cc62c154f3068b"
RUN wget -q "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz" -O /tmp/llama.tar.gz && \
    mkdir -p /opt/llama_cpp && \
    tar xzf /tmp/llama.tar.gz -C /opt/llama_cpp --strip-components=1 && \
    rm /tmp/llama.tar.gz
# Qwen3-TTS fixes not yet upstream (validated on desktop CPU/Vulkan):
#  - clip.cpp: zero-fill graph inputs between generator stages; stale
#    allocator memory in unused stage inputs trips CPU get_rows bounds
#    asserts (upstream issue #26632)
#  - qwen3tts-gen.cpp: ggml_cont on misaligned index views; the Vulkan
#    backend asserts on get_rows with unaligned buffer offsets
COPY patches/qwen3tts-fixes.patch /tmp/qwen3tts-fixes.patch
RUN cd /opt/llama_cpp && patch -p1 < /tmp/qwen3tts-fixes.patch
ENV LLAMA_CPP_SRC=/opt/llama_cpp

# ggml-vulkan needs Vulkan/SPIRV headers newer than the NDK ships; copy them
# over the NDK sysroot (same trick as android_builder's diffusion app).
ARG VK_VER="1.3.275"
RUN NDK_INC="${ANDROID_SDK_ROOT}/ndk/${NDK_VERSION}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include" && \
    wget -q "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${VK_VER}.tar.gz" -O /tmp/vkh.tar.gz && \
    tar xzf /tmp/vkh.tar.gz -C /tmp && \
    cp -r /tmp/Vulkan-Headers-${VK_VER}/include/vulkan/* ${NDK_INC}/vulkan/ && \
    cp -r /tmp/Vulkan-Headers-${VK_VER}/include/vk_video ${NDK_INC}/ 2>/dev/null; \
    wget -q "https://github.com/KhronosGroup/Vulkan-Hpp/archive/refs/tags/v${VK_VER}.tar.gz" -O /tmp/vkhpp.tar.gz && \
    tar xzf /tmp/vkhpp.tar.gz -C /tmp && \
    cp /tmp/Vulkan-Hpp-${VK_VER}/vulkan/*.hpp ${NDK_INC}/vulkan/ && \
    wget -q "https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/vulkan-sdk-${VK_VER}.0.tar.gz" -O /tmp/spvh.tar.gz && \
    tar xzf /tmp/spvh.tar.gz -C /tmp && \
    cp -r /tmp/SPIRV-Headers-vulkan-sdk-${VK_VER}.0/include/spirv ${NDK_INC}/ && \
    # ggml-vulkan does find_package(SPIRV-Headers CONFIG REQUIRED): install the
    # CMake package properly (header-only, instant)
    cmake -G Ninja -S /tmp/SPIRV-Headers-vulkan-sdk-${VK_VER}.0 -B /tmp/spvh-build -DCMAKE_INSTALL_PREFIX=/usr/local >/dev/null && \
    cmake --install /tmp/spvh-build >/dev/null && \
    rm -rf /tmp/vkh.tar.gz /tmp/vkhpp.tar.gz /tmp/spvh.tar.gz /tmp/spvh-build /tmp/Vulkan-* /tmp/SPIRV-*

# The NDK's bundled glslc is too old for ggml's coopmat/int-dot feature-probe
# shaders; shadow it with a current LunarG one.
RUN mkdir -p /opt/lunarg-glslc && \
    wget -q "https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz" -O /tmp/vksdk.tar.xz && \
    tar xJf /tmp/vksdk.tar.xz -C /opt/lunarg-glslc --strip-components=3 --wildcards "*/x86_64/bin/glslc" && \
    chmod +x /opt/lunarg-glslc/glslc && \
    rm /tmp/vksdk.tar.xz && \
    printf '%s\n' '#!/bin/bash' 'exec /opt/lunarg-glslc/glslc "$@"' > /usr/local/bin/glslc && \
    chmod +x /usr/local/bin/glslc

# Each app is bind-mounted here by build.sh.
WORKDIR /workspace

CMD ["gradle", "--no-daemon", "assembleDebug"]
