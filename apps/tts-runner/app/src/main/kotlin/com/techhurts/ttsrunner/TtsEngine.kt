package com.techhurts.ttsrunner

/** JNI facade over llama.cpp Qwen3-TTS (see cpp/tts_jni.cpp). All methods are
 *  blocking; call from a worker thread. The engine holds one loaded model. */
object TtsEngine {
    init {
        System.loadLibrary("ttsrunner_jni")
    }

    interface ProgressCallback {
        /** framesDone/framesMax; one frame is 1/12.5 s of audio. */
        fun onProgress(framesDone: Int, framesMax: Int)
    }

    @JvmStatic external fun nLoad(modelPath: String, mmprojPath: String, backend: String, nThreads: Int): Boolean
    @JvmStatic external fun nUnload()
    @JvmStatic external fun nCancel()
    /** Clear a pending cancel at the start of a new job. nGenerate never
     *  clears it itself, so a cancel can't be lost to a call-boundary race. */
    @JvmStatic external fun nResetCancel()
    @JvmStatic external fun nLastError(): String
    /** One line per ggml compute device with free/total memory. */
    @JvmStatic external fun nDeviceInfo(): String
    /** Requantize a GGUF (e.g. Q8_0 -> Q4_0 for the Adreno GPU kernels).
     *  Blocking, several minutes; call on a worker thread. */
    @JvmStatic external fun nQuantize(srcPath: String, dstPath: String, type: String): Boolean

    /** Returns a complete WAV file (24 kHz mono s16) or null on failure/cancel.
     *  instruct: VoiceDesign description ("" for none, needs a VD model). */
    @JvmStatic external fun nGenerate(
        text: String, speakerWavPath: String, lang: String, instruct: String,
        maxFrames: Int, seed: Int, temp: Float, topP: Float,
        progress: ProgressCallback?,
    ): ByteArray?
}
