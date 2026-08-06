package com.maxfridbe.ttsrunner

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
    @JvmStatic external fun nLastError(): String

    /** Returns a complete WAV file (24 kHz mono s16) or null on failure/cancel. */
    @JvmStatic external fun nGenerate(
        text: String, speakerWavPath: String, lang: String,
        maxFrames: Int, seed: Int, temp: Float, topP: Float,
        progress: ProgressCallback?,
    ): ByteArray?
}
