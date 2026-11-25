package com.threep.compressor

interface ProgressCallback {
    fun onProgress(fraction: Double, outBytes: Long)
}

object NativeEngine {
    init {
        System.loadLibrary("threep_native")
    }

    external fun processFile(
        srcPath: String,
        outDir: String,
        extract: Boolean,
        callback: ProgressCallback?
    ): String

    external fun cancelJob()
}
