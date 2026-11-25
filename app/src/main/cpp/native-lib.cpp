#include <jni.h>
#include <atomic>
#include <filesystem>
#include <string>
#include "engine3p.hpp"

namespace fs = std::filesystem;

static std::atomic<bool> g_cancel{false};

extern "C" JNIEXPORT void JNICALL
Java_com_threep_compressor_NativeEngine_cancelJob(JNIEnv*, jobject) {
    g_cancel.store(true, std::memory_order_relaxed);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_threep_compressor_NativeEngine_processFile(
    JNIEnv* env,
    jobject /* thiz */,
    jstring jSrcPath,
    jstring jOutDir,
    jboolean jExtract,
    jobject jCallback
) {
    g_cancel.store(false, std::memory_order_relaxed);

    const char* srcC = env->GetStringUTFChars(jSrcPath, nullptr);
    const char* outC = env->GetStringUTFChars(jOutDir, nullptr);
    std::string srcStr = srcC ? srcC : "";
    std::string outStr = outC ? outC : "";
    if (srcC) env->ReleaseStringUTFChars(jSrcPath, srcC);
    if (outC) env->ReleaseStringUTFChars(jOutDir, outC);

    jclass cbClass = jCallback ? env->GetObjectClass(jCallback) : nullptr;
    jmethodID onProg = cbClass ? env->GetMethodID(cbClass, "onProgress", "(DJ)V") : nullptr;

    p3::Options opt;
    opt.fast = true; // Fast (-f) default for mobile responsiveness
    opt.outDir = fs::path(outStr);
    opt.threads = 4;
    opt.memoryBudget = ((p3::u64)256 << 20); // 256 MB cap for mobile RAM safety

    p3::ProgressFn progress = [&](double frac, p3::u64 outBytes) -> bool {
        if (jCallback && onProg) {
            env->CallVoidMethod(jCallback, onProg, (jdouble)frac, (jlong)outBytes);
        }
        return !g_cancel.load(std::memory_order_relaxed);
    };

    p3::Result r = jExtract
        ? p3::restore_file(fs::path(srcStr), opt, progress)
        : p3::compress_file(fs::path(srcStr), opt, progress);

    if (r.status == p3::Status::Ok) {
        fs::path finalPath = r.outPath;
        // Rename compressed output from .3p to .kcp as requested
        if (!jExtract) {
            fs::path kcpPath = finalPath;
            kcpPath.replace_extension(".kcp");
            std::error_code ec;
            fs::rename(finalPath, kcpPath, ec);
            if (!ec) finalPath = kcpPath;
        }
        std::string okPayload = "OK|" + finalPath.string() + "|" +
            std::to_string(r.inSize) + "|" + std::to_string(r.outSize);
        return env->NewStringUTF(okPayload.c_str());
    }
    if (r.status == p3::Status::Cancelled) {
        return env->NewStringUTF("CANCELLED|Operation cancelled.");
    }
    std::string errPayload = "ERR|" + r.message;
    return env->NewStringUTF(errPayload.c_str());
}
