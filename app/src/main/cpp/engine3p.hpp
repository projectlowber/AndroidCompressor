// =============================================================================
// 3P Compressor v18.8-Final - Universal Calibrated Lossless Engine (engine3p.hpp)
// =============================================================================
// Universal Pipeline (Automatic Format & Entropy Routing):
//   0. Baseline JPEG      -> Lossless quantized DCT coefficient recompression
//   0b. Deflate Streams   -> Bit-exact zlib inflate/re-deflate unwrap (PNG/ZIP/PDF/GZ)
//   1. MT19937 PRNG       -> Exact 2.5 KB generator-state capture + verification
//   2. WAV / BMP Headers  -> Adaptive stereo PCM & 2D spatial pixel models
//   3. Global Dedup (v18.6+):
//        * In-RAM 2-way associative 16-byte rolling hash (files <= 256 MB)
//        * 8-byte u64 word match extension + repeat-distance continuation probe
//        * Split-stream + 4-LRU MTF distance-coded rANS header (kind = 10)
//   4. Entropy-Adaptive Segment Calibration (v18.7+ choose_seg_code):
//        * Deduped streams : 1 MB (<4M), 4 MB (4..48M), 8 MB (48..128M), 16 MB (>128M)
//        * Low H0 (< 3.5)  : 4 waves/core (256 KB..2 MB) for L2/L3 cache speed
//        * Mid H0 (3.5-6.5): 2 waves/core (512 KB..8 MB) to cut rANS table tax
//        * High H0 (>= 6.5): 1 wave/core  (512 KB..16 MB) + raw-passthrough gatekeeper
//   5. Dual Entropy Engines:
//        * Fast (-f, default): Multi-threaded LZ77 + Lazy Matching + 14-bit rANS
//        * Max  (-m)         : Multi-threaded Bit-Level PAQ Context Mixer (AVX2)
//   6. Integrity Guarantee:
//        * Per-segment + whole-file CRC-32 verification; automatic raw-passthrough
//          gatekeeper guarantees incompressible H=8.0 noise never bloats.
// =============================================================================
// engine3p.hpp - the 3P compression engine (portable C++17, header-only).
//
//  .3p file layout
//    [0xB3] [varint nameLen] [name, UTF-8] [varint originalSize] [u8 tableBits] [varint pixelStride] ([u8 bytesPerPixel])
//    then one FRAME per 8 MiB segment of the original:
//    [varint payloadLen*2 + modeled] [crc32 of the original segment, 4 bytes LE] [payload]
//
//  * Deflate streams (PNG / ZIP / GZ / zlib / PDF) that zlib's algorithm reproduces exactly are stored inflated (header kind 9).
//    PNG IDAT data is additionally unfiltered to pixels (pix contexts in the model); level-0 stored-block streams are recorded by block length.
//  * Segments are independent, so compression AND extraction run on every CPU core.
//  * Inside a modeled segment: 16 KB blocks, each coded by the model, stored raw, or skipped without
//    touching the model (long incompressible runs run ~15x faster than modeled data).
//  * Integer-only model (context mixing + match + SSE): bit-identical on every machine and compiler.
//  * Memory is bounded: one model (~110 MB) + two 8 MB buffers per worker thread; tables are
//    lazily committed (calloc), so small or incompressible inputs touch very little RAM.
#pragma once // Compiler pragma directive
// Optimization stack feature toggles (identical .3p output, ratio untouched):
//   P3_LAYER1_PREFETCH      : 1 (default ON)  - Byte/nibble-boundary prefetch in Model::update()
//   P3_LAYER2_BRANCHLESS_AC : 0 (default OFF) - Branchless arithmetic-coder norm (opt-in; regressed vs L1 on x86-64 GCC)
#ifndef P3_LAYER1_PREFETCH // Guard if macro is not defined
#define P3_LAYER1_PREFETCH 1 // Define preprocessor macro constant
#endif // End conditional compilation guard
#ifndef P3_LAYER2_BRANCHLESS_AC // Guard if macro is not defined
#define P3_LAYER2_BRANCHLESS_AC 0 // Define preprocessor macro constant
#endif // End conditional compilation guard
//   P3_AVX2_MIXER           : 1 (default ON when compiled with -mavx2 / /arch:AVX2) - 100% bit-identical AVX2 3x16 mixer
#ifndef P3_AVX2_MIXER // Guard if macro is not defined
#if defined(__AVX2__) // Conditional compilation check
#define P3_AVX2_MIXER 1 // Define preprocessor macro constant
#else // Fallback conditional compilation branch
#define P3_AVX2_MIXER 0 // Define preprocessor macro constant
#endif // End conditional compilation guard
#endif // End conditional compilation guard
//   P3_FAST_TRIALS          : 1 (default ON)  - Smaller trial-encode windows (32-64 KB) for faster compression (same .3p format)
#ifndef P3_FAST_TRIALS // Guard if macro is not defined
#define P3_FAST_TRIALS 1 // Define preprocessor macro constant
#endif // End conditional compilation guard
#if P3_AVX2_MIXER // Conditional compilation check
#include <immintrin.h> // Include required header
#endif // End conditional compilation guard

#include <algorithm> // Include required header
#include <atomic> // Include required header
#include <chrono> // Include required header
#include <cmath> // Include required header
#include <cstdint> // Include required header
#include <cstdio> // Include required header
#include <cctype> // Include required header
#include <cstdlib> // Include required header
#include <cstring> // Include required header
#include <filesystem> // Include required header
#include <functional> // Include required header
#include <map> // Include required header
#include <memory> // Include required header
#include <new> // Include required header
#include <string> // Include required header
#include <vector> // Include required header
#ifdef _WIN32 // Guard if macro is defined
#ifndef _WIN32_WINNT // Guard if macro is not defined
#define _WIN32_WINNT 0x0601   // Windows 7+: condition variables
#endif // End conditional compilation guard
#ifndef WINVER // Guard if macro is not defined
#define WINVER 0x0601 // Define preprocessor macro constant
#endif // End conditional compilation guard
#ifndef WIN32_LEAN_AND_MEAN // Guard if macro is not defined
#define WIN32_LEAN_AND_MEAN // Define preprocessor macro constant
#endif // End conditional compilation guard
#ifndef NOMINMAX // Guard if macro is not defined
#define NOMINMAX // Define preprocessor macro constant
#endif // End conditional compilation guard
#include <windows.h> // Include required header
#include <wchar.h> // Include required header
#else // Fallback conditional compilation branch
#include <condition_variable> // Include required header
#include <mutex> // Include required header
#include <thread> // Include required header
#endif // End conditional compilation guard

namespace p3 { // Open namespace scope
namespace fs = std::filesystem; // Open namespace scope
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64; // Define type alias
typedef int16_t i16; typedef int32_t i32; typedef int64_t i64; // Define type alias

constexpr u8 kMagic = 0xB3; // Define helper function or constant
constexpr size_t kSegment = (size_t)8 << 20;    // independent unit of parallel work
constexpr size_t kBlock = 1u << 14;             // model / raw / skip decision granularity
constexpr size_t kMaxName = 1000; // Define helper function or constant
inline const char* extension() { return ".3p"; } // Define helper function or constant


#ifdef P3_PROFILE // Guard if macro is defined
struct ProfStats { // Declare data structure / class
    // Run metadata
    std::string modeName; // Execute statement
    std::string detectedPath; // Execute statement
    u64 inBytes = 0, streamBytes = 0, outBytes = 0; // Compute and assign value
    int tbMax = 0, segCode = 0; // Segment sizing configuration
    size_t segSize = 0; // Segment sizing configuration
    u64 nseg = 0; // Compute and assign value
    unsigned hwThreads = 0, usedThreads = 0; // Compute and assign value
    bool fastMode = false; // Compute and assign value

    // Wall-clock pre-passes (microseconds)
    u64 wallTotalUs = 0; // Compute and assign value
    u64 preWavBmpUs = 0; // Compute and assign value
    u64 preJpegUs = 0; // Compute and assign value
    u64 preMtUs = 0; // Compute and assign value
    u64 preDeflateUs = 0; // Compute and assign value
    u64 preDedupUs = 0; // Compute and assign value
    u64 preDetectUs = 0; // Compute and assign value
    u64 preHeaderUs = 0; // Compute and assign value
    u64 workerWallUs = 0; // Compute and assign value
    u64 finalizeUs = 0; // Compute and assign value

    // Aggregated worker timings across all threads (microseconds)
    std::atomic<u64> wReadCrcUs{0}; // Execute statement
    std::atomic<u64> wTransformUs{0}; // Execute statement
    std::atomic<u64> wModelResetUs{0}; // Execute statement
    std::atomic<u64> wPaqModelUs{0}; // Execute statement
    std::atomic<u64> wPaqAcUs{0}; // Execute statement
    std::atomic<u64> wFastLzUs{0}; // Execute statement
    std::atomic<u64> wFastRansUs{0}; // Execute statement
    std::atomic<u64> wVerifyUs{0}; // Execute statement
    std::atomic<u64> wQueueWaitUs{0}; // Execute statement
    std::atomic<u64> wWriteDrainUs{0}; // Execute statement
    std::atomic<u64> wTotalActiveUs{0}; // Execute statement

    // Block / segment stats
    std::atomic<u64> blkModeled{0}, blkRaw{0}, blkSkipped{0}; // Execute statement
    std::atomic<u64> segMinUs{~(u64)0}, segMaxUs{0}, segSumUs{0}, segDone{0}; // Execute statement

    struct ThreadInfo { // Declare data structure / class
        unsigned id = 0; // Compute and assign value
        u64 segs = 0; // Compute and assign value
        u64 activeUs = 0; // Compute and assign value
        u64 waitUs = 0; // Compute and assign value
        u64 writeUs = 0; // Compute and assign value
    }; // Close scope / block
    std::vector<ThreadInfo> threads; // Allocate / manage dynamic buffer

    void clear() { // Execute statement
        modeName.clear(); detectedPath = "Plain"; // Compute and assign value
        inBytes = streamBytes = outBytes = 0; // Compute and assign value
        tbMax = segCode = 0; segSize = 0; nseg = 0; // Segment sizing configuration
        hwThreads = usedThreads = 0; fastMode = false; // Compute and assign value
        wallTotalUs = preWavBmpUs = preJpegUs = preMtUs = preDeflateUs = 0; // Compute and assign value
        preDedupUs = preDetectUs = preHeaderUs = workerWallUs = finalizeUs = 0; // Compute and assign value
        wReadCrcUs = wTransformUs = wModelResetUs = wPaqModelUs = wPaqAcUs = 0; // Compute and assign value
        wFastLzUs = wFastRansUs = wVerifyUs = wQueueWaitUs = wWriteDrainUs = wTotalActiveUs = 0; // Compute and assign value
        blkModeled = blkRaw = blkSkipped = 0; // Compute and assign value
        segMinUs = ~(u64)0; segMaxUs = 0; segSumUs = 0; segDone = 0; // Compute and assign value
        threads.clear(); // Execute statement
    } // Close scope / block
}; // Close scope / block
inline ProfStats g_prof; // Define helper function or constant
inline u64 prof_now_us() { // Define helper function or constant
    using namespace std::chrono; // Type alias or namespace import
    return (u64)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count(); // Return result to caller
} // Close scope / block
inline void prof_update_minmax(std::atomic<u64>& mn, std::atomic<u64>& mx, u64 v) { // Define helper function or constant
    u64 cur = mn.load(std::memory_order_relaxed); // Compute and assign value
    while (v < cur && !mn.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {} // Loop while condition holds
    cur = mx.load(std::memory_order_relaxed); // Compute and assign value
    while (v > cur && !mx.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {} // Loop while condition holds
} // Close scope / block
#endif // End conditional compilation guard

// ------------------------------------------------------------------ public API
enum class Status { Ok, Cancelled, IoError, BadFormat, Corrupt, OutOfMemory, DiskFull }; // Define enumeration constants
struct Result { // Declare data structure / class
    Status status = Status::Ok; // Compute and assign value
    std::string message; // Execute statement
    u64 inSize = 0, outSize = 0;     // compress: original / .3p size.  restore: .3p / restored size
    fs::path outPath; // Execute statement
    std::string storedName;          // original file name (UTF-8) kept in the metadata
}; // Close scope / block
struct Options { // Declare data structure / class
    unsigned threads = 0;            // 0 = automatic
    u64 memoryBudget = 0;            // bytes the engine may use for worker models; 0 = no limit
    bool verify = false;             // decode every segment right after encoding it
    bool dedup = true;               // find long-distance duplicates (any distance) before compressing
    bool detect = true;              // detect headerless PCM audio and fixed-size records / image rows
    bool fast = true;                // true = fast 8MB LZ+rANS (default), false = classical bit-level PAQ (-m)
    fs::path outDir;                 // empty = next to the source file
}; // Close scope / block
// progress(fractionDone 0..1, outputBytesSoFar). Return false to cancel. Called ~10x per second.
typedef std::function<bool(double, u64)> ProgressFn; // Define type alias

// ------------------------------------------------------------------ platform threading (no std::thread on Windows,
// so there is no dependency on the MinGW threading model)
namespace sys { // Open namespace scope
#ifdef _WIN32 // Guard if macro is defined
class Mutex { CRITICAL_SECTION cs_; public: // Declare data structure / class
    Mutex() { InitializeCriticalSection(&cs_); } ~Mutex() { DeleteCriticalSection(&cs_); } // Execute statement
    Mutex(const Mutex&) = delete; Mutex& operator=(const Mutex&) = delete; // Compute and assign value
    void lock() { EnterCriticalSection(&cs_); } void unlock() { LeaveCriticalSection(&cs_); } CRITICAL_SECTION* native() { return &cs_; } }; // Execute statement
class CondVar { CONDITION_VARIABLE cv_; public: // Declare data structure / class
    CondVar() { InitializeConditionVariable(&cv_); } // Execute statement
    CondVar(const CondVar&) = delete; CondVar& operator=(const CondVar&) = delete; // Compute and assign value
    void notify_all() { WakeAllConditionVariable(&cv_); } // Execute statement
    void wait(Mutex& m) { SleepConditionVariableCS(&cv_, m.native(), INFINITE); } // Execute statement
    void wait_ms(Mutex& m, unsigned ms) { SleepConditionVariableCS(&cv_, m.native(), ms); } }; // Execute statement
class Thread { HANDLE h_ = nullptr; std::function<void()> fn_; // Declare data structure / class
    static DWORD WINAPI tramp(LPVOID p) { static_cast<Thread*>(p)->fn_(); return 0; } // Define helper function or constant
public: // Member access specifier
    Thread() {} Thread(const Thread&) = delete; Thread& operator=(const Thread&) = delete; // Compute and assign value
    bool start(std::function<void()> f) { fn_ = std::move(f); h_ = CreateThread(nullptr, 0, tramp, this, 0, nullptr); return h_ != nullptr; } // Compute and assign value
    void join() { if (h_) { WaitForSingleObject(h_, INFINITE); CloseHandle(h_); h_ = nullptr; } } // Compute and assign value
    ~Thread() { join(); } }; // Execute statement
inline unsigned hardware_threads() { SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors ? (unsigned)si.dwNumberOfProcessors : 1u; } // Define helper function or constant
#else // Fallback conditional compilation branch
class Mutex { std::mutex m_; public: void lock() { m_.lock(); } void unlock() { m_.unlock(); } }; // Declare data structure / class
class CondVar { std::condition_variable_any cv_; public: // Declare data structure / class
    void notify_all() { cv_.notify_all(); } // Execute statement
    void wait(Mutex& m) { cv_.wait(m); } // Execute statement
    void wait_ms(Mutex& m, unsigned ms) { cv_.wait_for(m, std::chrono::milliseconds(ms)); } }; // Execute statement
class Thread { std::thread t_; public: // Declare data structure / class
    bool start(std::function<void()> f) { try { t_ = std::thread(std::move(f)); return true; } catch (...) { return false; } } // Compute and assign value
    void join() { if (t_.joinable()) t_.join(); } ~Thread() { join(); } }; // Execute statement
inline unsigned hardware_threads() { unsigned n = std::thread::hardware_concurrency(); return n ? n : 1u; } // Define helper function or constant
#endif // End conditional compilation guard
struct Lock { Mutex& m; explicit Lock(Mutex& mm) : m(mm) { m.lock(); } ~Lock() { m.unlock(); } }; // Declare data structure / class
}  // namespace sys

// ------------------------------------------------------------------ internals
namespace detail { // Open namespace scope

struct Failure { Status status; std::string msg; }; // Declare data structure / class
struct Cancelled {}; // Declare data structure / class

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); } // Define helper function or constant
inline int ceil_log2(u64 x) { int k = 0; while (k < 63 && ((u64)1 << k) < x) k++; return k; } // Define helper function or constant

#if defined(_MSC_VER) // Conditional compilation check
#include <intrin.h> // Include required header
#include <xmmintrin.h> // Include required header
inline void p3_prefetch(const void* p) { _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0); } // Define helper function or constant
inline u32  p3_clz32(u32 x) { unsigned long idx; _BitScanReverse(&idx, x); return 31u - (u32)idx; } // Define helper function or constant
inline u32  p3_bswap32(u32 x) { return _byteswap_ulong(x); } // Define helper function or constant
#else // Fallback conditional compilation branch
inline void p3_prefetch(const void* p) { __builtin_prefetch(p, 0, 3); } // Define helper function or constant
inline u32  p3_clz32(u32 x) { return (u32)__builtin_clz(x); } // Define helper function or constant
inline u32  p3_bswap32(u32 x) { return __builtin_bswap32(x); } // Define helper function or constant
#endif // End conditional compilation guard

template <class T> struct Arr {                  // zero-initialised, lazily committed & reused across segments
    T* p = nullptr; size_t cap = 0; // Compute and assign value
    Arr() {} Arr(const Arr&) = delete; Arr& operator=(const Arr&) = delete; // Compute and assign value
    void alloc(size_t n) { // Execute statement
        size_t need = n ? n : 1; // Compute and assign value
        if (p && cap >= need) { std::memset(p, 0, need * sizeof(T)); return; } // Conditional branch check
        std::free(p); p = (T*)std::calloc(need, sizeof(T)); if (!p) { cap = 0; throw std::bad_alloc(); } // Compute and assign value
        cap = need; // Compute and assign value
    } // Close scope / block
    ~Arr() { std::free(p); } // Execute statement
    T& operator[](size_t i) { return p[i]; } // Execute statement
}; // Close scope / block

struct Tables { // Declare data structure / class
    short stretch_[4096]; int dt_[1024]; double lg1_[4097]; // Execute statement
    static int squash(int d) { // Define helper function or constant
        static const int t[33] = {1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047, // Define helper function or constant
            2549,2994,3348,3607,3785,3901,3975,4022,4050,4068,4079,4085,4089,4092,4093,4094}; // Execute statement
        if (d > 2047) return 4095; // Conditional branch check
        if (d < -2047) return 0; // Conditional branch check
        int w = d & 127; d = (d >> 7) + 16; // Compute and assign value
        return (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7; // Return result to caller
    } // Close scope / block
    Tables() { // Execute statement
        int pi = 0; // Compute and assign value
        for (int x = -2047; x <= 2047; x++) { int v = squash(x); for (int i = pi; i <= v; i++) stretch_[i] = (short)x; pi = v + 1; } // Loop iteration
        for (int i = pi; i < 4096; i++) stretch_[i] = 2047; // Loop iteration
        for (int i = 0; i < 1024; i++) dt_[i] = 16384 / (i + i + 3); // Loop iteration
        for (int i = 1; i < 4096; i++) lg1_[i] = -std::log((double)i / 4096.0) / std::log(2.0);   // encoder-only decision aid
        lg1_[0] = lg1_[4096] = 12.0; // Compute and assign value
    } // Close scope / block
}; // Close scope / block
inline const Tables g_tables; // Define helper function or constant
inline const Tables& tables() { return g_tables; } // Define helper function or constant
inline int squash(int d) { return Tables::squash(d); } // Define helper function or constant

constexpr u32 kInit = 0x80000000u;               // counters are stored XOR kInit so zeroed memory means p=0.5, n=0
inline void sm_up(u32* t, int y, int limit) { // Define helper function or constant
    const Tables& T = tables(); // Compute and assign value
    int n = (int)(*t & 1023), p = (int)(*t >> 10); // Compute and assign value
    if (n < limit) ++*t; else *t = (*t & 0xfffffc00u) | (u32)limit; // Conditional branch check
    i64 d = (((i64)(y << 22) - p) >> 3) * T.dt_[n]; // Compute and assign value
    *t = (u32)((i64)*t + (d & ~(i64)1023)); // Compute and assign value
} // Close scope / block
inline u32 hmix(u32 h, u32 v) { // Define helper function or constant
    h ^= v + 0x9E3779B9u + (h << 6) + (h >> 2); // Compute and assign value
    h *= 0x85EBCA6Bu; h ^= h >> 15; h *= 0xC2B2AE35u; h ^= h >> 13; // Compute and assign value
    return h; // Return result to caller
} // Close scope / block

struct APM { // Declare data structure / class
    std::vector<u16> t; int idx = 0; // Allocate / manage dynamic buffer
    void init(int n) { // Execute statement
        t.resize((size_t)n * 24); idx = 0; // Compute and assign value
        for (int i = 0; i < n; i++) for (int j = 0; j < 24; j++) t[(size_t)i * 24 + j] = (u16)(squash(((j * 2 + 1) * 4096) / 48 - 2048) * 16); // Loop iteration
    } // Close scope / block
    int pp(int pr, int cx) { // Execute statement
        int s = (tables().stretch_[pr] + 2048) * 23, wt = s & 0xfff; // Compute and assign value
        cx = cx * 24 + (s >> 12); idx = cx + (wt >> 11); // Compute and assign value
        return (t[cx] * (4096 - wt) + t[cx + 1] * wt) >> 16; // Return result to caller
    } // Close scope / block
    void up(int y) { int g = (y << 16) + (y << 7) - y - y; t[idx] = (u16)(t[idx] + ((g - t[idx]) >> 7)); } // Compute and assign value
}; // Close scope / block

// ------------------------------------------------------------------ what the model is told about the data
// PCM WAV audio (8/16/24-bit, 1-8 channels): the "data" chunk is turned into prediction residuals before modelling.
struct Audio { // Declare data structure / class
    u64 dataOff = 0, dataLen = 0; int channels = 0, width = 0; // Compute and assign value
    bool on() const { return channels > 0; } // Execute statement
    int frame() const { return channels * width; } // Execute statement
}; // Close scope / block
// whole audio frames of segment [segStart, segStart+L) as local offsets [ts, te); empty region -> ts == te == 0
inline void audio_region(const Audio& a, u64 segStart, size_t L, size_t& ts, size_t& te) { // Define helper function or constant
    ts = te = 0; if (!a.on()) return; // Compute and assign value
    u64 f = (u64)a.frame(), e = segStart + L; // Compute and assign value
    u64 lo = std::max(segStart, a.dataOff), hi = std::min(e, a.dataOff + a.dataLen); // Compute and assign value
    if (hi <= lo) return; // Conditional branch check
    u64 rel = (lo - a.dataOff) % f; if (rel) lo += f - rel; // Compute and assign value
    if (lo >= hi) return; // Conditional branch check
    hi = lo + ((hi - lo) / f) * f; if (hi <= lo) return; // Compute and assign value
    ts = (size_t)(lo - segStart); te = (size_t)(hi - segStart); // Compute and assign value
} // Close scope / block
struct Hint {                                       // per-segment hints for the model contexts
    int stride = 0, bpp = 0;                        // BMP pixel geometry
    bool pix = false; int phase = 0;                // PNG pixels: richer image contexts; phase = channel index of local byte 0
    u64 ats = 0, ate = 0; int channels = 0, width = 0;   // audio residual region (local offsets)
}; // Close scope / block

// ------------------------------------------------------------------ the model
class Model { // Declare data structure / class
public: // Member access specifier
    static constexpr int NBIT = 5, NBYTE = 9, NCTX = NBIT + NBYTE, N_IN = NCTX + 2; // Define helper function or constant
    static constexpr int MINLEN = 4, LIMIT = 30, LR1 = 32, LR2 = 4, BIAS = 77; // Define helper function or constant

    Model() = default; // Compute and assign value
    Model(u64 len, int tbMax, const Hint& hint) { reset(len, tbMax, hint); } // Execute statement

    void reset(u64 len, int tbMax, const Hint& hint) { // Execute statement
        byteTb_ = clampi(ceil_log2(len ? len * 8 : 1) + 1, 16, tbMax); // Compute and assign value
        mtBits_ = clampi(ceil_log2(len ? len : 1) + 1, 12, tbMax); // Compute and assign value
        u64 w = 4096; while (w < len && w < ((u64)1 << 24)) w <<= 1; // Compute and assign value
        ring_ = (size_t)w; mask_ = w - 1; // Compute and assign value
        static const int ords[NBIT] = {2, 4, 8, 12, 16}; // Define helper function or constant
        for (int i = 0; i < NCTX; i++) tab_[i].alloc((size_t)1 << (i < NBIT ? ords[i] : byteTb_)); // Loop iteration
        buf_.alloc(ring_); mtab_.alloc((size_t)1 << mtBits_); // Execute statement
        for (int i = 0; i < 32; i++) mm_[i] = kInit; // Loop iteration
        mix1_.assign((size_t)256 * N_IN, 9830); mix2_.assign((size_t)216 * N_IN, 9830); mix3_.assign((size_t)256 * N_IN, 9830); // Execute statement
        for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++) mixf_[i][j] = 21845; // Loop iteration
        a1_.init(256); a2_.init(65536); // Execute statement
        hist_ = 0; c0_ = 1; c1_ = 0; c2_ = 0; c3_ = 0; c4_ = 0; word_ = 0; // Compute and assign value
        pos_ = 0; stride_ = 0; bpp_ = 0; phase_ = 0; pix_ = false; // Compute and assign value
        ats_ = 0; ate_ = 0; ach_ = 0; aw_ = 0; // Compute and assign value
        blen_ = 0; mptr_ = 0; mlen_ = 0; mexp_ = -1; mctx_ = 0; pf_ = 2048; // Compute and assign value
        if (hint.stride > 0 && hint.bpp > 0 && (u64)hint.stride + (u64)hint.bpp + 8 < ring_ / 2) { stride_ = hint.stride; bpp_ = hint.bpp; pix_ = hint.pix; phase_ = hint.phase; } // Conditional branch check
        if (hint.channels > 0 && hint.width > 0 && hint.ate > hint.ats) { ats_ = hint.ats; ate_ = hint.ate; ach_ = hint.channels; aw_ = hint.width; } // Conditional branch check
        setByteCtx(); // Execute statement
#if P3_LAYER1_PREFETCH // Conditional compilation check
        prefetchBases(0u); // Execute statement
#endif // End conditional compilation guard
    } // Close scope / block

    int predict() { // Execute statement
        const Tables& T = tables(); // Compute and assign value
        static const int ords[NBIT] = {2, 4, 8, 12, 16}; // Define helper function or constant
        for (int i = 0; i < NBIT; i++) slot_[i] = &tab_[i][hist_ & ((1u << ords[i]) - 1)]; // Loop iteration
#if !P3_LAYER1_PREFETCH // Conditional compilation check
        if (pos_ == 0 || pos_ == 4)                       // one cache line (16 counters) per nibble context
            for (int i = 0; i < NBYTE; i++) base_[i] = (hmix(bh_[i], pos_ == 0 ? 0u : c0_) >> (32 - byteTb_)) & ~15u; // Loop iteration
#endif // End conditional compilation guard
        int j = pos_ & 3; // Compute and assign value
        u32 nib = j == 0 ? 1u : ((1u << j) | (c0_ & ((1u << j) - 1))); // Compute and assign value
        for (int i = 0; i < NBYTE; i++) slot_[NBIT + i] = &tab_[NBIT + i][base_[i] + nib]; // Loop iteration
        for (int i = 0; i < NCTX; i++) { // Loop iteration
            u32 t = *slot_[i] ^ kInit; // Compute and assign value
            st_[i] = (t & 1023) ? T.stretch_[clampi((int)(t >> 20), 4, 4091)] : 0; // Compute and assign value
        } // Close scope / block
        mctx_ = 0; mexp_ = -1; st_[NCTX] = 0; // Compute and assign value
        if (mptr_) { // Conditional branch check
            u32 pb = (u32)buf_[(size_t)(mptr_ & mask_)] | 256u; // Compute and assign value
            if ((pb >> (8 - pos_)) == c0_) { // Conditional branch check
                mexp_ = (int)((pb >> (7 - pos_)) & 1); // Compute and assign value
                mctx_ = 1 + (int)(mlen_ < 30 ? mlen_ : 30); // Compute and assign value
                int s = T.stretch_[clampi((int)(mm_[mctx_] >> 20), 41, 4055)]; // Compute and assign value
                st_[NCTX] = mexp_ ? s : -s; // Compute and assign value
            } // Close scope / block
        } // Close scope / block
        st_[NCTX + 1] = BIAS; // Compute and assign value
        int mb = mexp_ < 0 ? 0 : 1 + (int)((mlen_ < 15 ? mlen_ : 15) / 2); // Compute and assign value
        wsel_[0] = &mix1_[(size_t)c0_ * N_IN]; // Compute and assign value
        wsel_[1] = &mix2_[(size_t)(((mb * 3) + (mexp_ + 1)) * 8 + pos_) * N_IN]; // Compute and assign value
        wsel_[2] = &mix3_[(size_t)c1_ * N_IN]; // Compute and assign value
#if P3_AVX2_MIXER // Conditional compilation check
        static_assert(N_IN == 16, "AVX2 mixer assumes N_IN == 16"); // Compute and assign value
        const __m256i s0 = _mm256_loadu_si256((const __m256i*)(st_ + 0)); // Compute and assign value
        const __m256i s1 = _mm256_loadu_si256((const __m256i*)(st_ + 8)); // Compute and assign value
        const __m256i s0_hi = _mm256_srli_epi64(s0, 32); // Compute and assign value
        const __m256i s1_hi = _mm256_srli_epi64(s1, 32); // Compute and assign value
        for (int k = 0; k < 3; k++) { // Loop iteration
            const __m256i w0 = _mm256_loadu_si256((const __m256i*)(wsel_[k] + 0)); // Compute and assign value
            const __m256i w1 = _mm256_loadu_si256((const __m256i*)(wsel_[k] + 8)); // Compute and assign value
            __m256i acc = _mm256_add_epi64( // Compute and assign value
                _mm256_add_epi64(_mm256_mul_epi32(w0, s0), _mm256_mul_epi32(_mm256_srli_epi64(w0, 32), s0_hi)), // Execute statement
                _mm256_add_epi64(_mm256_mul_epi32(w1, s1), _mm256_mul_epi32(_mm256_srli_epi64(w1, 32), s1_hi)) // Execute statement
            ); // Execute statement
            __m128i lo = _mm256_castsi256_si128(acc); // Compute and assign value
            __m128i hi = _mm256_extracti128_si256(acc, 1); // Compute and assign value
            __m128i sum2 = _mm_add_epi64(lo, hi); // Compute and assign value
            i64 s = (i64)_mm_cvtsi128_si64(sum2) + (i64)_mm_cvtsi128_si64(_mm_srli_si128(sum2, 8)); // Compute and assign value
            int d = clampi((int)(s >> 16), -2047, 2047); // Compute and assign value
            p1_[k] = squash(d); st2_[k] = d; // Compute and assign value
        } // Close scope / block
#else // Fallback conditional compilation branch
        for (int k = 0; k < 3; k++) { // Loop iteration
            i64 s = 0; // Compute and assign value
            for (int i = 0; i < N_IN; i++) s += (i64)wsel_[k][i] * st_[i]; // Loop iteration
            int d = clampi((int)(s >> 16), -2047, 2047); // Compute and assign value
            p1_[k] = squash(d); st2_[k] = d; // Compute and assign value
        } // Close scope / block
#endif // End conditional compilation guard
        st2_[3] = BIAS; // Compute and assign value
        i64 s = 0; // Compute and assign value
        for (int i = 0; i < 4; i++) s += (i64)mixf_[pos_][i] * st2_[i]; // Loop iteration
        int d = clampi((int)(s >> 16), -2047, 2047); // Compute and assign value
        pf_ = clampi(squash(d), 1, 4095); // Compute and assign value
        int a1 = a1_.pp(pf_, (int)c0_), a2 = a2_.pp(pf_, (int)((c1_ << 8) | c0_)); // Compute and assign value
        return clampi((2 * pf_ + a1 + a2 + 2) >> 2, 1, 4095); // Return result to caller
    } // Close scope / block

    void update(int y) { // Execute statement
#if P3_LAYER1_PREFETCH // Conditional compilation check
        u32 next_mh = 0; // Compute and assign value
        if (pos_ == 3) { // Conditional branch check
            u32 next_c0 = (c0_ << 1) | (u32)y; // Compute and assign value
            prefetchBases(next_c0); // Execute statement
            p3_prefetch(&a2_.t[(size_t)((c1_ << 8) | next_c0) * 24]); // Execute statement
        } else if (pos_ == 7) { // Compute and assign value
            u32 c = ((c0_ << 1) | (u32)y) & 255u; // Compute and assign value
            c4_ = c3_; c3_ = c2_; c2_ = c1_; c1_ = c; // Compute and assign value
            if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122) || (c >= 48 && c <= 57)) // Conditional branch check
                word_ = (word_ * 263u + c + 1u) & 0xFFFFFFu; // Compute and assign value
            else // Fallback branch
                word_ = 0; // Compute and assign value
            buf_[(size_t)(blen_ & mask_)] = (u8)c; // Compute and assign value
            blen_++; // Execute statement
            setByteCtx(); // Execute statement
            prefetchBases(0u); // Execute statement
            if (blen_ >= (u64)MINLEN) { // Conditional branch check
                next_mh = hmix(hmix(hmix((u32)back(1), (u32)back(2)), (u32)back(3)), (u32)back(4)) >> (32 - mtBits_); // Compute and assign value
                p3_prefetch(&mtab_[next_mh]); // Execute statement
            } // Close scope / block
            p3_prefetch(&a2_.t[(size_t)((c << 8) | 1u) * 24]); // Execute statement
            p3_prefetch(&mix3_[(size_t)c * N_IN]); // Execute statement
        } // Close scope / block
#endif // End conditional compilation guard
#if P3_AVX2_MIXER // Conditional compilation check
        const __m256i us0 = _mm256_loadu_si256((const __m256i*)(st_ + 0)); // Compute and assign value
        const __m256i us1 = _mm256_loadu_si256((const __m256i*)(st_ + 8)); // Compute and assign value
        const __m256i urnd = _mm256_set1_epi32(0x8000); // Compute and assign value
        for (int k = 0; k < 3; k++) { // Loop iteration
            int e0 = (y << 12) - p1_[k]; // Compute and assign value
            if (e0 > 6 || e0 < -6) { // Conditional branch check
                const __m256i uerr = _mm256_set1_epi32(e0 * LR1); // Compute and assign value
                __m256i d0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(us0, uerr), urnd), 16); // Compute and assign value
                __m256i d1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(us1, uerr), urnd), 16); // Compute and assign value
                __m256i w0 = _mm256_loadu_si256((const __m256i*)(wsel_[k] + 0)); // Compute and assign value
                __m256i w1 = _mm256_loadu_si256((const __m256i*)(wsel_[k] + 8)); // Compute and assign value
                _mm256_storeu_si256((__m256i*)(wsel_[k] + 0), _mm256_add_epi32(w0, d0)); // Execute statement
                _mm256_storeu_si256((__m256i*)(wsel_[k] + 8), _mm256_add_epi32(w1, d1)); // Execute statement
            } // Close scope / block
        } // Close scope / block
#else // Fallback conditional compilation branch
        for (int k = 0; k < 3; k++) { // Loop iteration
            int e0 = (y << 12) - p1_[k]; // Compute and assign value
            if (e0 > 6 || e0 < -6) { // Conditional branch check
                int err = e0 * LR1; // Compute and assign value
                for (int i = 0; i < N_IN; i++) wsel_[k][i] += (st_[i] * err + 0x8000) >> 16; // Loop iteration
            } // Close scope / block
        } // Close scope / block
#endif // End conditional compilation guard
        int err = ((y << 12) - pf_) * LR2; // Compute and assign value
        for (int i = 0; i < 4; i++) mixf_[pos_][i] += (st2_[i] * err + 0x8000) >> 16; // Loop iteration
        a1_.up(y); a2_.up(y); // Execute statement
        for (int i = 0; i < NCTX; i++) { u32 t = *slot_[i] ^ kInit; sm_up(&t, y, LIMIT); *slot_[i] = t ^ kInit; } // Loop iteration
        if (mexp_ >= 0) sm_up(&mm_[mctx_], y == mexp_, 60); // Conditional branch check
        hist_ = (hist_ << 1) | (u32)y; // Compute and assign value
        c0_ = (c0_ << 1) | (u32)y; // Compute and assign value
        if (++pos_ == 8) { // Conditional branch check
#if P3_LAYER1_PREFETCH // Conditional compilation check
            byteDoneMatch(c0_ & 255u, next_mh); // Execute statement
#else // Fallback conditional compilation branch
            byteDone(c0_ & 255u); // Execute statement
#endif // End conditional compilation guard
        } // Close scope / block
    } // Close scope / block

private: // Member access specifier
    static u32 bitlen(u32 v) { u32 n = 0; while (v) { n++; v >>= 1; } return n; } // Define helper function or constant
    int resAt(u64 idx) const {                            // signed residual stored at absolute index idx (aw_ bytes, little endian)
        u32 v = 0; for (int i = aw_ - 1; i >= 0; i--) v = (v << 8) | buf_.p[(size_t)((idx + (u64)i) & mask_)]; // Compute and assign value
        return (int)v;                                   // zigzag value: already a magnitude
    } // Close scope / block
    int back(u32 k) const { return (k && blen_ >= k) ? buf_.p[(size_t)((blen_ - k) & mask_)] : 0; } // Compute and assign value

    void setByteCtx() { // Execute statement
        u32 h; // Execute statement
        bh_[0] = hmix(0xA0, 0); // Compute and assign value
        h = hmix(0xA1, c1_);     bh_[1] = h; // Compute and assign value
        h = hmix(h ^ 0xA2, c2_); bh_[2] = h; // Compute and assign value
        h = hmix(h ^ 0xA3, c3_); bh_[3] = h; // Compute and assign value
        h = hmix(h ^ 0xA4, c4_); bh_[4] = h; // Compute and assign value
        bh_[5] = hmix(0xA5, word_); // Compute and assign value
        if (blen_ >= ats_ && blen_ < ate_ && ach_) {      // audio residuals: magnitude contexts of neighbouring samples
            u64 f = (u64)(ach_ * aw_), rel = (blen_ - ats_) % f, base = blen_ - (rel % (u64)aw_); // Compute and assign value
            int bi = (int)(rel % (u64)aw_); // Compute and assign value
            int p1 = base >= ats_ + f ? resAt(base - f) : 0, p2 = base >= ats_ + 2 * f ? resAt(base - 2 * f) : 0; // Compute and assign value
            int ot = base >= ats_ + (u64)aw_ ? resAt(base - (u64)aw_) : 0; // Compute and assign value
            u32 q1 = bitlen(p1 < 0 ? -p1 : p1), q2 = bitlen(p2 < 0 ? -p2 : p2), qo = bitlen(ot < 0 ? -ot : ot); // Compute and assign value
            u32 known = bi ? (u32)back(1) | (bi > 1 ? (u32)back(2) << 8 : 0u) : 0u;       // bytes of this sample already coded
            u32 avg = bitlen((u32)(((p1 < 0 ? -p1 : p1) + (p2 < 0 ? -p2 : p2) + (ot < 0 ? -ot : ot)) / 3)); // Compute and assign value
            bh_[6] = hmix(hmix(0xD0, (u32)rel), q1 | (q2 << 6) | (known >> 4 << 12)); // Compute and assign value
            bh_[7] = hmix(hmix(hmix(0xD1, (u32)rel), qo), known); // Compute and assign value
            bh_[8] = hmix(hmix(0xD2, (u32)rel), avg | (q1 << 6) | (qo << 12)); // Compute and assign value
        } else if (stride_) {                             // BMP pixels: left / above / above-left neighbours
            int l = back((u32)bpp_), a = back((u32)stride_), al = back((u32)(stride_ + bpp_)); // Compute and assign value
            if (pix_) {                                   // PNG pixels: gradient prediction, previous-channel error, channel index
                int ch = (int)((blen_ + (u64)phase_) % (u64)bpp_); // Compute and assign value
                int ar = (u32)stride_ > (u32)bpp_ ? back((u32)(stride_ - bpp_)) : a; // Compute and assign value
                int pg = clampi(l + a - al, 0, 255), e1 = 0; // Compute and assign value
                if (ch > 0) { int l1 = back((u32)(bpp_ + 1)), a1 = back((u32)(stride_ + 1)), al1 = back((u32)(stride_ + bpp_ + 1)); e1 = (int)c1_ - clampi(l1 + a1 - al1, 0, 255); } // Conditional branch check
                int pr = clampi(pg + e1, 0, 255); // Compute and assign value
                auto q = [](int d) { d = d < -24 ? -24 : d > 24 ? 24 : d; return (u32)(d + 24); }; // Compute and assign value
                bh_[4] = hmix(hmix(0xB4, (u32)pg), (u32)ch); // Compute and assign value
                bh_[5] = ch ? hmix(hmix(0xB5, (u32)pr), (u32)ch) : hmix(hmix(0xB7, (u32)((a + ar + 1) >> 1)), (u32)l); // Compute and assign value
                bh_[6] = hmix(hmix(0xB0, (u32)a), (u32)l | ((u32)ch << 8)); // Compute and assign value
                bh_[7] = hmix(hmix(0xB6, q(l - al) | (q(a - al) << 6)), (u32)ch | (q(ar - a) << 3)); // Compute and assign value
                bh_[8] = hmix(hmix(0xB2, (u32)((l + a + 1) >> 1)), q(e1) | ((u32)ch << 8)); // Compute and assign value
            } else { // Execute statement
            bh_[6] = hmix(hmix(0xB0, (u32)a), (u32)l); // Compute and assign value
            bh_[7] = hmix(0xB1, (u32)clampi(l + a - al, 0, 255)); // Compute and assign value
            bh_[8] = hmix(hmix(0xB2, (u32)a), c1_); // Compute and assign value
            } // Close scope / block
        } else {                                          // sparse "skip" contexts
            bh_[6] = hmix(0xC0, c2_); // Compute and assign value
            bh_[7] = hmix(hmix(0xC1, c2_), c3_); // Compute and assign value
            bh_[8] = hmix(hmix(0xC2, c1_), c3_); // Compute and assign value
        } // Close scope / block
    } // Close scope / block

    void byteDone(u32 c) { // Execute statement
        pos_ = 0; c0_ = 1; // Compute and assign value
        c4_ = c3_; c3_ = c2_; c2_ = c1_; c1_ = c; // Compute and assign value
        if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122) || (c >= 48 && c <= 57)) word_ = (word_ * 263u + c + 1u) & 0xFFFFFFu; else word_ = 0; // Conditional branch check
        buf_[(size_t)(blen_ & mask_)] = (u8)c; blen_++; // Compute and assign value
        if (mptr_ && buf_[(size_t)(mptr_ & mask_)] == c) { mlen_++; mptr_++; } else { mlen_ = 0; mptr_ = 0; } // Conditional branch check
        if (blen_ >= (u64)MINLEN) { // Conditional branch check
            u32 h = hmix(hmix(hmix((u32)back(1), (u32)back(2)), (u32)back(3)), (u32)back(4)) >> (32 - mtBits_); // Compute and assign value
            if (mptr_ == 0) { // Conditional branch check
                u32 cand = mtab_[h]; // Compute and assign value
                if (cand) { // Conditional branch check
                    u32 dist = (u32)blen_ - cand; // Compute and assign value
                    if (dist > 0 && (u64)dist + 40 < ring_) { // Conditional branch check
                        u64 candAbs = blen_ - dist; u32 l = 0; // Compute and assign value
                        while (l < 32 && candAbs > l && buf_[(size_t)((candAbs - 1 - l) & mask_)] == buf_[(size_t)((blen_ - 1 - l) & mask_)]) l++; // Loop while condition holds
                        if (l >= (u32)MINLEN) { mptr_ = candAbs; mlen_ = l; } // Conditional branch check
                    } // Close scope / block
                } // Close scope / block
            } // Close scope / block
            mtab_[h] = (u32)blen_; // Compute and assign value
        } // Close scope / block
        setByteCtx(); // Execute statement
    } // Close scope / block

#if P3_LAYER1_PREFETCH // Conditional compilation check
    void prefetchBases(u32 nibCtx) { // Execute statement
        for (int i = 0; i < NBYTE; i++) { // Loop iteration
            u32 b = (hmix(bh_[i], nibCtx) >> (32 - byteTb_)) & ~15u; // Compute and assign value
            base_[i] = b; // Compute and assign value
            p3_prefetch(&tab_[NBIT + i][b]); // Execute statement
        } // Close scope / block
    } // Close scope / block

    void byteDoneMatch(u32 c, u32 h) { // Execute statement
        pos_ = 0; c0_ = 1; // Compute and assign value
        if (mptr_ && buf_[(size_t)(mptr_ & mask_)] == c) { mlen_++; mptr_++; } else { mlen_ = 0; mptr_ = 0; } // Conditional branch check
        if (blen_ >= (u64)MINLEN) { // Conditional branch check
            if (mptr_ == 0) { // Conditional branch check
                u32 cand = mtab_[h]; // Compute and assign value
                if (cand) { // Conditional branch check
                    u32 dist = (u32)blen_ - cand; // Compute and assign value
                    if (dist > 0 && (u64)dist + 40 < ring_) { // Conditional branch check
                        u64 candAbs = blen_ - dist; u32 l = 0; // Compute and assign value
                        while (l < 32 && candAbs > l && buf_[(size_t)((candAbs - 1 - l) & mask_)] == buf_[(size_t)((blen_ - 1 - l) & mask_)]) l++; // Loop while condition holds
                        if (l >= (u32)MINLEN) { mptr_ = candAbs; mlen_ = l; } // Conditional branch check
                    } // Close scope / block
                } // Close scope / block
            } // Close scope / block
            mtab_[h] = (u32)blen_; // Compute and assign value
        } // Close scope / block
        if (mptr_) p3_prefetch(&buf_[(size_t)(mptr_ & mask_)]); // Conditional branch check
    } // Close scope / block
#endif // End conditional compilation guard

    int byteTb_ = 22, mtBits_ = 22; // Compute and assign value
    size_t ring_ = 4096; u64 mask_ = 4095; // Compute and assign value
    Arr<u32> tab_[NCTX]; u32* slot_[NCTX] = {}; u32 base_[NBYTE] = {}; // Compute and assign value
    u32 hist_ = 0, c0_ = 1, c1_ = 0, c2_ = 0, c3_ = 0, c4_ = 0, word_ = 0, bh_[NBYTE] = {}; // Compute and assign value
    int pos_ = 0, stride_ = 0, bpp_ = 0, phase_ = 0; bool pix_ = false; // Compute and assign value
    u64 ats_ = 0, ate_ = 0; int ach_ = 0, aw_ = 0; // Compute and assign value
    Arr<u8> buf_; u64 blen_ = 0; Arr<u32> mtab_; u64 mptr_ = 0; u32 mlen_ = 0; u32 mm_[32]; int mexp_ = -1, mctx_ = 0; // Compute and assign value
    std::vector<i32> mix1_, mix2_, mix3_; i32 mixf_[8][4]; i32* wsel_[3] = {}; // Allocate / manage dynamic buffer
    int st_[N_IN] = {}, st2_[4] = {}, p1_[3] = {}, pf_ = 2048; // Compute and assign value
    APM a1_, a2_; // Execute statement
}; // Close scope / block

// ------------------------------------------------------------------ audio predictor (integer only, bit-exact everywhere)
// Per channel: stage A = normalised LMS over the last `na` sample differences (+`mc` differences of the other channel;
// for the 2nd channel that includes the current sample of the 1st), stage B = a small LMS over stage A's past errors.
// Three presets trade adaptation speed against precision; the encoder picks the best one per segment.
// forward(): samples -> residuals (same byte count).   inverse(): residuals -> samples.
struct AudioPreset { int na, mc, sha, shb; }; // Declare data structure / class
inline const AudioPreset& audio_preset(int i) { // Define helper function or constant
    static const AudioPreset t[3] = {{32, 16, 8, 6}, {32, 8, 7, 5}, {16, 8, 6, 5}}; // Define helper function or constant
    return t[i < 0 ? 0 : (i > 2 ? 2 : i)]; // Return result to caller
} // Close scope / block
class AudioCodec { // Declare data structure / class
public: // Member access specifier
    static constexpr int NAX = 32, MCX = 16, NB = 16; // Define helper function or constant
    AudioCodec(int channels, int width, int preset) // Execute statement
        : C_(channels), W_(width), ms_(channels == 2 && preset == 2), P_(audio_preset(preset)), ch_((size_t)channels) {} // Compute and assign value
    void forward(u8* p, size_t frames) { // Execute statement
        if (ms_) { // Conditional branch check
            for (size_t f = 0; f < frames; f++) { // Loop iteration
                int L = getSample(p), R = getSample(p + W_); // Compute and assign value
                int S = wrap(L - R), M = wrap(R + (S >> 1)); // Compute and assign value
                int prM = predict(0); putRes(p, zig(wrap(M - prM))); update(0, M); // Compute and assign value
                int prS = predict(1); putRes(p + W_, zig(wrap(S - prS))); update(1, S); // Compute and assign value
                p += 2 * W_; // Compute and assign value
            } // Close scope / block
            return; // Return result to caller
        } // Close scope / block
        for (size_t f = 0; f < frames; f++) for (int c = 0; c < C_; c++) { // Loop iteration
            int x = getSample(p), pr = predict(c); // Compute and assign value
            putRes(p, zig(wrap(x - pr))); update(c, x); p += W_; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    void inverse(u8* p, size_t frames) { // Execute statement
        if (ms_) { // Conditional branch check
            for (size_t f = 0; f < frames; f++) { // Loop iteration
                int rM = unzig(getU(p)), prM = predict(0), M = wrap(rM + prM); update(0, M); // Compute and assign value
                int rS = unzig(getU(p + W_)), prS = predict(1), S = wrap(rS + prS); update(1, S); // Compute and assign value
                int R = wrap(M - (S >> 1)), L = wrap(S + R); // Compute and assign value
                putSample(p, L); putSample(p + W_, R); // Execute statement
                p += 2 * W_; // Compute and assign value
            } // Close scope / block
            return; // Return result to caller
        } // Close scope / block
        for (size_t f = 0; f < frames; f++) for (int c = 0; c < C_; c++) { // Loop iteration
            int r = unzig(getU(p)), pr = predict(c), x = wrap(r + pr); // Compute and assign value
            putSample(p, x); update(c, x); p += W_; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
private: // Member access specifier
    struct Ch { // Declare data structure / class
        alignas(32) i32 dh[NAX] = {}, eh[NB] = {}, wa[NAX + MCX] = {}, wb[NB] = {}, ia[NAX + MCX] = {}; // Compute and assign value
        i32 s1 = 0, pA = 0, pB = 0; i64 eA = 0, eB = 0; // Compute and assign value
    }; // Close scope / block
    int wrap(i64 v) const { int bits = 8 * W_; i64 m = ((i64)1 << bits) - 1; v &= m; if (v >= ((i64)1 << (bits - 1))) v -= ((i64)1 << bits); return (int)v; } // Compute and assign value
    int zig(int s) const {                                   // zigzag: small +/- residuals -> small unsigned values
        int bits = 8 * W_; u32 m = (bits >= 32) ? 0xffffffffu : ((1u << bits) - 1); // Compute and assign value
        return (int)((((u32)s << 1) ^ (u32)(s >> (bits - 1))) & m); // Return result to caller
    } // Close scope / block
    u32 getU(const u8* p) const { u32 v = 0; for (int i = W_ - 1; i >= 0; i--) v = (v << 8) | p[i]; return v; } // Compute and assign value
    int unzig(u32 u) const { return (int)(u >> 1) ^ -(int)(u & 1); } // Execute statement
    int getRes(const u8* p) const { u32 v = 0; for (int i = W_ - 1; i >= 0; i--) v = (v << 8) | p[i]; int bits = 8 * W_; return (int)((v ^ (1u << (bits - 1))) - (1u << (bits - 1))); } // Compute and assign value
    void putRes(u8* p, int r) const { u32 v = (u32)r; for (int i = 0; i < W_; i++) { p[i] = (u8)v; v >>= 8; } } // Compute and assign value
    int getSample(const u8* p) const { return W_ == 1 ? (int)p[0] - 128 : getRes(p); } // Compute and assign value
    void putSample(u8* p, int x) const { if (W_ == 1) p[0] = (u8)(x + 128); else putRes(p, x); } // Compute and assign value
#if P3_AVX2_MIXER // Conditional compilation check
    static inline i64 hsum256_epi64(__m256i v) { // Define helper function or constant
        __m128i lo = _mm256_castsi256_si128(v); // Compute and assign value
        __m128i hi = _mm256_extracti128_si256(v, 1); // Compute and assign value
        __m128i s = _mm_add_epi64(lo, hi); // Compute and assign value
        return (i64)_mm_cvtsi128_si64(s) + (i64)_mm_cvtsi128_si64(_mm_srli_si128(s, 8)); // Return result to caller
    } // Close scope / block
    static inline void dot_energy_avx2(const i32* w, const i32* x, int n, i64& outDot, i64& outEn) { // Define helper function or constant
        __m256i vdot = _mm256_setzero_si256(), ven = _mm256_setzero_si256(); // Compute and assign value
        int i = 0; // Compute and assign value
        for (; i + 8 <= n; i += 8) { // Loop iteration
            __m256i vw = _mm256_loadu_si256((const __m256i*)(w + i)); // Compute and assign value
            __m256i vx = _mm256_loadu_si256((const __m256i*)(x + i)); // Compute and assign value
            __m256i vwh = _mm256_srli_epi64(vw, 32); // Compute and assign value
            __m256i vxh = _mm256_srli_epi64(vx, 32); // Compute and assign value
            vdot = _mm256_add_epi64(vdot, _mm256_add_epi64(_mm256_mul_epi32(vw, vx), _mm256_mul_epi32(vwh, vxh))); // Compute and assign value
            ven  = _mm256_add_epi64(ven,  _mm256_add_epi64(_mm256_mul_epi32(vx, vx), _mm256_mul_epi32(vxh, vxh))); // Compute and assign value
        } // Close scope / block
        i64 d = hsum256_epi64(vdot), e = hsum256_epi64(ven); // Compute and assign value
        for (; i < n; i++) { d += (i64)w[i] * x[i]; e += (i64)x[i] * x[i]; } // Loop iteration
        outDot = d; outEn = e; // Compute and assign value
    } // Close scope / block
#endif // End conditional compilation guard
    int predict(int c) { // Execute statement
        Ch& s = ch_[(size_t)c]; const int na = P_.na, m = C_ == 2 ? P_.mc : 0; // Compute and assign value
        const Ch* o = C_ == 2 ? &ch_[(size_t)(1 - c)] : nullptr; // Compute and assign value
        std::memcpy(s.ia, s.dh, (size_t)na * sizeof(i32)); // Fast raw memory copy
        if (m) std::memcpy(s.ia + na, o->dh, (size_t)m * sizeof(i32)); // Conditional branch check
        i64 sa = 0, ea = 0, sb = 0, eb = 0; // Compute and assign value
#if P3_AVX2_MIXER // Conditional compilation check
        dot_energy_avx2(s.wa, s.ia, na + m, sa, ea); // Execute statement
        dot_energy_avx2(s.wb, s.eh, NB, sb, eb); // Execute statement
#else // Fallback conditional compilation branch
        for (int i = 0; i < na + m; i++) { sa += (i64)s.wa[i] * s.ia[i]; ea += (i64)s.ia[i] * s.ia[i]; } // Loop iteration
        for (int i = 0; i < NB; i++) { sb += (i64)s.wb[i] * s.eh[i]; eb += (i64)s.eh[i] * s.eh[i]; } // Loop iteration
#endif // End conditional compilation guard
        s.eA = ea; s.pA = (int)((sa + 32768) >> 16); // Compute and assign value
        s.eB = eb; s.pB = (int)((sb + 32768) >> 16); // Compute and assign value
        i64 pr = (i64)s.s1 + s.pA + s.pB, lo = -((i64)1 << (8 * W_ - 1)), hi = ((i64)1 << (8 * W_ - 1)) - 1; // Compute and assign value
        return (int)(pr < lo ? lo : (pr > hi ? hi : pr)); // Return result to caller
    } // Close scope / block
    void update(int c, int x) { // Execute statement
        Ch& s = ch_[(size_t)c]; const int na = P_.na, m = C_ == 2 ? P_.mc : 0; // Compute and assign value
        int d = x - s.s1, errA = d - s.pA, errB = errA - s.pB; // Compute and assign value
        const i64 kEps = 2048, kMaxW = (i64)1 << 24; // Compute and assign value
        i64 g = ((i64)errA * ((i64)1 << (32 - P_.sha))) / (s.eA + kEps); // Compute and assign value
        for (int i = 0; i < na + m; i++) { i64 w = s.wa[i] + ((g * s.ia[i]) >> 16); s.wa[i] = (i32)(w > kMaxW ? kMaxW : (w < -kMaxW ? -kMaxW : w)); } // Loop iteration
        i64 g2 = ((i64)errB * ((i64)1 << (32 - P_.shb))) / (s.eB + kEps); // Compute and assign value
        for (int i = 0; i < NB; i++) { i64 w = s.wb[i] + ((g2 * s.eh[i]) >> 16); s.wb[i] = (i32)(w > kMaxW ? kMaxW : (w < -kMaxW ? -kMaxW : w)); } // Loop iteration
        std::memmove(s.dh + 1, s.dh, (size_t)(na - 1) * sizeof(i32)); s.dh[0] = d; // Compute and assign value
        std::memmove(s.eh + 1, s.eh, (NB - 1) * sizeof(i32)); s.eh[0] = errA; // Compute and assign value
        s.s1 = x; // Compute and assign value
    } // Close scope / block
    int C_, W_; bool ms_; AudioPreset P_; std::vector<Ch> ch_; // Allocate / manage dynamic buffer
}; // Close scope / block

// ------------------------------------------------------------------ grid transform (tables, series, images)
// Words of `width` bytes are replaced by their difference to a prediction: mode 0 = the word one record (stride bytes) back,
// mode 1 = gradient left + above - above-left (images, stride = row bytes). zz = zigzag the signed difference.
struct Grid { // Declare data structure / class
    int stride = 0, width = 0, mode = 0, zz = 0; // Compute and assign value
    bool on() const { return stride > 0; } // Execute statement
}; // Close scope / block
inline bool grid_region(const Grid& g, u64 segStart, size_t L, size_t& ts, size_t& te) {   // whole records only
    ts = te = 0; if (!g.on()) return false; // Compute and assign value
    u64 S = (u64)g.stride, first = ((segStart + S - 1) / S) * S; // Compute and assign value
    if (first >= segStart + L) return false; // Conditional branch check
    u64 lo = first - segStart, recs = ((u64)L - lo) / S; // Compute and assign value
    if (recs == 0) return false; // Conditional branch check
    ts = (size_t)lo; te = (size_t)(lo + recs * S); return true; // Compute and assign value
} // Close scope / block
class GridCodec { // Declare data structure / class
public: // Member access specifier
    explicit GridCodec(const Grid& g) : g_(g), w_(g.width), cols_((size_t)(g.stride / g.width)), mask_(g.width == 4 ? 0xffffffffull : (((u64)1 << (8 * g.width)) - 1)) {} // Compute and assign value
    void forward(u8* p, size_t bytes) {            // descending: predictors (earlier words) are still the originals
        for (size_t k = bytes / (size_t)w_; k-- > 0;) { u64 x = get(p, k), pr = predict(p, k); put(p, k, enc((x - pr) & mask_)); } // Loop iteration
    } // Close scope / block
    void inverse(u8* p, size_t bytes) {            // ascending: predictors are already restored
        size_t n = bytes / (size_t)w_; // Compute and assign value
        for (size_t k = 0; k < n; k++) { u64 r = dec(get(p, k)), pr = predict(p, k); put(p, k, (r + pr) & mask_); } // Loop iteration
    } // Close scope / block
private: // Member access specifier
    u64 get(const u8* p, size_t k) const { u64 v = 0; const u8* q = p + k * (size_t)w_; for (int i = w_ - 1; i >= 0; i--) v = (v << 8) | q[i]; return v; } // Compute and assign value
    void put(u8* p, size_t k, u64 v) const { u8* q = p + k * (size_t)w_; for (int i = 0; i < w_; i++) { q[i] = (u8)v; v >>= 8; } } // Compute and assign value
    u64 predict(const u8* p, size_t k) const { // Execute statement
        if (g_.mode == 0) return k >= cols_ ? get(p, k - cols_) : 0; // Conditional branch check
        size_t col = k % cols_; i64 l = col ? (i64)get(p, k - 1) : 0, a = k >= cols_ ? (i64)get(p, k - cols_) : 0, al = (k >= cols_ && col) ? (i64)get(p, k - cols_ - 1) : 0; // Compute and assign value
        i64 v = l + a - al; return (u64)(v < 0 ? 0 : (v > (i64)mask_ ? (i64)mask_ : v)); // Compute and assign value
    } // Close scope / block
    u64 enc(u64 r) const {                          // r = wrapped difference
        if (!g_.zz) return r; // Conditional branch check
        i64 sgn = (r >> (8 * w_ - 1)) ? (i64)r - ((i64)1 << (8 * w_)) : (i64)r; // Compute and assign value
        return (u64)((sgn << 1) ^ (sgn >> 63)) & mask_; // Return result to caller
    } // Close scope / block
    u64 dec(u64 u) const { if (!g_.zz) return u; return (u64)(((i64)(u >> 1)) ^ -(i64)(u & 1)) & mask_; } // Execute statement
    Grid g_; int w_; size_t cols_; u64 mask_; // Execute statement
}; // Close scope / block

// ------------------------------------------------------------------ executable filters (x86 / ARM64 / ARM32)
// Relative branch targets become absolute addresses, so the same function called from many places gives the same bytes.
// Each rewrite is addition modulo 2^k on a field that is detected from bytes the filter leaves unchanged, so it is
// exactly reversible by construction. base = file offset of the segment (instruction addresses).
//   kind 1: x86 CALL/JMP (E8/E9 rel32 whose top byte is 00/FF: a 25-bit signed value), kind 2: ARM64 BL, kind 3: ARM32 BL.
inline void exec_filter(u8* p, size_t L, u64 base, int kind, bool forward) { // Define helper function or constant
    auto rd = [](const u8* q) { return (u32)q[0] | ((u32)q[1] << 8) | ((u32)q[2] << 16) | ((u32)q[3] << 24); }; // Compute and assign value
    auto wr = [](u8* q, u32 v) { q[0] = (u8)v; q[1] = (u8)(v >> 8); q[2] = (u8)(v >> 16); q[3] = (u8)(v >> 24); }; // Compute and assign value
    if (kind == 1) { // Conditional branch check
        for (size_t i = 0; i + 5 <= L;) { // Loop iteration
            if (p[i] != 0xE8 && p[i] != 0xE9) { i++; continue; } // Conditional branch check
            u32 v = rd(p + i + 1), top = v >> 24; // Compute and assign value
            if (top == 0 || top == 0xFF) { // Conditional branch check
                u32 low = v & 0x1FFFFFFu, off = (u32)((base + i + 5) & 0x1FFFFFFu); // Compute and assign value
                low = forward ? ((low + off) & 0x1FFFFFFu) : ((low - off) & 0x1FFFFFFu); // Compute and assign value
                wr(p + i + 1, (low & 0x1000000u) ? (low | 0xFE000000u) : low); // Execute statement
            } // Close scope / block
            i += 5; // Compute and assign value
        } // Close scope / block
    } else if (kind == 2) { // Compute and assign value
        for (size_t i = 0; i + 4 <= L; i += 4) { // Loop iteration
            u32 w = rd(p + i); // Compute and assign value
            if ((w >> 26) == 0x25) { // Conditional branch check
                u32 pc = (u32)((base + i) >> 2) & 0x3FFFFFFu, imm = w & 0x3FFFFFFu; // Compute and assign value
                imm = forward ? ((imm + pc) & 0x3FFFFFFu) : ((imm - pc) & 0x3FFFFFFu); // Compute and assign value
                wr(p + i, (w & 0xFC000000u) | imm); // Execute statement
            } // Close scope / block
        } // Close scope / block
    } else if (kind == 3) { // Compute and assign value
        for (size_t i = 0; i + 4 <= L; i += 4) { // Loop iteration
            if (p[i + 3] != 0xEB) continue; // Conditional branch check
            u32 w = rd(p + i), pc = (u32)((base + i + 8) >> 2) & 0xFFFFFFu, imm = w & 0xFFFFFFu; // Compute and assign value
            imm = forward ? ((imm + pc) & 0xFFFFFFu) : ((imm - pc) & 0xFFFFFFu); // Compute and assign value
            wr(p + i, (w & 0xFF000000u) | imm); // Execute statement
        } // Close scope / block
    } // Close scope / block
} // Close scope / block

// ------------------------------------------------------------------ arithmetic coder over memory buffers
class Enc { // Declare data structure / class
public: // Member access specifier
    explicit Enc(std::vector<u8>& o) : out_(o) {} // Allocate / manage dynamic buffer
    void bit(int y, int p) { // Execute statement
        u32 xm = x1_ + ((x2_ - x1_) >> 12) * (u32)p; // Compute and assign value
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        u32 m = -(u32)(y != 0); // Compute and assign value
        x2_ = (xm & m) | (x2_ & ~m); // Compute and assign value
        x1_ = (x1_ & m) | ((xm + 1u) & ~m); // Compute and assign value
#else // Fallback conditional compilation branch
        if (y) x2_ = xm; else x1_ = xm + 1; // Conditional branch check
#endif // End conditional compilation guard
        norm(); // Execute statement
    } // Close scope / block
    void half(int y) { // Execute statement
        u32 xm = x1_ + ((x2_ - x1_) >> 1); // Compute and assign value
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        u32 m = -(u32)(y != 0); // Compute and assign value
        x2_ = (xm & m) | (x2_ & ~m); // Compute and assign value
        x1_ = (x1_ & m) | ((xm + 1u) & ~m); // Compute and assign value
#else // Fallback conditional compilation branch
        if (y) x2_ = xm; else x1_ = xm + 1; // Conditional branch check
#endif // End conditional compilation guard
        norm(); // Execute statement
    } // Close scope / block
    void finish() {                                         // shortest byte string (zero padded by the decoder) inside [x1,x2]
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        flush(); // Execute statement
#endif // End conditional compilation guard
        if (x1_ == 0) return; // Conditional branch check
        for (int nb = 1; nb <= 4; nb++) { // Loop iteration
            int shift = 32 - 8 * nb; // Compute and assign value
            u64 v = (((u64)x1_ + ((u64)1 << shift) - 1) >> shift) << shift; // Compute and assign value
            if (v <= x2_) { for (int k = 0; k < nb; k++) out_.push_back((u8)(v >> (24 - 8 * k))); return; } // Conditional branch check
        } // Close scope / block
        for (int k = 0; k < 4; k++) out_.push_back((u8)(x1_ >> (24 - 8 * k))); // Loop iteration
    } // Close scope / block
private: // Member access specifier
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
    void flush() { if (bp_) { out_.insert(out_.end(), buf_, buf_ + bp_); bp_ = 0; } } // Compute and assign value
    void norm() { // Execute statement
        u32 diff = x1_ ^ x2_; // Compute and assign value
        u32 nbytes = (diff == 0) ? 4u : (p3_clz32(diff) >> 3); // Compute and assign value
        u32 be = p3_bswap32(x2_); // Compute and assign value
        std::memcpy(buf_ + bp_, &be, 4); // Fast raw memory copy
        bp_ += nbytes; // Compute and assign value
        u32 sh = nbytes << 3; // Compute and assign value
        x1_ = (u32)((u64)x1_ << sh); // Compute and assign value
        x2_ = (u32)(((u64)x2_ << sh) | ((1ULL << sh) - 1ULL)); // Compute and assign value
        if (bp_ >= 4096) flush(); // Conditional branch check
    } // Close scope / block
    u8 buf_[4096 + 8]; size_t bp_ = 0; // Compute and assign value
#else // Fallback conditional compilation branch
    void norm() { while (((x1_ ^ x2_) & 0xff000000u) == 0) { out_.push_back((u8)(x2_ >> 24)); x1_ <<= 8; x2_ = (x2_ << 8) | 255u; } } // Append element to container
#endif // End conditional compilation guard
    std::vector<u8>& out_; u32 x1_ = 0, x2_ = 0xffffffffu; // Allocate / manage dynamic buffer
}; // Close scope / block
class Dec { // Declare data structure / class
public: // Member access specifier
    Dec(const u8* p, size_t n) : p_(p), n_(n) { for (int i = 0; i < 4; i++) x_ = (x_ << 8) | next(); } // Compute and assign value
    int bit(int p) { // Execute statement
        u32 xm = x1_ + ((x2_ - x1_) >> 12) * (u32)p; // Compute and assign value
        int y = x_ <= xm; // Compute and assign value
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        u32 m = -(u32)y; // Compute and assign value
        x2_ = (xm & m) | (x2_ & ~m); // Compute and assign value
        x1_ = (x1_ & m) | ((xm + 1u) & ~m); // Compute and assign value
#else // Fallback conditional compilation branch
        if (y) x2_ = xm; else x1_ = xm + 1; // Conditional branch check
#endif // End conditional compilation guard
        norm(); // Execute statement
        return y; // Return result to caller
    } // Close scope / block
    int half() { // Execute statement
        u32 xm = x1_ + ((x2_ - x1_) >> 1); // Compute and assign value
        int y = x_ <= xm; // Compute and assign value
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        u32 m = -(u32)y; // Compute and assign value
        x2_ = (xm & m) | (x2_ & ~m); // Compute and assign value
        x1_ = (x1_ & m) | ((xm + 1u) & ~m); // Compute and assign value
#else // Fallback conditional compilation branch
        if (y) x2_ = xm; else x1_ = xm + 1; // Conditional branch check
#endif // End conditional compilation guard
        norm(); // Execute statement
        return y; // Return result to caller
    } // Close scope / block
private: // Member access specifier
    u8 next() { return pos_ < n_ ? p_[pos_++] : 0; } // Execute statement
    void norm() { // Execute statement
#if P3_LAYER2_BRANCHLESS_AC // Conditional compilation check
        if (pos_ + 4 <= n_) { // Conditional branch check
            u32 diff = x1_ ^ x2_; // Compute and assign value
            u32 nbytes = (diff == 0) ? 4u : (p3_clz32(diff) >> 3); // Compute and assign value
            u32 sh = nbytes << 3; // Compute and assign value
            u32 v; std::memcpy(&v, p_ + pos_, 4); // Fast raw memory copy
            v = p3_bswap32(v); // Compute and assign value
            x1_ = (u32)((u64)x1_ << sh); // Compute and assign value
            x2_ = (u32)(((u64)x2_ << sh) | ((1ULL << sh) - 1ULL)); // Compute and assign value
            x_  = (u32)(((u64)x_  << sh) | ((u64)v >> (32 - sh))); // Compute and assign value
            pos_ += nbytes; // Compute and assign value
            return; // Return result to caller
        } // Close scope / block
#endif // End conditional compilation guard
        while (((x1_ ^ x2_) & 0xff000000u) == 0) { x1_ <<= 8; x2_ = (x2_ << 8) | 255u; x_ = (x_ << 8) | next(); } // Loop while condition holds
    } // Close scope / block
    const u8* p_; size_t n_, pos_ = 0; u32 x1_ = 0, x2_ = 0xffffffffu, x_ = 0; // Compute and assign value
}; // Close scope / block

inline void put_varint_mem(std::vector<u8>& v, u64 n) { while (n >= 128) { v.push_back((u8)(n | 128)); n >>= 7; } v.push_back((u8)n); } // Allocate / manage dynamic buffer

struct Counters { std::atomic<u64> done{0}, out{0}; std::atomic<bool> cancel{false}; }; // Declare data structure / class

// order-0 entropy of a block in bits per byte (encoder-side heuristic only, never affects decoding)
inline double block_entropy(const u8* p, size_t n) { // Define helper function or constant
    u32 c[256]; std::memset(c, 0, sizeof c); // Initialize memory buffer
    for (size_t i = 0; i < n; i++) c[p[i]]++; // Loop iteration
    double s = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) if (c[i]) s += (double)c[i] * std::log2((double)c[i]); // Loop iteration
    return std::log2((double)n) - s / (double)n; // Return result to caller
} // Close scope / block

// ------------------------------------------------------------------ segment codec
// Block types: 0 = coded by the model, 1 = stored raw inside the arithmetic stream (model still learns),
// 2 = SKIPPED: the model is not run at all and the bytes go to a plain side area (memcpy speed).
// After 4 raw blocks in a row the encoder skips 15 blocks, then probes one block with the model again;
// repeated failed probes lengthen the skip up to 255 blocks (4 MB); a block whose byte entropy drops below 7.7 bits ends it at once.
// payload = [varint arithLen] [arithmetic stream] [bytes of the skipped blocks, in order]

// ------------------------------------------------------------------ fast byte-level LZ77 + rANS segment codec (tbMax == 1)
namespace fast { // Open namespace scope
static constexpr u32 kProbBits = 12, kProbScale = 1u << kProbBits; // Define helper function or constant
static constexpr u32 kRansL = 1u << 23; // Define helper function or constant

struct FastScratch { // Declare data structure / class
    std::vector<u32> head4, head3; // Allocate / manage dynamic buffer
    std::vector<u8> lits, toks, offLo, offMid, offHi, extra, rev, o0; // Allocate / manage dynamic buffer
    void reset(size_t L, size_t hash4Size, size_t ways, size_t hash3Size) { // Execute statement
        if (head4.size() != hash4Size * ways) head4.assign(hash4Size * ways, 0); // Conditional branch check
        else std::memset(head4.data(), 0, head4.size() * sizeof(u32)); // Fallback branch
        if (head3.size() != hash3Size) head3.assign(hash3Size, 0); // Conditional branch check
        else std::memset(head3.data(), 0, head3.size() * sizeof(u32)); // Fallback branch
        lits.clear();   if (lits.capacity()   < L / 2 + 64) lits.reserve(L / 2 + 64); // Execute statement
        toks.clear();   if (toks.capacity()   < L / 6 + 64) toks.reserve(L / 6 + 64); // Execute statement
        offLo.clear();  if (offLo.capacity()  < L / 8 + 64) offLo.reserve(L / 8 + 64); // Execute statement
        offMid.clear(); if (offMid.capacity() < L / 8 + 64) offMid.reserve(L / 8 + 64); // Execute statement
        offHi.clear();  if (offHi.capacity()  < L / 8 + 64) offHi.reserve(L / 8 + 64); // Execute statement
        extra.clear(); // Execute statement
    } // Close scope / block
}; // Close scope / block

inline void put_u32_le(std::vector<u8>& o, u32 v) { // Allocate / manage dynamic buffer
    o.push_back((u8)v); o.push_back((u8)(v >> 8)); o.push_back((u8)(v >> 16)); o.push_back((u8)(v >> 24)); // Append element to container
} // Close scope / block
inline u32 get_u32_le(const u8* p) { // Define helper function or constant
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); // Return result to caller
} // Close scope / block

inline void normalize_freqs(const u32 cnt[256], size_t n, u16 freq[256], u16 cum[256]) { // Define helper function or constant
    u32 total = 0; int nsym = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) { // Loop iteration
        freq[i] = 0; // Compute and assign value
        if (cnt[i]) { // Conditional branch check
            nsym++; // Execute statement
            u32 f = (u32)(((u64)cnt[i] * kProbScale + n / 2) / n); // Compute and assign value
            if (f == 0) f = 1; // Conditional branch check
            freq[i] = (u16)f; total += f; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    if (total == 0) { freq[0] = (u16)kProbScale; total = kProbScale; } // Conditional branch check
    // Proportional rescale of symbols with freq > 1 when tail clamping pushes total > kProbScale
    if (total > kProbScale) { // Conditional branch check
        u32 fixedOnes = 0, flexSum = 0; // Compute and assign value
        for (int i = 0; i < 256; i++) { // Loop iteration
            if (freq[i] == 1) fixedOnes++; // Conditional branch check
            else if (freq[i] > 1) flexSum += freq[i]; // Alternative conditional branch
        } // Close scope / block
        if (flexSum > 0 && kProbScale > fixedOnes) { // Conditional branch check
            u32 targetFlex = kProbScale - fixedOnes; // Compute and assign value
            total = fixedOnes; // Compute and assign value
            for (int i = 0; i < 256; i++) if (freq[i] > 1) { // Loop iteration
                u32 nf = (u32)(((u64)freq[i] * targetFlex + flexSum / 2) / flexSum); // Compute and assign value
                if (nf < 1) nf = 1; // Conditional branch check
                freq[i] = (u16)nf; total += nf; // Compute and assign value
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    // Exact 1-unit-at-a-time greedy KL-divergence adjustment (preserves optimal probabilities)
    while (total > kProbScale) { // Loop while condition holds
        int best = -1; double bestCost = 1e300; // Compute and assign value
        for (int i = 0; i < 256; i++) if (freq[i] > 1) { // Loop iteration
            double cost = (double)cnt[i] * std::log2((double)freq[i] / (double)(freq[i] - 1)); // Compute and assign value
            if (cost < bestCost) { bestCost = cost; best = i; } // Conditional branch check
        } // Close scope / block
        if (best < 0) break; // Conditional branch check
        freq[best]--; total--; // Execute statement
    } // Close scope / block
    while (total < kProbScale) { // Loop while condition holds
        int best = -1; double bestGain = -1.0; // Compute and assign value
        for (int i = 0; i < 256; i++) if (cnt[i]) { // Loop iteration
            double gain = (double)cnt[i] * std::log2((double)(freq[i] + 1) / (double)freq[i]); // Compute and assign value
            if (gain > bestGain) { bestGain = gain; best = i; } // Conditional branch check
        } // Close scope / block
        if (best < 0) break; // Conditional branch check
        freq[best]++; total++; // Execute statement
    } // Close scope / block
    u32 c = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) { cum[i] = (u16)c; c += freq[i]; } // Loop iteration
} // Close scope / block

inline void write_freq_table(std::vector<u8>& hdr, const u16 freq[256]) { // Allocate / manage dynamic buffer
    int nsym = 0; for (int i = 0; i < 256; i++) if (freq[i]) nsym++; // Compute and assign value
    hdr.push_back((u8)(nsym - 1)); // Append element to container
    for (int i = 0; i < 256; i++) { // Loop iteration
        if (freq[i]) { // Conditional branch check
            hdr.push_back((u8)i); // Append element to container
            u32 f = freq[i] - 1; // Compute and assign value
            if (f < 128) hdr.push_back((u8)f); // Conditional branch check
            else { hdr.push_back((u8)((f & 127) | 128)); hdr.push_back((u8)(f >> 7)); } // Fallback branch
        } // Close scope / block
    } // Close scope / block
} // Close scope / block

inline void read_freq_table(const u8*& p, const u8* end, u16 freq[256], u16 cum[256], u8 lut[kProbScale]) { // Define helper function or constant
    if (p >= end) throw Failure{Status::Corrupt, "fast freq hdr"}; // Conditional branch check
    int nsym = (int)(*p++) + 1; // Compute and assign value
    std::memset(freq, 0, 256 * sizeof(u16)); // Initialize memory buffer
    std::memset(cum, 0, 256 * sizeof(u16)); // Initialize memory buffer
    for (int k = 0; k < nsym; k++) { // Loop iteration
        if (p + 2 > end) throw Failure{Status::Corrupt, "fast freq table"}; // Conditional branch check
        u8 s = *p++; u32 lo = *p++; // Compute and assign value
        u32 f = (lo & 128) ? ((lo & 127) | ((u32)(*p++) << 7)) : lo; // Compute and assign value
        freq[s] = (u16)(f + 1); // Compute and assign value
    } // Close scope / block
    u32 c = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) { // Loop iteration
        if (freq[i]) { // Conditional branch check
            cum[i] = (u16)c; // Compute and assign value
            if (c + freq[i] > kProbScale) throw Failure{Status::Corrupt, "fast freq overflow"}; // Conditional branch check
            std::memset(lut + c, i, freq[i]); // Initialize memory buffer
            c += freq[i]; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    if (c != kProbScale) throw Failure{Status::Corrupt, "fast freq sum"}; // Conditional branch check
} // Close scope / block

// Order-0 32-bit rANS stream encoder (falls back to raw if not smaller)
inline void normalize_freqs_14(const u32 cnt[256], size_t n, u16 freq[256], u16 cum[256]) { // Define helper function or constant
    static constexpr u32 kScale14 = 16384u; // Define helper function or constant
    u32 total = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) { // Loop iteration
        freq[i] = 0; // Compute and assign value
        if (cnt[i]) { // Conditional branch check
            u32 f = (u32)(((u64)cnt[i] * kScale14 + n / 2) / n); // Compute and assign value
            if (f == 0) f = 1; // Conditional branch check
            freq[i] = (u16)f; total += f; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    if (total == 0) { freq[0] = (u16)kScale14; total = kScale14; } // Conditional branch check
    if (total > kScale14) { // Conditional branch check
        u32 fixedOnes = 0, flexSum = 0; // Compute and assign value
        for (int i = 0; i < 256; i++) { // Loop iteration
            if (freq[i] == 1) fixedOnes++; // Conditional branch check
            else if (freq[i] > 1) flexSum += freq[i]; // Alternative conditional branch
        } // Close scope / block
        if (flexSum > 0 && kScale14 > fixedOnes) { // Conditional branch check
            u32 targetFlex = kScale14 - fixedOnes; // Compute and assign value
            total = fixedOnes; // Compute and assign value
            for (int i = 0; i < 256; i++) if (freq[i] > 1) { // Loop iteration
                u32 nf = (u32)(((u64)freq[i] * targetFlex + flexSum / 2) / flexSum); // Compute and assign value
                if (nf < 1) nf = 1; // Conditional branch check
                freq[i] = (u16)nf; total += nf; // Compute and assign value
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    while (total > kScale14) { // Loop while condition holds
        int best = -1; double bestCost = 1e300; // Compute and assign value
        for (int i = 0; i < 256; i++) if (freq[i] > 1) { // Loop iteration
            double cost = (double)cnt[i] * std::log2((double)freq[i] / (double)(freq[i] - 1)); // Compute and assign value
            if (cost < bestCost) { bestCost = cost; best = i; } // Conditional branch check
        } // Close scope / block
        if (best < 0) break; // Conditional branch check
        freq[best]--; total--; // Execute statement
    } // Close scope / block
    while (total < kScale14) { // Loop while condition holds
        int best = -1; double bestGain = -1.0; // Compute and assign value
        for (int i = 0; i < 256; i++) if (cnt[i]) { // Loop iteration
            double gain = (double)cnt[i] * std::log2((double)(freq[i] + 1) / (double)freq[i]); // Compute and assign value
            if (gain > bestGain) { bestGain = gain; best = i; } // Conditional branch check
        } // Close scope / block
        if (best < 0) break; // Conditional branch check
        freq[best]++; total++; // Execute statement
    } // Close scope / block
    u32 c = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) { cum[i] = (u16)c; c += freq[i]; } // Loop iteration
} // Close scope / block

inline void rans_encode_stream(const u8* in, size_t n, std::vector<u8>& out, std::vector<u8>* revBuf = nullptr) { // Allocate / manage dynamic buffer
    put_varint_mem(out, n); // Encode / decode variable-length integer
    if (n == 0) return; // Conditional branch check
    u32 cnt[256] = {}; // Compute and assign value
    for (size_t i = 0; i < n; i++) cnt[in[i]]++; // Loop iteration

    // High-precision 14-bit rANS (mode 5, 16-bit renorm, division-free) when heavy rare tail would be clamped under 12-bit scale
    if (n >= 32768) { // Conditional branch check
        int rareCount = 0; // Compute and assign value
        for (int i = 0; i < 256; i++) if (cnt[i] && (u64)cnt[i] * 4096ull < n) rareCount++; // Loop iteration
        if (rareCount >= 64) { // Conditional branch check
            u16 f14[256], c14[256]; // Execute statement
            normalize_freqs_14(cnt, n, f14, c14); // Execute statement
            static constexpr u32 kL14 = 1u << 15; // Define helper function or constant
            u64 rcp14[256]; // Execute statement
            for (int i = 0; i < 256; i++) rcp14[i] = f14[i] ? (((1ULL << 47) + f14[i] - 1) / f14[i]) : 0; // Loop iteration
            std::vector<u16> rev16; rev16.reserve(n / 2 + 64); // Allocate / manage dynamic buffer
            u32 x = kL14; // Compute and assign value
            for (size_t i = n; i-- > 0;) { // Loop iteration
                u8 s = in[i]; u32 f = f14[s]; // Compute and assign value
                u32 x_max = (2u << 16) * f; // Compute and assign value
                if (x >= x_max) { rev16.push_back((u16)(x & 0xffffu)); x >>= 16; } // Conditional branch check
                u32 q = (u32)(((u64)x * rcp14[s]) >> 47); // Compute and assign value
                x = (q << 14) + (x - q * f) + c14[s]; // Compute and assign value
            } // Close scope / block
            int nsym = 0; for (int i = 0; i < 256; i++) if (f14[i]) nsym++; // Compute and assign value
            size_t compSize = 1 + 1 + (size_t)nsym * 3 + 4 + 4 + rev16.size() * 2; // Compute and assign value
            if (compSize < 1 + n) { // Conditional branch check
                out.push_back(5); // mode 5 = 14-bit-scale Order-0 rANS (16-bit renorm)
                out.push_back((u8)(nsym - 1)); // Append element to container
                for (int i = 0; i < 256; i++) if (f14[i]) { // Loop iteration
                    out.push_back((u8)i); // Append element to container
                    out.push_back((u8)(f14[i] & 0xff)); // Append element to container
                    out.push_back((u8)((f14[i] >> 8) & 0xff)); // Append element to container
                } // Close scope / block
                put_varint_mem(out, rev16.size()); // Encode / decode variable-length integer
                put_u32_le(out, x); // Execute statement
                for (size_t i = rev16.size(); i-- > 0;) { // Loop iteration
                    u16 w = rev16[i]; // Compute and assign value
                    out.push_back((u8)(w & 0xff)); // Append element to container
                    out.push_back((u8)(w >> 8)); // Append element to container
                } // Close scope / block
                return; // Return result to caller
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block

    u16 freq[256], cum[256]; // Execute statement
    normalize_freqs(cnt, n, freq, cum); // Execute statement

    std::vector<u8> localRev; // Allocate / manage dynamic buffer
    std::vector<u8>& rev = revBuf ? *revBuf : localRev; // Allocate / manage dynamic buffer
    rev.clear(); if (rev.capacity() < n + 64) rev.reserve(n + 64); // Execute statement
    u64 rcp[256]; // Execute statement
    for (int i = 0; i < 256; i++) rcp[i] = freq[i] ? (((1ULL << 43) + freq[i] - 1) / freq[i]) : 0; // Loop iteration
    u32 x = kRansL; // Compute and assign value
    for (size_t i = n; i-- > 0;) { // Loop iteration
        u8 s = in[i]; u32 f = freq[s]; // Compute and assign value
        u32 x_max = ((kRansL >> kProbBits) << 8) * f; // Compute and assign value
        while (x >= x_max) { rev.push_back((u8)(x & 0xff)); x >>= 8; } // Loop while condition holds
        u32 q = (u32)(((u64)x * rcp[s]) >> 43); // Compute and assign value
        x = (q << kProbBits) + (x - q * f) + cum[s]; // Compute and assign value
    } // Close scope / block
    std::vector<u8> hdr; hdr.reserve(512); // Allocate / manage dynamic buffer
    write_freq_table(hdr, freq); // Execute statement
    size_t compSize = 1 + hdr.size() + 4 + rev.size(); // Compute and assign value

    // For small streams (n < 4096), try compact 8-bit-scale rANS (mode 4) with 32-byte presence bitmap
    if (n < 4096) { // Conditional branch check
        u16 f8[256] = {}, c8[256] = {}; // Compute and assign value
        u32 tot8 = 0, maxC8 = 0; int maxS8 = 0, nsym8 = 0; // Compute and assign value
        for (int i = 0; i < 256; i++) if (cnt[i]) { // Loop iteration
            nsym8++; // Execute statement
            u32 f = (u32)(((u64)cnt[i] * 256u + n / 2) / n); // Compute and assign value
            if (f == 0) f = 1; // Conditional branch check
            f8[i] = (u16)f; tot8 += f; // Compute and assign value
            if (cnt[i] > maxC8) { maxC8 = cnt[i]; maxS8 = i; } // Conditional branch check
        } // Close scope / block
        if (nsym8 > 0 && nsym8 <= 256) { // Conditional branch check
            while (tot8 > 256u) { // Loop while condition holds
                int best = -1; u32 bf = 1; // Compute and assign value
                for (int i = 0; i < 256; i++) if (f8[i] > bf) { bf = f8[i]; best = i; } // Loop iteration
                if (best < 0) break; // Conditional branch check
                u32 sub = std::min(tot8 - 256u, (u32)f8[best] - 1u); // Compute and assign value
                f8[best] = (u16)(f8[best] - sub); tot8 -= sub; // Compute and assign value
            } // Close scope / block
            if (tot8 < 256u) f8[maxS8] = (u16)(f8[maxS8] + (256u - tot8)); // Conditional branch check
            if (tot8 == 256u) { // Conditional branch check
                u32 cc = 0; for (int i = 0; i < 256; i++) { c8[i] = (u16)cc; cc += f8[i]; } // Compute and assign value
                std::vector<u8> rev8; rev8.reserve(n + 32); // Allocate / manage dynamic buffer
                u32 x8 = kRansL; // Compute and assign value
                for (size_t i = n; i-- > 0;) { // Loop iteration
                    u8 s = in[i]; u32 f = f8[s]; // Compute and assign value
                    u32 xm = ((kRansL >> 8) << 8) * f; // Compute and assign value
                    while (x8 >= xm) { rev8.push_back((u8)(x8 & 0xff)); x8 >>= 8; } // Loop while condition holds
                    x8 = ((x8 / f) << 8) + (x8 % f) + c8[s]; // Compute and assign value
                } // Close scope / block
                // Compact header: 32-byte bitmap + 1 byte per active symbol (f8-1)
                size_t m4Size = 1 + 32 + (size_t)nsym8 + 2 + 4 + rev8.size(); // Compute and assign value
                if (m4Size < compSize && m4Size < 1 + n) { // Conditional branch check
                    out.push_back(4); // mode 4 = Compact 8-bit-scale rANS
                    u8 bm[32] = {}; // Compute and assign value
                    for (int i = 0; i < 256; i++) if (f8[i]) bm[i >> 3] |= (u8)(1u << (i & 7)); // Loop iteration
                    out.insert(out.end(), bm, bm + 32); // Execute statement
                    for (int i = 0; i < 256; i++) if (f8[i]) out.push_back((u8)(f8[i] - 1)); // Loop iteration
                    put_varint_mem(out, rev8.size()); // Encode / decode variable-length integer
                    put_u32_le(out, x8); // Execute statement
                    for (size_t i = rev8.size(); i-- > 0;) out.push_back(rev8[i]); // Loop iteration
                    return; // Return result to caller
                } // Close scope / block
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block

    if (compSize >= 1 + n) { // Conditional branch check
        out.push_back(0); // raw
        out.insert(out.end(), in, in + n); // Execute statement
        return; // Return result to caller
    } // Close scope / block
    out.push_back(1); // mode 1 = Order-0 rANS
    out.insert(out.end(), hdr.begin(), hdr.end()); // Execute statement
    put_varint_mem(out, rev.size()); // Encode / decode variable-length integer
    put_u32_le(out, x); // Execute statement
    for (size_t i = rev.size(); i-- > 0;) out.push_back(rev[i]); // Loop iteration
} // Close scope / block

// Order-1 (16-context, prev >> 4) 32-bit rANS encoder for literals (fast entropy pre-check avoids double encoding)
inline void rans_encode_o1(const u8* in, size_t n, std::vector<u8>& out, FastScratch* sc = nullptr) { // Allocate / manage dynamic buffer
    if (n < 4096) { rans_encode_stream(in, n, out, sc ? &sc->rev : nullptr); return; } // Conditional branch check
    u32 cnt[16][256] = {}; // Compute and assign value
    u32 cnt0[256] = {}; // Compute and assign value
    size_t ctot[16] = {}; // Compute and assign value
    for (size_t i = 0; i < n; i++) { // Loop iteration
        int ctx = i ? (in[i - 1] >> 4) : 0; // Compute and assign value
        u8 v = in[i]; // Compute and assign value
        cnt[ctx][v]++; ctot[ctx]++; cnt0[v]++; // Execute statement
    } // Close scope / block
    // Fast theoretical size estimate: O1 vs O0 (avoids running both rANS loops)
    double o0Bits = 0.0, o1Bits = 0.0; // Compute and assign value
    size_t o0Syms = 0, o1Syms = 0; // Compute and assign value
    for (int i = 0; i < 256; i++) if (cnt0[i]) { o0Syms++; o0Bits -= (double)cnt0[i] * std::log2((double)cnt0[i] / (double)n); } // Loop iteration
    for (int c = 0; c < 16; c++) if (ctot[c]) { // Loop iteration
        double inv = 1.0 / (double)ctot[c]; // Compute and assign value
        for (int i = 0; i < 256; i++) if (cnt[c][i]) { o1Syms++; o1Bits -= (double)cnt[c][i] * std::log2((double)cnt[c][i] * inv); } // Loop iteration
    } // Close scope / block
    double estO0Bytes = o0Bits / 8.0 + (double)o0Syms * 2.5 + 16.0; // Compute and assign value
    double estO1Bytes = o1Bits / 8.0 + (double)o1Syms * 2.5 + 32.0; // Compute and assign value
    if (estO1Bytes + 16.0 >= estO0Bytes) { // Conditional branch check
        rans_encode_stream(in, n, out, sc ? &sc->rev : nullptr); // rANS entropy coding step
        return; // Return result to caller
    } // Close scope / block
    u16 freq[16][256], cum[16][256]; // Execute statement
    std::vector<u8> hdr; hdr.reserve(4096); // Allocate / manage dynamic buffer
    for (int c = 0; c < 16; c++) { // Loop iteration
        normalize_freqs(cnt[c], ctot[c] ? ctot[c] : 1, freq[c], cum[c]); // Execute statement
        write_freq_table(hdr, freq[c]); // Execute statement
    } // Close scope / block
    std::vector<u8> localRev; // Allocate / manage dynamic buffer
    std::vector<u8>& rev = sc ? sc->rev : localRev; // Allocate / manage dynamic buffer
    rev.clear(); if (rev.capacity() < n + 64) rev.reserve(n + 64); // Execute statement
    u64 rcp[16][256]; // Execute statement
    for (int c = 0; c < 16; c++) for (int i = 0; i < 256; i++) rcp[c][i] = freq[c][i] ? (((1ULL << 43) + freq[c][i] - 1) / freq[c][i]) : 0; // Loop iteration
    u32 x = kRansL; // Compute and assign value
    for (size_t i = n; i-- > 0;) { // Loop iteration
        int ctx = i ? (in[i - 1] >> 4) : 0; // Compute and assign value
        u8 s = in[i]; u32 f = freq[ctx][s]; // Compute and assign value
        u32 x_max = ((kRansL >> kProbBits) << 8) * f; // Compute and assign value
        while (x >= x_max) { rev.push_back((u8)(x & 0xff)); x >>= 8; } // Loop while condition holds
        u32 q = (u32)(((u64)x * rcp[ctx][s]) >> 43); // Compute and assign value
        x = (q << kProbBits) + (x - q * f) + cum[ctx][s]; // Compute and assign value
    } // Close scope / block
    size_t o1Size = 1 + hdr.size() + 4 + rev.size() + 4; // Compute and assign value
    if (o1Size >= 1 + n) { // Conditional branch check
        put_varint_mem(out, n); // Encode / decode variable-length integer
        out.push_back(0); // Append element to container
        out.insert(out.end(), in, in + n); // Execute statement
        return; // Return result to caller
    } // Close scope / block
    put_varint_mem(out, n); // Encode / decode variable-length integer
    out.push_back(2); // mode 2 = Order-1 (16-ctx) rANS
    out.insert(out.end(), hdr.begin(), hdr.end()); // Execute statement
    put_varint_mem(out, rev.size()); // Encode / decode variable-length integer
    put_u32_le(out, x); // Execute statement
    for (size_t i = rev.size(); i-- > 0;) out.push_back(rev[i]); // Loop iteration
} // Close scope / block

inline void rans_decode_stream(const u8*& p, const u8* end, std::vector<u8>& out, size_t maxOut = ((size_t)64 << 20)) { // Bounded rANS stream decoder
    u64 n = 0; int sh = 0; // Decoded output length varint
    for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast stream header"}; u8 c = *p++; n |= (u64)(c & 127) << sh; if (!(c & 128)) break; sh += 7; } // Read varint
    if (n > (u64)maxOut) throw Failure{Status::Corrupt, "fast stream size exceeds limit"}; // Prevent OOM DoS from crafted varint
    out.resize((size_t)n); // Safe bounded allocation
    if (n == 0) return; // Conditional branch check
    if (p >= end) throw Failure{Status::Corrupt, "fast stream mode"}; // Conditional branch check
    u8 mode = *p++; // Compute and assign value
    if (mode == 0) { // Conditional branch check
        if ((size_t)(end - p) < (size_t)n) throw Failure{Status::Corrupt, "fast raw truncated"}; // Conditional branch check
        std::memcpy(out.data(), p, (size_t)n); p += (size_t)n; return; // Fast raw memory copy
    } // Close scope / block
    if (mode == 1) { // Conditional branch check
        u16 freq[256], cum[256]; u8 lut[kProbScale]; // Execute statement
        read_freq_table(p, end, freq, cum, lut); // Execute statement
        u64 rlen = 0; sh = 0; // Compute and assign value
        for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast rlen"}; u8 b = *p++; rlen |= (u64)(b & 127) << sh; if (!(b & 128)) break; sh += 7; } // Loop iteration
        if ((u64)(end - p) < 4 + rlen) throw Failure{Status::Corrupt, "fast rans truncated"}; // Conditional branch check
        u32 x = get_u32_le(p); p += 4; // Compute and assign value
        const u8* rp = p; const u8* rend = p + (size_t)rlen; // Compute and assign value
        u8* dst = out.data(); // Compute and assign value
        for (size_t i = 0; i < (size_t)n; i++) { // Loop iteration
            u32 slot = x & (kProbScale - 1); // Compute and assign value
            u8 s = lut[slot]; // Compute and assign value
            dst[i] = s; // Compute and assign value
            x = freq[s] * (x >> kProbBits) + (slot - cum[s]); // Compute and assign value
            while (x < kRansL && rp < rend) x = (x << 8) | *rp++; // Loop while condition holds
        } // Close scope / block
        p = rend; // Compute and assign value
        return; // Return result to caller
    } // Close scope / block
    if (mode == 2) { // Conditional branch check
        u16 freq[16][256], cum[16][256]; // Execute statement
        std::vector<u8> lut(16 * kProbScale); // Allocate / manage dynamic buffer
        for (int c = 0; c < 16; c++) read_freq_table(p, end, freq[c], cum[c], lut.data() + c * kProbScale); // Loop iteration
        u64 rlen = 0; sh = 0; // Compute and assign value
        for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast o1 rlen"}; u8 b = *p++; rlen |= (u64)(b & 127) << sh; if (!(b & 128)) break; sh += 7; } // Loop iteration
        if ((u64)(end - p) < 4 + rlen) throw Failure{Status::Corrupt, "fast o1 truncated"}; // Conditional branch check
        u32 x = get_u32_le(p); p += 4; // Compute and assign value
        const u8* rp = p; const u8* rend = p + (size_t)rlen; // Compute and assign value
        u8* dst = out.data(); // Compute and assign value
        int ctx = 0; // Compute and assign value
        for (size_t i = 0; i < (size_t)n; i++) { // Loop iteration
            u32 slot = x & (kProbScale - 1); // Compute and assign value
            u8 s = lut[ctx * kProbScale + slot]; // Compute and assign value
            dst[i] = s; // Compute and assign value
            x = freq[ctx][s] * (x >> kProbBits) + (slot - cum[ctx][s]); // Compute and assign value
            while (x < kRansL && rp < rend) x = (x << 8) | *rp++; // Loop while condition holds
            ctx = s >> 4; // Compute and assign value
        } // Close scope / block
        p = rend; // Compute and assign value
        return; // Return result to caller
    } // Close scope / block
    if (mode == 4) { // Conditional branch check
        if ((size_t)(end - p) < 32) throw Failure{Status::Corrupt, "fast m4 bm"}; // Conditional branch check
        const u8* bm = p; p += 32; // Compute and assign value
        u16 freq[256] = {}, cum[256] = {}; u8 lut[256]; // Compute and assign value
        u32 c = 0; // Compute and assign value
        for (int i = 0; i < 256; i++) { // Loop iteration
            if (bm[i >> 3] & (1u << (i & 7))) { // Conditional branch check
                if (p >= end) throw Failure{Status::Corrupt, "fast m4 freq"}; // Conditional branch check
                u16 f = (u16)(*p++) + 1; // Compute and assign value
                freq[i] = f; cum[i] = (u16)c; // Compute and assign value
                if (c + f > 256u) throw Failure{Status::Corrupt, "fast m4 sum"}; // Conditional branch check
                std::memset(lut + c, i, f); // Initialize memory buffer
                c += f; // Compute and assign value
            } // Close scope / block
        } // Close scope / block
        if (c != 256u) throw Failure{Status::Corrupt, "fast m4 total"}; // Conditional branch check
        u64 rlen = 0; sh = 0; // Compute and assign value
        for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast m4 rlen"}; u8 b = *p++; rlen |= (u64)(b & 127) << sh; if (!(b & 128)) break; sh += 7; } // Loop iteration
        if ((u64)(end - p) < 4 + rlen) throw Failure{Status::Corrupt, "fast m4 trunc"}; // Conditional branch check
        u32 x = get_u32_le(p); p += 4; // Compute and assign value
        const u8* rp = p; const u8* rend = p + (size_t)rlen; // Compute and assign value
        u8* dst = out.data(); // Compute and assign value
        for (size_t i = 0; i < (size_t)n; i++) { // Loop iteration
            u32 slot = x & 255u; // Compute and assign value
            u8 s = lut[slot]; // Compute and assign value
            dst[i] = s; // Compute and assign value
            x = freq[s] * (x >> 8) + (slot - cum[s]); // Compute and assign value
            while (x < kRansL && rp < rend) x = (x << 8) | *rp++; // Loop while condition holds
        } // Close scope / block
        p = rend; // Compute and assign value
        return; // Return result to caller
    } // Close scope / block
    if (mode == 5) { // Conditional branch check
        if (p >= end) throw Failure{Status::Corrupt, "fast m5 nsym"}; // Conditional branch check
        int nsym = (int)(*p++) + 1; // Compute and assign value
        if ((size_t)(end - p) < (size_t)nsym * 3) throw Failure{Status::Corrupt, "fast m5 tbl"}; // Conditional branch check
        u16 freq[256] = {}, cum[256] = {}; // Compute and assign value
        std::vector<u8> lut(16384); // Allocate / manage dynamic buffer
        u32 c = 0; // Compute and assign value
        for (int k = 0; k < nsym; k++) { // Loop iteration
            u8 sym = *p++; // Compute and assign value
            u16 f = (u16)p[0] | ((u16)p[1] << 8); p += 2; // Compute and assign value
            if (f == 0 || c + f > 16384u) throw Failure{Status::Corrupt, "fast m5 freq"}; // Conditional branch check
            freq[sym] = f; cum[sym] = (u16)c; // Compute and assign value
            std::memset(lut.data() + c, sym, f); // Initialize memory buffer
            c += f; // Compute and assign value
        } // Close scope / block
        if (c != 16384u) throw Failure{Status::Corrupt, "fast m5 total"}; // Conditional branch check
        u64 wlen = 0; sh = 0; // Compute and assign value
        for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast m5 wlen"}; u8 b = *p++; wlen |= (u64)(b & 127) << sh; if (!(b & 128)) break; sh += 7; } // Loop iteration
        if ((u64)(end - p) < 4 + wlen * 2) throw Failure{Status::Corrupt, "fast m5 trunc"}; // Conditional branch check
        u32 x = get_u32_le(p); p += 4; // Compute and assign value
        const u8* rp = p; const u8* rend = p + (size_t)(wlen * 2); // Compute and assign value
        u8* dst = out.data(); // Compute and assign value
        static constexpr u32 kL14 = 1u << 15; // Define helper function or constant
        for (size_t i = 0; i < (size_t)n; i++) { // Loop iteration
            u32 slot = x & 16383u; // Compute and assign value
            u8 s = lut[slot]; // Compute and assign value
            dst[i] = s; // Compute and assign value
            x = freq[s] * (x >> 14) + (slot - cum[s]); // Compute and assign value
            if (x < kL14 && rp + 2 <= rend) { // Conditional branch check
                u32 w = (u32)rp[0] | ((u32)rp[1] << 8); rp += 2; // Compute and assign value
                x = (x << 16) | w; // Compute and assign value
            } // Close scope / block
        } // Close scope / block
        p = rend; // Compute and assign value
        return; // Return result to caller
    } // Close scope / block
    throw Failure{Status::Corrupt, "fast mode invalid"}; // Raise runtime exception
} // Close scope / block

// 16-context rANS encoder/decoder driven by an explicit context array (0..15) for Fast Audio
inline u32 fast_bitlen(u32 v) { u32 n = 0; while (v) { n++; v >>= 1; } return n; } // Define helper function or constant

inline void rans_encode_ctx16(const u8* in, const u8* ctxArr, size_t n, std::vector<u8>& out, FastScratch* sc = nullptr) { // Allocate / manage dynamic buffer
    if (n < 2048) { rans_encode_stream(in, n, out, sc ? &sc->rev : nullptr); return; } // Conditional branch check
    u32 cnt[16][256] = {}; // Compute and assign value
    size_t ctot[16] = {}; // Compute and assign value
    for (size_t i = 0; i < n; i++) { // Loop iteration
        int c = ctxArr[i] & 15; // Compute and assign value
        cnt[c][in[i]]++; ctot[c]++; // Execute statement
    } // Close scope / block
    u16 freq[16][256], cum[16][256]; // Execute statement
    std::vector<u8> hdr; hdr.reserve(4096); // Allocate / manage dynamic buffer
    for (int c = 0; c < 16; c++) { // Loop iteration
        normalize_freqs(cnt[c], ctot[c] ? ctot[c] : 1, freq[c], cum[c]); // Execute statement
        write_freq_table(hdr, freq[c]); // Execute statement
    } // Close scope / block
    std::vector<u8> localRev; // Allocate / manage dynamic buffer
    std::vector<u8>& rev = sc ? sc->rev : localRev; // Allocate / manage dynamic buffer
    rev.clear(); if (rev.capacity() < n + 64) rev.reserve(n + 64); // Execute statement
    u64 rcp[16][256]; // Execute statement
    for (int c = 0; c < 16; c++) for (int i = 0; i < 256; i++) rcp[c][i] = freq[c][i] ? (((1ULL << 43) + freq[c][i] - 1) / freq[c][i]) : 0; // Loop iteration
    u32 x = kRansL; // Compute and assign value
    for (size_t i = n; i-- > 0;) { // Loop iteration
        int c = ctxArr[i] & 15; // Compute and assign value
        u8 s = in[i]; u32 f = freq[c][s]; // Compute and assign value
        u32 x_max = ((kRansL >> kProbBits) << 8) * f; // Compute and assign value
        while (x >= x_max) { rev.push_back((u8)(x & 0xff)); x >>= 8; } // Loop while condition holds
        u32 q = (u32)(((u64)x * rcp[c][s]) >> 43); // Compute and assign value
        x = (q << kProbBits) + (x - q * f) + cum[c][s]; // Compute and assign value
    } // Close scope / block
    std::vector<u8> localO0; // Allocate / manage dynamic buffer
    std::vector<u8>& o0 = sc ? sc->o0 : localO0; // Allocate / manage dynamic buffer
    o0.clear(); if (o0.capacity() < n + 512) o0.reserve(n + 512); // Execute statement
    rans_encode_stream(in, n, o0); // rANS entropy coding step
    size_t c16Size = 1 + hdr.size() + 4 + rev.size() + 4; // Compute and assign value
    if (c16Size >= o0.size()) { // Conditional branch check
        out.insert(out.end(), o0.begin(), o0.end()); // Execute statement
        return; // Return result to caller
    } // Close scope / block
    put_varint_mem(out, n); // Encode / decode variable-length integer
    out.push_back(3); // mode 3 = 16-context magnitude-conditioned rANS
    out.insert(out.end(), hdr.begin(), hdr.end()); // Execute statement
    put_varint_mem(out, rev.size()); // Encode / decode variable-length integer
    put_u32_le(out, x); // Execute statement
    for (size_t i = rev.size(); i-- > 0;) out.push_back(rev[i]); // Loop iteration
} // Close scope / block

// Decodes mode 0/1/2 via rans_decode_stream, or mode 3 via causal context callback
template <class CtxFn> // Template declaration
inline void rans_decode_ctx16(const u8*& p, const u8* end, std::vector<u8>& out, CtxFn nextCtx, size_t maxOut = ((size_t)64 << 20)) { // Bounded 16-ctx rANS decoder
    const u8* p0 = p; // Save start pointer for fallback
    u64 n = 0; int sh = 0; // Decoded length varint
    for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast ctx16 hdr"}; u8 c = *p++; n |= (u64)(c & 127) << sh; if (!(c & 128)) break; sh += 7; } // Read varint
    if (n > (u64)maxOut) throw Failure{Status::Corrupt, "fast ctx16 size exceeds limit"}; // Prevent OOM DoS
    if (n == 0) { out.clear(); return; } // Empty stream
    if (p >= end) throw Failure{Status::Corrupt, "fast ctx16 mode"}; // Bounds check
    u8 mode = *p; // Stream mode byte
    if (mode != 3) { p = p0; rans_decode_stream(p, end, out, maxOut); return; } // Bounded fallback to O0/O1
    p++; // Consume mode byte
    out.resize((size_t)n); // Safe bounded allocation
    u16 freq[16][256], cum[16][256]; // Execute statement
    std::vector<u8> lut(16 * kProbScale); // Allocate / manage dynamic buffer
    for (int c = 0; c < 16; c++) read_freq_table(p, end, freq[c], cum[c], lut.data() + c * kProbScale); // Loop iteration
    u64 rlen = 0; sh = 0; // Compute and assign value
    for (;;) { if (p >= end || sh > 56) throw Failure{Status::Corrupt, "fast ctx16 rlen"}; u8 b = *p++; rlen |= (u64)(b & 127) << sh; if (!(b & 128)) break; sh += 7; } // Loop iteration
    if ((u64)(end - p) < 4 + rlen) throw Failure{Status::Corrupt, "fast ctx16 truncated"}; // Conditional branch check
    u32 x = get_u32_le(p); p += 4; // Compute and assign value
    const u8* rp = p; const u8* rend = p + (size_t)rlen; // Compute and assign value
    u8* dst = out.data(); // Compute and assign value
    for (size_t i = 0; i < (size_t)n; i++) { // Loop iteration
        int ctx = nextCtx(i) & 15; // Compute and assign value
        u32 slot = x & (kProbScale - 1); // Compute and assign value
        u8 s = lut[ctx * kProbScale + slot]; // Compute and assign value
        dst[i] = s; // Compute and assign value
        x = freq[ctx][s] * (x >> kProbBits) + (slot - cum[ctx][s]); // Compute and assign value
        while (x < kRansL && rp < rend) x = (x << 8) | *rp++; // Loop while condition holds
    } // Close scope / block
    p = rend; // Compute and assign value
} // Close scope / block

// LZMA-style Fast Codec: 8MB window, 4-way 4-byte hash + 3-byte hash, 4 rep-distance caches, bit-cost lazy parse, Order-1 rANS
// Token byte format:
//   bit 7 == 0 (normal match): [0 | llCode(3b: 0..6, 7=extra) | mlCode(4b: 0..14 -> len 3..17, 15=18+extra)]
//   bit 7 == 1 (rep match):    [1 | repIdx(2b: 0..3) | llCode(2b: 0..2, 3=extra) | mlCode(3b: 0..6 -> len 2..8, 7=9+extra)]
inline void encode_segment_fast(const u8* data, size_t L, const Hint& hint, std::vector<u8>& out, Counters& cn, FastScratch* reuseSc = nullptr) { // Allocate / manage dynamic buffer
    if (cn.cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
    // Fast Audio path: MSB-first byte-plane split + per-channel magnitude-conditioned 16-context rANS
    if (hint.ate > hint.ats && hint.width >= 1 && hint.width <= 3) { // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
        u64 _tAR0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        out.clear(); // Execute statement
        const size_t ats = (size_t)hint.ats, ate = (size_t)hint.ate, w = (size_t)hint.width; // Compute and assign value
        const size_t chs = (size_t)(hint.channels > 0 && hint.channels <= 8 ? hint.channels : 1); // Compute and assign value
        const size_t nSamp = (ate - ats) / w; // Compute and assign value
        std::vector<u8> plane(nSamp), ctx(nSamp); // Allocate / manage dynamic buffer
        std::vector<u32> hiAcc(nSamp, 0); // already-coded higher bytes of each sample
        rans_encode_stream(data, ats, out, reuseSc ? &reuseSc->rev : nullptr); // rANS entropy coding step
        for (int b = (int)w - 1; b >= 0; b--) { // Loop iteration
            const u8* p = data + ats + (size_t)b; // Compute and assign value
            u32 avg[8] = {}; // Compute and assign value
            for (size_t k = 0; k < nSamp; k++) { // Loop iteration
                size_t ch = k % chs; // Compute and assign value
                u8 v = p[k * w]; // Compute and assign value
                plane[k] = v; // Compute and assign value
                u32 bl = fast_bitlen(avg[ch] >> 4); // Compute and assign value
                if (b == (int)w - 1) { // Conditional branch check
                    // Top byte (MSB): condition on same-channel running magnitude (0..8) + non-zero flag
                    ctx[k] = (u8)std::min<u32>(15u, bl * 2u + (avg[ch] >= 16u ? 1u : 0u)); // Compute and assign value
                } else { // Execute statement
                    // Lower byte (LSB/mid): condition on whether current sample's higher bytes are zero + running magnitude
                    u32 hi = hiAcc[k]; // Compute and assign value
                    if (hi == 0) ctx[k] = (u8)std::min<u32>(11u, bl); // Conditional branch check
                    else if (hi == 1) ctx[k] = 12; // Alternative conditional branch
                    else if (hi == 2) ctx[k] = 13; // Alternative conditional branch
                    else if (hi < 8)  ctx[k] = 14; // Alternative conditional branch
                    else              ctx[k] = 15; // Fallback branch
                } // Close scope / block
                u32 curMag = (hiAcc[k] << 8) | (u32)v; // Compute and assign value
                hiAcc[k] = curMag; // Compute and assign value
                avg[ch] = (avg[ch] * 3u + (curMag << 4) + 2u) >> 2; // Compute and assign value
            } // Close scope / block
            rans_encode_ctx16(plane.data(), ctx.data(), nSamp, out, reuseSc); // rANS entropy coding step
        } // Close scope / block
        rans_encode_stream(data + ate, L - ate, out, reuseSc ? &reuseSc->rev : nullptr); // rANS entropy coding step
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.wFastRansUs.fetch_add(prof_now_us() - _tAR0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
        cn.done.fetch_add(L, std::memory_order_relaxed); // Execute statement
        cn.out.fetch_add(out.size(), std::memory_order_relaxed); // Execute statement
        return; // Return result to caller
    } // Close scope / block
    if (cn.cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
    static constexpr size_t kHash4Bits = 15, kHash4Size = 1u << kHash4Bits, kWays = 8; // Define helper function or constant
    static constexpr size_t kHash3Bits = 15, kHash3Size = 1u << kHash3Bits; // Define helper function or constant
    static constexpr size_t kMaxDist = (1u << 23) - 1; // Define helper function or constant
    FastScratch localSc; // Execute statement
    FastScratch& sc = reuseSc ? *reuseSc : localSc; // Compute and assign value
    sc.reset(L, kHash4Size, kWays, kHash3Size); // Execute statement
    auto& head4 = sc.head4; auto& head3 = sc.head3; // Compute and assign value
    auto& lits = sc.lits; auto& toks = sc.toks; // Compute and assign value
    auto& offLo = sc.offLo; auto& offMid = sc.offMid; auto& offHi = sc.offHi; auto& extra = sc.extra; // Compute and assign value

    auto h4 = [](const u8* p) -> u32 { // Compute and assign value
        u32 v; std::memcpy(&v, p, 4); // Fast raw memory copy
        return (v * 2654435761u) >> (32 - kHash4Bits); // Return result to caller
    }; // Close scope / block
    auto h3 = [](const u8* p) -> u32 { // Compute and assign value
        u32 v = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16); // Compute and assign value
        return (v * 2654435761u) >> (32 - kHash3Bits); // Return result to caller
    }; // Close scope / block

    size_t rep[4] = {1, 1, 1, 1}; // Compute and assign value

    struct BestMatch { // Declare data structure / class
        size_t len = 0; // Compute and assign value
        size_t dist = 0; // Compute and assign value
        int repIdx = -1; // -1 = normal, 0..3 = rep cache
        int score = 0;   // estimated net bit savings
    }; // Close scope / block

    // Calibrate literal bit cost from actual segment order-0 entropy (< 0.05 ms)
    const size_t hSample = std::min<size_t>(L, 16384); // Compute and assign value
    const double segH0 = hSample ? block_entropy(data, hSample) : 8.0; // Compute and assign value
    const int litBitsQ4 = clampi((int)(segH0 * 16.0 + 0.5), 4, 128); // Compute and assign value

    const bool highEnt = (segH0 >= 7.0); // Compute and assign value
    auto eval_score = [litBitsQ4, highEnt](size_t len, size_t dist, int repIdx) -> int { // Compute and assign value
        int litBits = ((int)len * litBitsQ4 + 8) >> 4; // Compute and assign value
        if (repIdx >= 0) { // Conditional branch check
            int cost = (highEnt ? 4 : 6) + (repIdx > 0 ? 1 : 0) + (len >= 9 ? 7 : 0); // Compute and assign value
            return litBits - cost; // Return result to caller
        } // Close scope / block
        int dCost = highEnt // Compute and assign value
            ? ((dist < 256) ? 8 : (dist < 16384 ? 13 : (dist < 65536 ? 16 : 20))) // Execute statement
            : ((dist < 256) ? 10 : (dist < 16384 ? 16 : (dist < 65536 ? 19 : 24))); // Execute statement
        int cost = (highEnt ? 5 : 7) + dCost + (len >= 18 ? 7 : 0); // Compute and assign value
        return litBits - cost; // Return result to caller
    }; // Close scope / block

    auto match_len = [&](size_t ref, size_t pos, size_t startLen, size_t maxM) -> size_t { // Compute and assign value
        size_t mlen = startLen; // Compute and assign value
        while (mlen + 8 <= maxM && std::memcmp(data + ref + mlen, data + pos + mlen, 8) == 0) mlen += 8; // Loop while condition holds
        while (mlen < maxM && data[ref + mlen] == data[pos + mlen]) mlen++; // Loop while condition holds
        return mlen; // Return result to caller
    }; // Close scope / block

    auto find_best = [&](size_t pos) -> BestMatch { // Compute and assign value
        BestMatch best; // Execute statement
        size_t maxM = L - pos; // Compute and assign value
        if (maxM < 3) return best; // Conditional branch check

        // 1. Check 4 rep-distance caches first (allow len >= 2)
        for (int r = 0; r < 4; r++) { // Loop iteration
            size_t d = rep[r]; // Compute and assign value
            if (d == 0 || d > pos) continue; // Conditional branch check
            size_t ref = pos - d; // Compute and assign value
            if (data[ref] == data[pos] && data[ref + 1] == data[pos + 1]) { // Conditional branch check
                size_t ml = match_len(ref, pos, 2, maxM); // Compute and assign value
                int sc = eval_score(ml, d, r); // Compute and assign value
                if (sc > best.score) { best = {ml, d, r, sc}; } // Conditional branch check
            } // Close scope / block
        } // Close scope / block

        // 2. Check 3-byte hash for short-distance 3-byte matches
        u32 k3 = h3(data + pos); // Compute and assign value
        u32 cand3 = head3[k3]; // Compute and assign value
        head3[k3] = (u32)(pos + 1); // Compute and assign value
        if (cand3) { // Conditional branch check
            size_t ref = (size_t)(cand3 - 1); // Compute and assign value
            size_t dist = pos - ref; // Compute and assign value
            if (dist > 0 && dist <= 16384 && data[ref] == data[pos] && data[ref + 1] == data[pos + 1] && data[ref + 2] == data[pos + 2]) { // Conditional branch check
                size_t ml = match_len(ref, pos, 3, maxM); // Compute and assign value
                int rIdx = -1; // Compute and assign value
                for (int r = 0; r < 4; r++) if (rep[r] == dist) { rIdx = r; break; } // Loop iteration
                int sc = eval_score(ml, dist, rIdx); // Compute and assign value
                if (sc > best.score) { best = {ml, dist, rIdx, sc}; } // Conditional branch check
            } // Close scope / block
        } // Close scope / block

        // 3. Check 4-way 4-byte hash up to 8 MB
        if (maxM >= 4) { // Conditional branch check
            u32 k4 = h4(data + pos); // Compute and assign value
            u32* bucket = &head4[k4 * kWays]; // Compute and assign value
            for (size_t w = 0; w < kWays; w++) { // Loop iteration
                u32 cand = bucket[w]; // Compute and assign value
                if (!cand) break; // Conditional branch check
                size_t ref = (size_t)(cand - 1); // Compute and assign value
                size_t dist = pos - ref; // Compute and assign value
                if (dist == 0 || dist > kMaxDist) continue; // Conditional branch check
                if (best.len >= 4 && best.len < maxM && data[ref + best.len] != data[pos + best.len]) continue; // Conditional branch check
                if (std::memcmp(data + ref, data + pos, 4) == 0) { // Conditional branch check
                    size_t ml = match_len(ref, pos, 4, maxM); // Compute and assign value
                    int rIdx = -1; // Compute and assign value
                    for (int r = 0; r < 4; r++) if (rep[r] == dist) { rIdx = r; break; } // Loop iteration
                    int sc = eval_score(ml, dist, rIdx); // Compute and assign value
                    if (sc > best.score) { best = {ml, dist, rIdx, sc}; if (ml >= 128) break; } // Conditional branch check
                } // Close scope / block
            } // Close scope / block
            for (size_t xw = kWays - 1; xw > 0; xw--) bucket[xw] = bucket[xw - 1]; // Loop iteration
            bucket[0] = (u32)(pos + 1); // Compute and assign value
        } // Close scope / block
        return best; // Return result to caller
    }; // Close scope / block

    auto update_rep = [&](int rIdx, size_t dist) { // Compute and assign value
        if (rIdx == 0) return; // Conditional branch check
        if (rIdx == 1) { std::swap(rep[0], rep[1]); return; } // Conditional branch check
        if (rIdx == 2) { size_t d = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; return; } // Conditional branch check
        if (rIdx == 3) { size_t d = rep[3]; rep[3] = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; return; } // Conditional branch check
        rep[3] = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = dist; // Compute and assign value
    }; // Close scope / block

#ifdef P3_PROFILE // Guard if macro is defined
    u64 _tLz0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
    size_t i = 0, anchor = 0, misses = 0; // Compute and assign value
    while (i + 4 <= L) { // Loop while condition holds
        BestMatch m = find_best(i); // Compute and assign value
        if (m.score > 0) { // Conditional branch check
            misses = 0; // Compute and assign value
            // 1-step lazy evaluation: check if i+1 saves significantly more bits
            if (m.len < 64 && i + 5 <= L) { // Conditional branch check
                BestMatch m2 = find_best(i + 1); // Compute and assign value
                if (m2.score > m.score + 6) { // Conditional branch check
                    i++; // Execute statement
                    m = m2; // Compute and assign value
                } // Close scope / block
            } // Close scope / block
            size_t litLen = i - anchor; // Compute and assign value
            if (litLen) lits.insert(lits.end(), data + anchor, data + i); // Conditional branch check

            if (m.repIdx >= 0) { // Conditional branch check
                u32 llCode = litLen < 3 ? (u32)litLen : 3u; // Compute and assign value
                u32 mlCode = (m.len - 2) < 7 ? (u32)(m.len - 2) : 7u; // Compute and assign value
                toks.push_back((u8)(0x80u | ((u32)m.repIdx << 5) | (llCode << 3) | mlCode)); // Append element to container
                if (llCode == 3) put_varint_mem(extra, litLen - 3); // Conditional branch check
                if (mlCode == 7) put_varint_mem(extra, m.len - 2 - 7); // Conditional branch check
                update_rep(m.repIdx, m.dist); // Execute statement
            } else { // Execute statement
                u32 llCode = litLen < 7 ? (u32)litLen : 7u; // Compute and assign value
                u32 mlCode = (m.len - 3) < 15 ? (u32)(m.len - 3) : 15u; // Compute and assign value
                toks.push_back((u8)((llCode << 4) | mlCode)); // Append element to container
                if (llCode == 7) put_varint_mem(extra, litLen - 7); // Conditional branch check
                if (mlCode == 15) put_varint_mem(extra, m.len - 3 - 15); // Conditional branch check
                offLo.push_back((u8)(m.dist & 0xff)); // Append element to container
                offMid.push_back((u8)((m.dist >> 8) & 0xff)); // Append element to container
                offHi.push_back((u8)((m.dist >> 16) & 0xff)); // Append element to container
                update_rep(-1, m.dist); // Execute statement
            } // Close scope / block

            size_t endM = i + m.len; // Compute and assign value
            size_t step = m.len > 24 ? (m.len >> 3) : 2; // Compute and assign value
            for (size_t p = i + 2; p < endM && p + 4 <= L; p += step) { // Loop iteration
                head3[h3(data + p)] = (u32)(p + 1); // Compute and assign value
                u32* b = &head4[h4(data + p) * kWays]; // Compute and assign value
                for (size_t xw = kWays - 1; xw > 0; xw--) b[xw] = b[xw - 1]; // Loop iteration
                b[0] = (u32)(p + 1); // Compute and assign value
            } // Close scope / block
            i = endM; // Compute and assign value
            anchor = i; // Compute and assign value
            continue; // Advance to next loop iteration
        } // Close scope / block
        if (highEnt) { // Conditional branch check
            i++; // Execute statement
        } else { // Execute statement
            size_t skip = 1u + (misses >> 5); // Compute and assign value
            if (skip > 8u) skip = 8u; // Conditional branch check
            i += skip; // Compute and assign value
            misses++; // Execute statement
        } // Close scope / block
    } // Close scope / block
    if (anchor < L) { // Conditional branch check
        size_t litLen = L - anchor; // Compute and assign value
        u32 llCode = litLen < 7 ? (u32)litLen : 7u; // Compute and assign value
        toks.push_back((u8)(llCode << 4)); // Append element to container
        if (llCode == 7) put_varint_mem(extra, litLen - 7); // Conditional branch check
        lits.insert(lits.end(), data + anchor, data + L); // Execute statement
    } // Close scope / block

#ifdef P3_PROFILE // Guard if macro is defined
    u64 _tRans0 = prof_now_us(); // Compute and assign value
    g_prof.wFastLzUs.fetch_add(_tRans0 - _tLz0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
    out.clear(); // Execute statement
    // Pure-rANS fast path (segMode == 1) when LZ77 found negligible matches (< 5% reduction in literals)
    if (toks.size() <= 1 || lits.size() > (L - L / 20)) { // Conditional branch check
        out.push_back(1); // Append element to container
        rans_encode_o1(data, L, out, &sc); // rANS entropy coding step
    } else { // Execute statement
        out.push_back(0); // Append element to container
        rans_encode_stream(toks.data(), toks.size(), out, &sc.rev); // rANS entropy coding step
        rans_encode_o1(lits.data(), lits.size(), out, &sc); // rANS entropy coding step
        rans_encode_stream(offLo.data(), offLo.size(), out, &sc.rev); // rANS entropy coding step
        rans_encode_stream(offMid.data(), offMid.size(), out, &sc.rev); // rANS entropy coding step
        rans_encode_stream(offHi.data(), offHi.size(), out, &sc.rev); // rANS entropy coding step
        rans_encode_stream(extra.data(), extra.size(), out, &sc.rev); // rANS entropy coding step
        // Safety check: if LZ77+rANS did worse than theoretical Order-0 rANS, try Pure-rANS and keep smaller
        size_t estPure = (size_t)((double)L * segH0 / 8.0) + 512; // Compute and assign value
        if (out.size() > estPure) { // Conditional branch check
            std::vector<u8> pure; pure.reserve(estPure + 256); // Allocate / manage dynamic buffer
            pure.push_back(1); // Append element to container
            rans_encode_o1(data, L, pure, &sc); // rANS entropy coding step
            if (pure.size() < out.size()) out.swap(pure); // Conditional branch check
        } // Close scope / block
    } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
    g_prof.wFastRansUs.fetch_add(prof_now_us() - _tRans0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
    cn.done.fetch_add(L, std::memory_order_relaxed); // Execute statement
    cn.out.fetch_add(out.size(), std::memory_order_relaxed); // Execute statement
} // Close scope / block

inline void decode_segment_fast(const u8* payload, size_t plen, u8* out, size_t L, const Hint& hint, Counters* cn) { // Define helper function or constant
    if (cn && cn->cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
    if (hint.ate > hint.ats && hint.width >= 1 && hint.width <= 3) { // Conditional branch check
        const u8* p = payload; const u8* end = payload + plen; // Compute and assign value
        const size_t ats = (size_t)hint.ats, ate = (size_t)hint.ate, w = (size_t)hint.width; // Compute and assign value
        const size_t chs = (size_t)(hint.channels > 0 && hint.channels <= 8 ? hint.channels : 1); // Compute and assign value
        const size_t nSamp = (ate - ats) / w; // Compute and assign value
        std::vector<u8> tmp; // Allocate / manage dynamic buffer
        std::vector<u32> hiAcc(nSamp, 0); // Allocate / manage dynamic buffer
        rans_decode_stream(p, end, tmp, ats); // Bounded audio prefix decode
        if (tmp.size() != ats) throw Failure{Status::Corrupt, "fast audio prefix"}; // Conditional branch check
        if (ats) std::memcpy(out, tmp.data(), ats); // Conditional branch check
        for (int b = (int)w - 1; b >= 0; b--) { // Loop iteration
            u32 avg[8] = {}; // Compute and assign value
            size_t prevK = ~(size_t)0; // Compute and assign value
            rans_decode_ctx16(p, end, tmp, [&](size_t k) -> int { // rANS entropy coding step
                // Update running average from the just-decoded sample k-1
                if (prevK != ~(size_t)0) { // Conditional branch check
                    size_t pch = prevK % chs; // Compute and assign value
                    u32 curMag = (hiAcc[prevK] << 8) | (u32)tmp[prevK]; // Compute and assign value
                    hiAcc[prevK] = curMag; // Compute and assign value
                    avg[pch] = (avg[pch] * 3u + (curMag << 4) + 2u) >> 2; // Compute and assign value
                } // Close scope / block
                prevK = k; // Compute and assign value
                size_t ch = k % chs; // Compute and assign value
                u32 bl = fast_bitlen(avg[ch] >> 4); // Compute and assign value
                if (b == (int)w - 1) return (int)std::min<u32>(15u, bl * 2u + (avg[ch] >= 16u ? 1u : 0u)); // Conditional branch check
                u32 hi = hiAcc[k]; // Compute and assign value
                if (hi == 0) return (int)std::min<u32>(11u, bl); // Conditional branch check
                if (hi == 1) return 12; // Conditional branch check
                if (hi == 2) return 13; // Conditional branch check
                if (hi < 8)  return 14; // Conditional branch check
                return 15; // Return result to caller
            }, nSamp); // Bounded audio plane decode
            if (tmp.size() != nSamp) throw Failure{Status::Corrupt, "fast audio plane"}; // Conditional branch check
            // If mode != 3 (fallback to O0/raw), hiAcc wasn't updated inside callback; update cleanly here
            if (prevK == ~(size_t)0) { // Conditional branch check
                for (size_t k = 0; k < nSamp; k++) hiAcc[k] = (hiAcc[k] << 8) | (u32)tmp[k]; // Loop iteration
            } else { // Execute statement
                hiAcc[prevK] = (hiAcc[prevK] << 8) | (u32)tmp[prevK]; // Compute and assign value
            } // Close scope / block
            u8* dst = out + ats + (size_t)b; // Compute and assign value
            for (size_t k = 0; k < nSamp; k++) dst[k * w] = tmp[k]; // Loop iteration
        } // Close scope / block
        rans_decode_stream(p, end, tmp, L - ate); // Bounded audio suffix decode
        if (tmp.size() != L - ate) throw Failure{Status::Corrupt, "fast audio suffix"}; // Conditional branch check
        if (L > ate) std::memcpy(out + ate, tmp.data(), L - ate); // Conditional branch check
        if (cn) { cn->done.fetch_add(L, std::memory_order_relaxed); cn->out.fetch_add(L, std::memory_order_relaxed); } // Conditional branch check
        return; // Return result to caller
    } // Close scope / block
    if (cn && cn->cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
    const u8* p = payload; const u8* end = payload + plen; // Compute and assign value
    if (p >= end) throw Failure{Status::Corrupt, "fast seg mode"}; // Conditional branch check
    u8 segMode = *p++; // Compute and assign value
    if (segMode == 1) { // Conditional branch check
        std::vector<u8> tmp; // Allocate / manage dynamic buffer
        rans_decode_stream(p, end, tmp, L); // Bounded pure-rANS decode
        if (tmp.size() != L) throw Failure{Status::Corrupt, "fast pure rans length"}; // Conditional branch check
        std::memcpy(out, tmp.data(), L); // Fast raw memory copy
        if (cn) { cn->done.fetch_add(L, std::memory_order_relaxed); cn->out.fetch_add(L, std::memory_order_relaxed); } // Conditional branch check
        return; // Return result to caller
    } // Close scope / block
    if (segMode != 0) throw Failure{Status::Corrupt, "fast seg mode invalid"}; // Conditional branch check
    std::vector<u8> toks, lits, offLo, offMid, offHi, extra; // Allocate / manage dynamic buffer
    const size_t maxLz = L + 64; // Tight upper bound for any LZ77 component stream
    rans_decode_stream(p, end, toks, maxLz); // Bounded token decode
    rans_decode_stream(p, end, lits, L); // Bounded literal decode
    rans_decode_stream(p, end, offLo, maxLz); // Bounded offset-low decode
    rans_decode_stream(p, end, offMid, maxLz); // Bounded offset-mid decode
    rans_decode_stream(p, end, offHi, maxLz); // Bounded offset-high decode
    rans_decode_stream(p, end, extra, maxLz); // Bounded overflow varints decode

    size_t rep[4] = {1, 1, 1, 1}; // Compute and assign value
    size_t li = 0, oi = 0, ei = 0, op = 0; // Compute and assign value
    auto get_ex = [&]() -> size_t { // Compute and assign value
        u64 v = 0; int sh = 0; // Compute and assign value
        for (;;) { if (ei >= extra.size() || sh > 56) throw Failure{Status::Corrupt, "fast extra"}; u8 c = extra[ei++]; v |= (u64)(c & 127) << sh; if (!(c & 128)) break; sh += 7; } // Loop iteration
        return (size_t)v; // Return result to caller
    }; // Close scope / block
    for (size_t ti = 0; ti < toks.size(); ti++) { // Loop iteration
        u8 tk = toks[ti]; // Compute and assign value
        if (tk & 0x80u) { // Conditional branch check
            int rIdx = (tk >> 5) & 3; // Compute and assign value
            size_t litLen = (tk >> 3) & 3; // Compute and assign value
            if (litLen == 3) litLen += get_ex(); // Conditional branch check
            if (li + litLen > lits.size() || op + litLen > L) throw Failure{Status::Corrupt, "fast rep lit bounds"}; // Conditional branch check
            if (litLen) { std::memcpy(out + op, lits.data() + li, litLen); li += litLen; op += litLen; } // Conditional branch check
            size_t mlen = (tk & 7) + 2; // Compute and assign value
            if ((tk & 7) == 7) mlen += get_ex(); // Conditional branch check
            size_t dist = rep[rIdx]; // Compute and assign value
            if (rIdx == 1) std::swap(rep[0], rep[1]); // Conditional branch check
            else if (rIdx == 2) { size_t d = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; } // Alternative conditional branch
            else if (rIdx == 3) { size_t d = rep[3]; rep[3] = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = d; } // Alternative conditional branch
            if (dist == 0 || dist > op || op + mlen > L) throw Failure{Status::Corrupt, "fast rep match bounds"}; // Conditional branch check
            u8* dst = out + op; const u8* src = dst - dist; // Compute and assign value
            if (dist >= mlen) std::memcpy(dst, src, mlen); // Conditional branch check
            else for (size_t k = 0; k < mlen; k++) dst[k] = src[k]; // Fallback branch
            op += mlen; // Compute and assign value
        } else { // Execute statement
            size_t litLen = (tk >> 4) & 7; // Compute and assign value
            if (litLen == 7) litLen += get_ex(); // Conditional branch check
            if (li + litLen > lits.size() || op + litLen > L) throw Failure{Status::Corrupt, "fast lit bounds"}; // Conditional branch check
            if (litLen) { std::memcpy(out + op, lits.data() + li, litLen); li += litLen; op += litLen; } // Conditional branch check
            if (op == L) break; // Conditional branch check
            size_t mlen = (tk & 15) + 3; // Compute and assign value
            if ((tk & 15) == 15) mlen += get_ex(); // Conditional branch check
            if (oi >= offLo.size() || oi >= offMid.size() || oi >= offHi.size()) throw Failure{Status::Corrupt, "fast off bounds"}; // Conditional branch check
            size_t dist = (size_t)offLo[oi] | ((size_t)offMid[oi] << 8) | ((size_t)offHi[oi] << 16); oi++; // Compute and assign value
            rep[3] = rep[2]; rep[2] = rep[1]; rep[1] = rep[0]; rep[0] = dist; // Compute and assign value
            if (dist == 0 || dist > op || op + mlen > L) throw Failure{Status::Corrupt, "fast match bounds"}; // Conditional branch check
            u8* dst = out + op; const u8* src = dst - dist; // Compute and assign value
            if (dist >= mlen) std::memcpy(dst, src, mlen); // Conditional branch check
            else for (size_t k = 0; k < mlen; k++) dst[k] = src[k]; // Fallback branch
            op += mlen; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    if (op != L) throw Failure{Status::Corrupt, "fast length mismatch"}; // Conditional branch check
    if (cn) { cn->done.fetch_add(L, std::memory_order_relaxed); cn->out.fetch_add(L, std::memory_order_relaxed); } // Conditional branch check
} // Close scope / block
}  // namespace fast

inline void encode_segment(const u8* data, size_t L, int tbMax, const Hint& hint, std::vector<u8>& out, Counters& cn, Model* reuseModel = nullptr, fast::FastScratch* reuseFast = nullptr) { // Allocate / manage dynamic buffer
    if (tbMax == 1) { fast::encode_segment_fast(data, L, hint, out, cn, reuseFast); return; } // Conditional branch check
    const Tables& T = tables(); // Compute and assign value
    std::vector<u8> arith, side; arith.reserve(L / 2 + 64); // Allocate / manage dynamic buffer
    Enc e(arith); // Execute statement
#ifdef P3_PROFILE // Guard if macro is defined
    u64 _tRst0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
    std::unique_ptr<Model> localModel; // Execute statement
    if (reuseModel) reuseModel->reset(L, tbMax, hint); // Conditional branch check
    else localModel.reset(new Model(L, tbMax, hint)); // Fallback branch
    Model& m = reuseModel ? *reuseModel : *localModel; // Compute and assign value
#ifdef P3_PROFILE // Guard if macro is defined
    g_prof.wModelResetUs.fetch_add(prof_now_us() - _tRst0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
    std::vector<u16> pk(kBlock * 8); // Allocate / manage dynamic buffer
    u32 fcRaw[3] = {kInit, kInit, kInit}, fcSkip[2] = {kInit, kInit}; // Compute and assign value
    int prevType = 0, rawRun = 0, skipLeft = 0, skipLen = 15; size_t off = 0, lastOut = 0; // Compute and assign value
    while (off < L) { // Loop while condition holds
        if (cn.cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
        size_t B = std::min(kBlock, L - off); int type; size_t k = 0; // Compute and assign value
        if (skipLeft > 0 && block_entropy(data + off, B) < 7.7) { skipLeft = 0; rawRun = 0; skipLen = 15; }   // obvious structure: stop skipping
        if (skipLeft > 0) { type = 2; skipLeft--; } // Conditional branch check
        else { // Fallback branch
#ifdef P3_PROFILE // Guard if macro is defined
            u64 _tM0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
            double cost = 0; // Compute and assign value
            for (size_t i = 0; i < B; i++) { // Loop iteration
                u32 byte = data[off + i]; // Compute and assign value
                for (int b = 7; b >= 0; b--) { // Loop iteration
                    int y = (byte >> b) & 1, p = m.predict(); // Compute and assign value
                    cost += y ? T.lg1_[p] : T.lg1_[4096 - p]; // Compute and assign value
                    pk[k++] = (u16)((p << 1) | y); // Compute and assign value
                    m.update(y); // Execute statement
                } // Close scope / block
            } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
            g_prof.wPaqModelUs.fetch_add(prof_now_us() - _tM0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
            type = cost + 0.5 >= 8.0 * (double)B ? 1 : 0; // Compute and assign value
            if (type) { if (++rawRun >= 4) { skipLeft = skipLen; skipLen = std::min(skipLen * 2 + 1, 255); } } // Conditional branch check
            else { rawRun = 0; skipLen = 15; } // Fallback branch
        } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
        if (type == 0) g_prof.blkModeled.fetch_add(1, std::memory_order_relaxed); // Conditional branch check
        else if (type == 1) g_prof.blkRaw.fetch_add(1, std::memory_order_relaxed); // Alternative conditional branch
        else g_prof.blkSkipped.fetch_add(1, std::memory_order_relaxed); // Fallback branch
        u64 _tAc0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        e.bit(type != 0, clampi((int)(fcRaw[prevType] >> 20), 16, 4080)); sm_up(&fcRaw[prevType], type != 0, 30); // Compute and assign value
        if (type) { int c = prevType == 2; e.bit(type == 2, clampi((int)(fcSkip[c] >> 20), 16, 4080)); sm_up(&fcSkip[c], type == 2, 30); } // Conditional branch check
        prevType = type; // Compute and assign value
        if (type == 0) for (size_t j = 0; j < k; j++) e.bit(pk[j] & 1, pk[j] >> 1); // Conditional branch check
        else if (type == 1) for (size_t j = 0; j < k; j++) e.half(pk[j] & 1); // Alternative conditional branch
        else side.insert(side.end(), data + off, data + off + B); // Fallback branch
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.wPaqAcUs.fetch_add(prof_now_us() - _tAc0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
        off += B; // Compute and assign value
        cn.done.fetch_add(B, std::memory_order_relaxed); // Execute statement
        size_t now = arith.size() + side.size(); // Compute and assign value
        cn.out.fetch_add(now - lastOut, std::memory_order_relaxed); lastOut = now; // Compute and assign value
    } // Close scope / block
    e.finish(); // Execute statement
    cn.out.fetch_add(arith.size() + side.size() - lastOut, std::memory_order_relaxed); // Execute statement
    out.clear(); put_varint_mem(out, arith.size()); // Encode / decode variable-length integer
    out.insert(out.end(), arith.begin(), arith.end()); out.insert(out.end(), side.begin(), side.end()); // Execute statement
} // Close scope / block

inline void decode_segment(const u8* payload, size_t plen, u8* out, size_t L, int tbMax, const Hint& hint, Counters* cn, Model* reuseModel = nullptr) { // Define helper function or constant
    if (tbMax == 1) { fast::decode_segment_fast(payload, plen, out, L, hint, cn); return; } // Conditional branch check
    u64 alen = 0; size_t hdr = 0; int shift = 0; // Compute and assign value
    for (;;) { if (hdr >= plen || shift > 56) throw Failure{Status::Corrupt, "file is damaged (segment header)"}; // Loop iteration
               u8 c = payload[hdr++]; alen |= (u64)(c & 127) << shift; if (!(c & 128)) break; shift += 7; } // Compute and assign value
    if (alen > plen - hdr) throw Failure{Status::Corrupt, "file is damaged (segment size)"}; // Conditional branch check
    Dec d(payload + hdr, (size_t)alen); // Execute statement
    const u8* side = payload + hdr + alen; size_t sideLeft = plen - hdr - (size_t)alen; // Compute and assign value
    std::unique_ptr<Model> localModel; // Execute statement
    if (reuseModel) reuseModel->reset(L, tbMax, hint); // Conditional branch check
    else localModel.reset(new Model(L, tbMax, hint)); // Fallback branch
    Model& m = reuseModel ? *reuseModel : *localModel; // Compute and assign value
    u32 fcRaw[3] = {kInit, kInit, kInit}, fcSkip[2] = {kInit, kInit}; // Compute and assign value
    int prevType = 0; size_t off = 0; // Compute and assign value
    while (off < L) { // Loop while condition holds
        if (cn && cn->cancel.load(std::memory_order_relaxed)) throw Cancelled(); // Conditional branch check
        size_t B = std::min(kBlock, L - off); // Compute and assign value
        int type = d.bit(clampi((int)(fcRaw[prevType] >> 20), 16, 4080)); sm_up(&fcRaw[prevType], type, 30); // Compute and assign value
        if (type) { int c = prevType == 2; int sk = d.bit(clampi((int)(fcSkip[c] >> 20), 16, 4080)); sm_up(&fcSkip[c], sk, 30); type = sk ? 2 : 1; } // Conditional branch check
        prevType = type; // Compute and assign value
        u8* o = out + off; // Compute and assign value
        if (type == 2) { // Conditional branch check
            if (sideLeft < B) throw Failure{Status::Corrupt, "file is damaged (missing data)"}; // Conditional branch check
            std::memcpy(o, side, B); side += B; sideLeft -= B; // Fast raw memory copy
        } else for (size_t i = 0; i < B; i++) { // Compute and assign value
            int v = 0; // Compute and assign value
            for (int b = 0; b < 8; b++) { int p = m.predict(), y = type ? d.half() : d.bit(p); m.update(y); v = (v << 1) | y; } // Loop iteration
            o[i] = (u8)v; // Compute and assign value
        } // Close scope / block
        off += B; // Compute and assign value
        if (cn) { cn->done.fetch_add(B, std::memory_order_relaxed); cn->out.fetch_add(B, std::memory_order_relaxed); } // Conditional branch check
    } // Close scope / block
} // Close scope / block

// ------------------------------------------------------------------ helpers
struct Crc { // Declare data structure / class
    u32 t[8][256]; // Execute statement
    Crc() { // Execute statement
        for (u32 i = 0; i < 256; i++) { u32 c = i; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; t[0][i] = c; } // Loop iteration
        for (u32 i = 0; i < 256; i++) for (int k = 1; k < 8; k++) t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 255]; // Loop iteration
    } // Close scope / block
}; // Close scope / block
inline const Crc g_crc; // Define helper function or constant
inline u32 crc32_update(u32 prev, const u8* p, size_t n) { // Update / verify CRC-32 integrity checksum
    const Crc& C = g_crc; u32 r = ~prev; // Compute and assign value
    while (n >= 8) { // Loop while condition holds
        u32 a = r ^ ((u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24)); // Compute and assign value
        r = C.t[7][a & 255] ^ C.t[6][(a >> 8) & 255] ^ C.t[5][(a >> 16) & 255] ^ C.t[4][a >> 24] ^ // Compute and assign value
            C.t[3][p[4]] ^ C.t[2][p[5]] ^ C.t[1][p[6]] ^ C.t[0][p[7]]; // Execute statement
        p += 8; n -= 8; // Compute and assign value
    } // Close scope / block
    while (n--) r = C.t[0][(r ^ *p++) & 255] ^ (r >> 8); // Loop while condition holds
    return ~r; // Return result to caller
} // Close scope / block
inline u32 crc32(const u8* p, size_t n) { return crc32_update(0, p, n); } // Update / verify CRC-32 integrity checksum

inline FILE* fopen_path(const fs::path& p, const char* mode) { // Define helper function or constant
#ifdef _WIN32 // Guard if macro is defined
    std::wstring m(mode, mode + std::strlen(mode)); return _wfopen(p.c_str(), m.c_str()); // Execute statement
#else // Fallback conditional compilation branch
    return std::fopen(p.c_str(), mode); // Return result to caller
#endif // End conditional compilation guard
} // Close scope / block
inline int seek64(FILE* f, u64 off) { // Define helper function or constant
#ifdef _WIN32 // Guard if macro is defined
    return _fseeki64(f, (__int64)off, SEEK_SET); // Return result to caller
#else // Fallback conditional compilation branch
    return fseeko(f, (off_t)off, SEEK_SET); // Return result to caller
#endif // End conditional compilation guard
} // Close scope / block
struct File { // Declare data structure / class
    FILE* f = nullptr; File() {} File(const File&) = delete; File& operator=(const File&) = delete; // Compute and assign value
    bool open(const fs::path& p, const char* mode) { close(); f = fopen_path(p, mode); return f != nullptr; } // Compute and assign value
    void close() { if (f) { std::fclose(f); f = nullptr; } } ~File() { close(); } // Compute and assign value
}; // Close scope / block
inline std::string to_utf8(const fs::path& p) { auto s = p.u8string(); return std::string(s.begin(), s.end()); } // Define helper function or constant
inline fs::path from_utf8(const std::string& s) { // Define helper function or constant
#if defined(__cpp_char8_t) // Conditional compilation check
    return fs::path(std::u8string(s.begin(), s.end())); // Return result to caller
#else // Fallback conditional compilation branch
    return fs::u8path(s); // Return result to caller
#endif // End conditional compilation guard
} // Close scope / block
inline std::string safe_name(std::string n) {            // never trust a name that came out of a file
    size_t cut = n.find_last_of("/\\"); if (cut != std::string::npos) n = n.substr(cut + 1); // Compute and assign value
    for (char& c : n) if ((unsigned char)c < 32 || std::strchr("<>:\"|?*", c)) c = '_'; // Loop iteration
    while (!n.empty() && (n.back() == '.' || n.back() == ' ')) n.pop_back(); // Loop while condition holds
    if (n.empty()) n = "restored_file"; // Conditional branch check
    std::string up; for (char c : n.substr(0, n.find('.'))) up += (char)std::toupper((unsigned char)c); // Compute and assign value
    static const char* bad[] = {"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9","LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"}; // Define helper function or constant
    for (const char* b : bad) if (up == b) { n = "_" + n; break; } // Loop iteration
    if (n.size() > 200) { size_t dot = n.rfind('.'); std::string ext = (dot != std::string::npos && n.size() - dot < 20) ? n.substr(dot) : ""; n = n.substr(0, 200 - ext.size()) + ext; } // Conditional branch check
    return n; // Return result to caller
} // Close scope / block
inline fs::path unique_path(const fs::path& dir, const std::string& name) { // Define helper function or constant
    fs::path base = from_utf8(name), p = dir / base; std::error_code ec; // Compute and assign value
    if (!fs::exists(p, ec)) return p; // Conditional branch check
    std::string stem = to_utf8(base.stem()), ext = to_utf8(base.extension()); // Compute and assign value
    for (int i = 1; i < 100000; i++) { p = dir / from_utf8(stem + " (" + std::to_string(i) + ")" + ext); if (!fs::exists(p, ec)) return p; } // Loop iteration
    return dir / base; // Return result to caller
} // Close scope / block
inline void put_varint(std::vector<u8>& v, u64 n) { while (n >= 128) { v.push_back((u8)(n | 128)); n >>= 7; } v.push_back((u8)n); } // Allocate / manage dynamic buffer
inline bool get_varint(FILE* f, u64& n, u64& used) { // Encode / decode variable-length integer
    n = 0; int shift = 0; // Compute and assign value
    for (;;) { int c = std::fgetc(f); if (c == EOF || shift > 63) return false; used++; n |= (u64)(c & 127) << shift; if (!(c & 128)) return true; shift += 7; } // Loop iteration
} // Close scope / block
struct Match { u64 gap = 0, len = 0, dist = 0; };   // gap = literal bytes since the previous match; copy len bytes from dist back
inline size_t seg_size_from_code(int code) { // Define helper function or constant
    static const size_t tbl[8] = { // Define helper function or constant
        (size_t)8 << 20, (size_t)4 << 20, (size_t)2 << 20, // Execute statement
        (size_t)1 << 20, (size_t)512 << 10, (size_t)256 << 10, // Execute statement
        (size_t)16 << 20, (size_t)32 << 20 // Execute statement
    }; // Close scope / block
    return tbl[(code >= 0 && code < 8) ? code : 0]; // Return result to caller
} // Close scope / block
inline int choose_seg_code(u64 streamLen, unsigned wantThreads, bool fast, double sampleH0 = 5.0, bool isDedup = false) { // Define helper function or constant
    if (wantThreads <= 1 || streamLen < ((u64)512 << 10)) return 0; // Conditional branch check
    if (!fast) { // Conditional branch check
        size_t minSeg = (size_t)256 << 10; // Compute and assign value
        u64 target = streamLen / ((u64)wantThreads * 4u); // ~4 waves per core to eliminate PAQ stragglers
        if (target < minSeg) target = minSeg; // Conditional branch check
        for (int c = 5; c >= 0; c--) { // Loop iteration
            if (seg_size_from_code(c) >= target) return c; // Conditional branch check
        } // Close scope / block
        return 0; // Return result to caller
    } // Close scope / block
    if (isDedup) { // Conditional branch check
        if (streamLen < ((u64)4 << 20)) return 3;          // 1 MB
        if (streamLen <= ((u64)48 << 20)) return 1;        // 4 MB (optimal ratio + multi-core balance)
        if (streamLen <= ((u64)128 << 20)) return 0;       // 8 MB
        return 6;                                          // 16 MB
    } // Close scope / block
    // Fast mode entropy-adaptive segment sizing:
    //   H0 < 3.5  : 4 waves/core (256 KB..2 MB) for max L2/L3 speed
    //   3.5..6.5  : 2 waves/core (512 KB..8 MB) to cut rANS table tax
    //   H0 >= 6.5 : 1 wave/core  (512 KB..8 MB, 16 MB on >=128 MB) for near-Shannon ratio
    u64 waves = (sampleH0 < 3.5) ? 4u : ((sampleH0 < 6.5) ? 2u : 1u); // Compute and assign value
    size_t minSeg = (sampleH0 < 3.5) ? ((size_t)256 << 10) : ((size_t)512 << 10); // Compute and assign value
    size_t maxSeg = (sampleH0 < 3.5) ? ((size_t)2 << 20) : ((size_t)8 << 20); // Compute and assign value
    u64 target = streamLen / ((u64)wantThreads * waves); // Compute and assign value
    if (target < minSeg) target = minSeg; // Conditional branch check
    if (sampleH0 >= 6.5 && streamLen >= ((u64)128 << 20) && target >= ((u64)12 << 20)) return 6; // 16 MB
    if (target > maxSeg) target = maxSeg; // Conditional branch check
    for (int c = 5; c >= 0; c--) { // Loop iteration
        if (seg_size_from_code(c) >= target) return c; // Conditional branch check
    } // Close scope / block
    return 0; // Return result to caller
} // Close scope / block
// Reversible Byte-Plane Shuffle + Plane-Wise Delta (v18.9 auto-detected stage)
// Separates interleaved multi-byte records (int16/float32/int64) into contiguous byte planes
// and applies mod-256 delta per plane. Tail bytes (L % stride) remain untouched in place.
inline void byte_plane_shuffle_delta(u8* p, size_t L, int stride, bool forward, std::vector<u8>* reuseScratch = nullptr) { // Reversible byte-plane shuffle + delta (zero-churn scratch reuse)
    if (stride != 2 && stride != 4 && stride != 8) return; // Validate supported record stride
    const size_t st = (size_t)stride; // Unsigned stride width
    const size_t count = L / st; // Number of complete multi-byte records
    if (count < 2) return; // Need at least 2 records for delta coding
    const size_t body = count * st; // Total aligned bytes to transform
    std::vector<u8> localTmp; // Fallback scratch if no worker buffer supplied
    std::vector<u8>& tmp = reuseScratch ? *reuseScratch : localTmp; // Reuse per-worker scratch to avoid heap zigzag
    if (tmp.size() < body) tmp.resize(body); // Grow once per worker
    if (forward) { // Forward transform: transpose into byte planes + plane-wise delta
        for (size_t k = 0; k < st; k++) { // Iterate over each byte lane
            u8* dst = tmp.data() + k * count; // Destination plane start
            u8 prev = p[k]; // First sample in lane k
            dst[0] = prev; // Store raw anchor byte
            for (size_t i = 1; i < count; i++) { // Delta-encode remaining lane bytes
                u8 cur = p[i * st + k]; // Read interleaved byte
                dst[i] = (u8)(cur - prev); // Store mod-256 delta
                prev = cur; // Update running predictor
            } // End lane loop
        } // End plane loop
        std::memcpy(p, tmp.data(), body); // Copy transformed planes back in place
    } else { // Inverse transform: undo plane-wise delta + untranspose to interleaved records
        for (size_t k = 0; k < st; k++) { // Iterate over each byte lane
            const u8* sp = p + k * count; // Source plane start
            u8 acc = sp[0]; // Running prefix sum accumulator
            tmp[k] = acc; // Restore first interleaved byte
            for (size_t i = 1; i < count; i++) { // Reconstruct remaining lane bytes
                acc = (u8)(acc + sp[i]); // Accumulate mod-256 delta
                tmp[i * st + k] = acc; // Write back to interleaved slot
            } // End lane loop
        } // End plane loop
        std::memcpy(p, tmp.data(), body); // Copy restored records back in place
    } // End forward/inverse branch
} // End byte_plane_shuffle_delta()

struct Header { std::string name; u64 total = 0; int tbMax = 22, segCode = 0, stride = 0, bpp = 0; Audio audio; Grid grid; int exec = 0, bshuf = 0; u64 bytes = 0; // Declare data structure / class
                bool dedup = false; u32 fileCrc = 0; u64 litTotal = 0; std::vector<Match> matches; // Allocate / manage dynamic buffer
                bool jpeg = false; std::vector<u8> jpegPayload; // Allocate / manage dynamic buffer
                bool gen = false; std::vector<u8> genPrefix, genTail; // Allocate / manage dynamic buffer
                bool defl = false; int deflFlag = 0, dStride = 0, dBpp = 0, dPix = 0; u64 dPhase = 0; u64 xTotal = 0, deflCount = 0, deflRaw = 0; std::vector<u8> deflTbl; };   // defl: deflate streams replaced by their inflated data          // gen: whole file = Mersenne Twister output
inline std::vector<u8> make_header(const Header& h) { // Allocate / manage dynamic buffer
    std::vector<u8> v; v.push_back(kMagic); // Allocate / manage dynamic buffer
    put_varint(v, h.name.size()); v.insert(v.end(), h.name.begin(), h.name.end()); // Encode / decode variable-length integer
    put_varint(v, h.total); v.push_back((u8)((h.tbMax & 31) | ((h.segCode & 7) << 5))); // Append element to container
    if (h.audio.on()) { v.push_back(2); put_varint(v, h.audio.dataOff); put_varint(v, h.audio.dataLen); v.push_back((u8)h.audio.channels); v.push_back((u8)h.audio.width); } // Conditional branch check
    else if (h.stride) { v.push_back(1); put_varint(v, (u64)h.stride); v.push_back((u8)h.bpp); } // Alternative conditional branch
    else if (h.dedup) { // Alternative conditional branch
        if (!h.exec && h.matches.size() >= 64) { // Conditional branch check
            std::vector<u8> gBuf, lBuf, dBuf; // Allocate / manage dynamic buffer
            gBuf.reserve(h.matches.size() * 2); lBuf.reserve(h.matches.size() * 2); dBuf.reserve(h.matches.size() * 3); // Execute statement
            u64 dhist[4] = {0, 0, 0, 0}; // Compute and assign value
            for (const Match& m : h.matches) { // Loop iteration
                put_varint(gBuf, m.gap); // Encode / decode variable-length integer
                put_varint(lBuf, m.len); // Encode / decode variable-length integer
                int hit = -1; // Compute and assign value
                for (int r = 0; r < 4; r++) if (dhist[r] == m.dist) { hit = r; break; } // Loop iteration
                if (hit >= 0) { // Conditional branch check
                    put_varint(dBuf, (u64)hit); // Encode / decode variable-length integer
                    u64 d = dhist[hit]; // Compute and assign value
                    for (int r = hit; r > 0; r--) dhist[r] = dhist[r - 1]; // Loop iteration
                    dhist[0] = d; // Compute and assign value
                } else { // Execute statement
                    put_varint(dBuf, m.dist + 4u); // Encode / decode variable-length integer
                    dhist[3] = dhist[2]; dhist[2] = dhist[1]; dhist[1] = dhist[0]; dhist[0] = m.dist; // Compute and assign value
                } // Close scope / block
            } // Close scope / block
            std::vector<u8> cG, cL, cD; // Allocate / manage dynamic buffer
            fast::rans_encode_stream(gBuf.data(), gBuf.size(), cG); // rANS entropy coding step
            fast::rans_encode_stream(lBuf.data(), lBuf.size(), cL); // rANS entropy coding step
            fast::rans_encode_stream(dBuf.data(), dBuf.size(), cD); // rANS entropy coding step
            v.push_back(10); // 10 = split rANS + 4-LRU MTF distance-coded copy records
            put_varint(v, h.matches.size()); for (int i = 0; i < 4; i++) v.push_back((u8)(h.fileCrc >> (8 * i))); // Append element to container
            put_varint(v, cG.size()); v.insert(v.end(), cG.begin(), cG.end()); // Encode / decode variable-length integer
            put_varint(v, cL.size()); v.insert(v.end(), cL.begin(), cL.end()); // Encode / decode variable-length integer
            put_varint(v, cD.size()); v.insert(v.end(), cD.begin(), cD.end()); // Encode / decode variable-length integer
        } else { // Execute statement
            if (h.exec) { v.push_back(7); v.push_back((u8)h.exec); } else v.push_back(3);      // 7 = copy records + executable filter
            put_varint(v, h.matches.size()); for (int i = 0; i < 4; i++) v.push_back((u8)(h.fileCrc >> (8 * i))); // Append element to container
            for (const Match& m : h.matches) { put_varint(v, m.gap); put_varint(v, m.len); put_varint(v, m.dist); } // Loop iteration
        } // Close scope / block
    } // Close scope / block
    else if (h.exec) { v.push_back(6); v.push_back((u8)h.exec); } // Alternative conditional branch
    else if (h.bshuf) { v.push_back(11); v.push_back((u8)h.bshuf); } // Byte-plane shuffle + delta stride
    else if (h.grid.on()) { v.push_back(4); put_varint(v, (u64)h.grid.stride); v.push_back((u8)h.grid.width); v.push_back((u8)(h.grid.mode | (h.grid.zz << 1))); } // Alternative conditional branch
    else if (h.jpeg) { v.push_back(8); for (int i = 0; i < 4; i++) v.push_back((u8)(h.fileCrc >> (8 * i))); put_varint(v, h.jpegPayload.size()); v.insert(v.end(), h.jpegPayload.begin(), h.jpegPayload.end()); } // Alternative conditional branch
    else if (h.defl) { v.push_back(9); for (int i = 0; i < 4; i++) v.push_back((u8)(h.fileCrc >> (8 * i))); put_varint(v, h.xTotal); put_varint(v, h.deflCount); put_varint(v, h.deflRaw); v.push_back((u8)h.deflFlag); put_varint(v, (u64)h.dStride); v.push_back((u8)h.dBpp); v.push_back((u8)h.dPix); put_varint(v, h.dPhase); put_varint(v, h.deflTbl.size()); v.insert(v.end(), h.deflTbl.begin(), h.deflTbl.end()); } // Alternative conditional branch
    else if (h.gen) { v.push_back(5); v.insert(v.end(), h.genPrefix.begin(), h.genPrefix.end()); for (int i = 0; i < 4; i++) v.push_back((u8)(h.fileCrc >> (8 * i))); v.push_back((u8)h.genTail.size()); v.insert(v.end(), h.genTail.begin(), h.genTail.end()); } // Alternative conditional branch
    else v.push_back(0); // Fallback branch
    return v; // Return result to caller
} // Close scope / block
inline bool read_header(FILE* f, Header& h, u64 fileSize = ~(u64)0) { // Security-bounded .3p header parser
    u64 used = 0, n = 0; // Compute and assign value
    if (std::fgetc(f) != kMagic) return false; // Conditional branch check
    used = 1; // Compute and assign value
    if (!get_varint(f, n, used) || n == 0 || n > kMaxName) return false; // Conditional branch check
    h.name.resize((size_t)n); if (std::fread(&h.name[0], 1, (size_t)n, f) != (size_t)n) return false; used += n; // Buffered file I/O operation
    if (!get_varint(f, h.total, used)) return false; // Conditional branch check
    int tbRaw = std::fgetc(f); if (tbRaw < 0) return false; int sc = (tbRaw >> 5) & 7; int tb = tbRaw & 31; if (sc > 7 || (tb != 1 && (tb < 16 || tb > 22))) return false; h.tbMax = tb; h.segCode = sc; used++; // Segment sizing configuration
    int kind = std::fgetc(f); used++; // Compute and assign value
    if (kind == 7) { int e = std::fgetc(f); used++; if (e < 1 || e > 3) return false; h.exec = e; kind = 3; } // Conditional branch check
    if (kind == 1) { // Conditional branch check
        u64 st = 0; if (!get_varint(f, st, used) || st == 0 || st > 0x7fffffff) return false; h.stride = (int)st; // Encode / decode variable-length integer
        int b = std::fgetc(f); if (b < 1 || b > 8) return false; h.bpp = b; used++; // Compute and assign value
    } else if (kind == 2) { // Compute and assign value
        u64 off = 0, len = 0; if (!get_varint(f, off, used) || !get_varint(f, len, used)) return false; // Encode / decode variable-length integer
        int ch = std::fgetc(f), w = std::fgetc(f); used += 2; // Compute and assign value
        if (ch < 1 || ch > 8 || w < 1 || w > 3 || off > h.total || len > h.total - off) return false; // Conditional branch check
        h.audio.dataOff = off; h.audio.dataLen = len; h.audio.channels = ch; h.audio.width = w; // Compute and assign value
    } else if (kind == 10) { // Compute and assign value
        u64 cnt = 0; if (!get_varint(f, cnt, used) || cnt > h.total / 16 + 1 || cnt > ((u64)16 << 20)) return false; // Cap match count to prevent OOM
        u8 c4[4]; if (std::fread(c4, 1, 4, f) != 4) return false; used += 4; // Buffered file I/O operation
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        auto read_rans_vec = [&](std::vector<u8>& dst) -> bool { // Allocate / manage dynamic buffer
            u64 clen = 0; if (!get_varint(f, clen, used) || clen > ((u64)64 << 20) || clen > fileSize) return false; // Bound compressed header stream to fileSize
            std::vector<u8> cb((size_t)clen); // Allocate / manage dynamic buffer
            if (clen && std::fread(cb.data(), 1, (size_t)clen, f) != (size_t)clen) return false; // Conditional branch check
            used += clen; // Compute and assign value
            try { const u8* p = cb.data(); fast::rans_decode_stream(p, p + cb.size(), dst); } catch (...) { return false; } // Begin exception-guarded block
            return true; // Return result to caller
        }; // Close scope / block
        std::vector<u8> gBuf, lBuf, dBuf; // Allocate / manage dynamic buffer
        if (!read_rans_vec(gBuf) || !read_rans_vec(lBuf) || !read_rans_vec(dBuf)) return false; // Conditional branch check
        auto get_vi_mem = [](const u8*& p, const u8* end, u64& v) -> bool { // Compute and assign value
            v = 0; int sh = 0; // Compute and assign value
            for (;;) { if (p >= end || sh > 56) return false; u8 b = *p++; v |= (u64)(b & 127) << sh; if (!(b & 128)) return true; sh += 7; } // Loop iteration
        }; // Close scope / block
        const u8 *gp = gBuf.data(), *ge = gp + gBuf.size(); // Compute and assign value
        const u8 *lp = lBuf.data(), *le = lp + lBuf.size(); // Compute and assign value
        const u8 *dp = dBuf.data(), *de = dp + dBuf.size(); // Compute and assign value
        u64 dhist[4] = {0, 0, 0, 0}; // Compute and assign value
        h.matches.reserve((size_t)std::min<u64>(cnt, (fileSize / 3) + 1)); u64 cursor = 0, matched = 0; // Safe bounded reserve
        for (u64 i = 0; i < cnt; i++) { // Loop iteration
            Match m; u64 dcode = 0; // Compute and assign value
            if (!get_vi_mem(gp, ge, m.gap) || !get_vi_mem(lp, le, m.len) || !get_vi_mem(dp, de, dcode)) return false; // Conditional branch check
            if (dcode < 4u) { // Conditional branch check
                int hit = (int)dcode; m.dist = dhist[hit]; // Compute and assign value
                for (int r = hit; r > 0; r--) dhist[r] = dhist[r - 1]; // Loop iteration
                dhist[0] = m.dist; // Compute and assign value
            } else { // Execute statement
                m.dist = dcode - 4u; // Compute and assign value
                dhist[3] = dhist[2]; dhist[2] = dhist[1]; dhist[1] = dhist[0]; dhist[0] = m.dist; // Compute and assign value
            } // Close scope / block
            if (m.len == 0 || m.dist == 0 || m.gap > h.total || m.len > h.total) return false; // Conditional branch check
            cursor += m.gap; if (cursor > h.total || m.dist > cursor) return false; // Compute and assign value
            cursor += m.len; if (cursor > h.total) return false; // Compute and assign value
            matched += m.len; h.matches.push_back(m); // Append element to container
        } // Close scope / block
        h.dedup = true; h.litTotal = h.total - matched; // Compute and assign value
    } else if (kind == 9) { // Compute and assign value
        u64 cnt = 0; if (!get_varint(f, cnt, used) || cnt > h.total / 16 + 1 || cnt > ((u64)16 << 20)) return false; // Cap match count to prevent OOM
        u8 c4[4]; if (std::fread(c4, 1, 4, f) != 4) return false; used += 4; // Buffered file I/O operation
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        u64 clen = 0; if (!get_varint(f, clen, used) || clen > ((u64)64 << 20) || clen > fileSize) return false; // Bound compressed match stream to fileSize
        std::vector<u8> compM((size_t)clen); // Allocate / manage dynamic buffer
        if (clen && std::fread(compM.data(), 1, (size_t)clen, f) != (size_t)clen) return false; // Conditional branch check
        used += clen; // Compute and assign value
        std::vector<u8> rawM; // Allocate / manage dynamic buffer
        try { // Begin exception-guarded block
            const u8* p = compM.data(); const u8* end = p + compM.size(); // Compute and assign value
            fast::rans_decode_stream(p, end, rawM); // rANS entropy coding step
        } catch (...) { return false; } // Execute statement
        auto get_vi_mem = [](const u8*& p, const u8* end, u64& v) -> bool { // Compute and assign value
            v = 0; int sh = 0; // Compute and assign value
            for (;;) { if (p >= end || sh > 56) return false; u8 b = *p++; v |= (u64)(b & 127) << sh; if (!(b & 128)) return true; sh += 7; } // Loop iteration
        }; // Close scope / block
        const u8* rp = rawM.data(); const u8* rend = rp + rawM.size(); // Compute and assign value
        h.matches.reserve((size_t)std::min<u64>(cnt, (fileSize / 3) + 1)); u64 cursor = 0, matched = 0; // Safe bounded reserve
        for (u64 i = 0; i < cnt; i++) { // Loop iteration
            Match m; if (!get_vi_mem(rp, rend, m.gap) || !get_vi_mem(rp, rend, m.len) || !get_vi_mem(rp, rend, m.dist)) return false; // Execute statement
            if (m.len == 0 || m.dist == 0 || m.gap > h.total || m.len > h.total) return false; // Conditional branch check
            cursor += m.gap; if (cursor > h.total || m.dist > cursor) return false; // Compute and assign value
            cursor += m.len; if (cursor > h.total) return false; // Compute and assign value
            matched += m.len; h.matches.push_back(m); // Append element to container
        } // Close scope / block
        h.dedup = true; h.litTotal = h.total - matched; // Compute and assign value
    } else if (kind == 3) { // Compute and assign value
        u64 cnt = 0; if (!get_varint(f, cnt, used) || cnt > h.total / 16 + 1 || cnt > ((u64)16 << 20)) return false; // Cap match count to prevent OOM
        u8 c4[4]; if (std::fread(c4, 1, 4, f) != 4) return false; used += 4; // Buffered file I/O operation
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        h.matches.reserve((size_t)std::min<u64>(cnt, (fileSize / 3) + 1)); u64 cursor = 0, matched = 0; // Safe bounded reserve
        for (u64 i = 0; i < cnt; i++) { // Loop iteration
            Match m; if (!get_varint(f, m.gap, used) || !get_varint(f, m.len, used) || !get_varint(f, m.dist, used)) return false; // Encode / decode variable-length integer
            if (m.len == 0 || m.dist == 0 || m.gap > h.total || m.len > h.total) return false; // Conditional branch check
            cursor += m.gap; if (cursor > h.total || m.dist > cursor) return false; // Compute and assign value
            cursor += m.len; if (cursor > h.total) return false; // Compute and assign value
            matched += m.len; h.matches.push_back(m); // Append element to container
        } // Close scope / block
        h.dedup = true; h.litTotal = h.total - matched; // Compute and assign value
    } else if (kind == 4) { // Compute and assign value
        u64 st = 0; if (!get_varint(f, st, used) || st == 0 || st > ((u64)1 << 26)) return false; // Encode / decode variable-length integer
        int w = std::fgetc(f), fl = std::fgetc(f); used += 2; // Compute and assign value
        if ((w != 1 && w != 2 && w != 4) || fl < 0 || fl > 3 || st % (u64)w) return false; // Conditional branch check
        h.grid.stride = (int)st; h.grid.width = w; h.grid.mode = fl & 1; h.grid.zz = (fl >> 1) & 1; // Compute and assign value
    } else if (kind == 6) { // Compute and assign value
        int e = std::fgetc(f); used++; if (e < 1 || e > 3) return false; h.exec = e; // Compute and assign value
    } else if (kind == 11) { // Byte-plane shuffle + delta header tag
        int bs = std::fgetc(f); used++; if (bs != 2 && bs != 4 && bs != 8) return false; h.bshuf = bs; // Validate and store stride
    } else if (kind == 8) { // Compute and assign value
        u8 c4[4]; u64 pl = 0; if (std::fread(c4, 1, 4, f) != 4) return false; used += 4; // Buffered file I/O operation
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        if (!get_varint(f, pl, used) || pl > h.total + ((u64)1 << 22) || pl > fileSize) return false; // Bound JPEG payload to actual fileSize
        h.jpegPayload.resize((size_t)pl); if (pl && std::fread(h.jpegPayload.data(), 1, (size_t)pl, f) != (size_t)pl) return false; // Buffered file I/O operation
        used += pl; h.jpeg = true; // Compute and assign value
    } else if (kind == 9) { // Compute and assign value
        u8 c4[4]; if (std::fread(c4, 1, 4, f) != 4) return false; used += 4; // Buffered file I/O operation
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        u64 xt = 0, cnt = 0, rw = 0, pl = 0; // Compute and assign value
        if (!get_varint(f, xt, used) || !get_varint(f, cnt, used) || !get_varint(f, rw, used)) return false; // Conditional branch check
        int fl = std::fgetc(f); used++; if (fl != 0 && fl != 1) return false; // Compute and assign value
        u64 ds = 0; if (!get_varint(f, ds, used) || ds > ((u64)1 << 26)) return false; // Encode / decode variable-length integer
        int db = std::fgetc(f); used++; if (db < 0 || db > 8 || (ds && db < 1)) return false; // Compute and assign value
        int dp = std::fgetc(f); used++; u64 dph = 0; if (dp != 0 && dp != 1) return false; // Compute and assign value
        if (!get_varint(f, dph, used) || dph > ((u64)3 << 30)) return false; // Conditional branch check
        h.dStride = (int)ds; h.dBpp = db; h.dPix = dp; h.dPhase = dph; // Compute and assign value
        if (!get_varint(f, pl, used)) return false; // Conditional branch check
        if (cnt == 0 || cnt > 200000 || rw > kSegment || pl > kSegment + 65536 || pl > fileSize || xt > h.total * 1100 + 4096 || xt > ((u64)3 << 30)) return false; // Conditional branch check
        h.deflTbl.resize((size_t)pl); if (pl && std::fread(h.deflTbl.data(), 1, (size_t)pl, f) != (size_t)pl) return false; // Buffered file I/O operation
        used += pl; h.defl = true; h.xTotal = xt; h.deflCount = cnt; h.deflRaw = rw; h.deflFlag = fl; // Compute and assign value
    } else if (kind == 5) { // Compute and assign value
        h.genPrefix.resize(624 * 4); u8 c4[4]; // Execute statement
        if (std::fread(h.genPrefix.data(), 1, h.genPrefix.size(), f) != h.genPrefix.size() || std::fread(c4, 1, 4, f) != 4) return false; // Conditional branch check
        h.fileCrc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
        int tl = std::fgetc(f); if (tl < 0 || tl > 3) return false; // Compute and assign value
        h.genTail.resize((size_t)tl); if (tl && std::fread(h.genTail.data(), 1, (size_t)tl, f) != (size_t)tl) return false; // Buffered file I/O operation
        used += h.genPrefix.size() + 4 + 1 + (u64)tl; // Compute and assign value
        if (h.total < h.genPrefix.size() + (u64)tl || (h.total - (u64)tl) % 4) return false; // Conditional branch check
        h.gen = true; // Compute and assign value
    } else if (kind != 0) return false; // Compute and assign value
    h.bytes = used; return true; // Compute and assign value
} // Close scope / block
inline void detect_geometry(const u8* b, size_t n, int& stride, int& bpp) {      // uncompressed BMP: 8/24/32 bit
    stride = bpp = 0; // Compute and assign value
    if (n < 30 || b[0] != 'B' || b[1] != 'M') return; // Conditional branch check
    u32 w = (u32)b[18] | ((u32)b[19] << 8) | ((u32)b[20] << 16); int bits = b[28]; // Compute and assign value
    if ((bits == 8 || bits == 24 || bits == 32) && b[29] == 0 && w > 0 && w < 100000) { bpp = bits / 8; stride = (int)((((u64)w * (u64)bits + 31) / 32) * 4); } // Conditional branch check
} // Close scope / block
// PCM WAV (RIFF/WAVE, 8/16/24-bit, up to 8 channels): finds the "data" chunk
inline bool detect_wav(FILE* f, u64 fileSize, Audio& a) { // Define helper function or constant
    u8 h[12]; if (seek64(f, 0) != 0 || std::fread(h, 1, 12, f) != 12 || std::memcmp(h, "RIFF", 4) || std::memcmp(h + 8, "WAVE", 4)) return false; // Buffered file I/O operation
    u64 pos = 12; int channels = 0, bits = 0, blockAlign = 0; bool pcm = false; // Compute and assign value
    for (int guard = 0; guard < 64 && pos + 8 <= fileSize; guard++) { // Loop iteration
        u8 c[8]; if (seek64(f, pos) != 0 || std::fread(c, 1, 8, f) != 8) return false; // Buffered file I/O operation
        u64 size = (u64)c[4] | ((u64)c[5] << 8) | ((u64)c[6] << 16) | ((u64)c[7] << 24); // Compute and assign value
        if (!std::memcmp(c, "fmt ", 4) && size >= 16) { // Conditional branch check
            u8 b[40]; size_t want = size >= 40 ? 40 : 16; if (std::fread(b, 1, want, f) != want) return false; // Buffered file I/O operation
            int tag = b[0] | (b[1] << 8); channels = b[2] | (b[3] << 8); blockAlign = b[12] | (b[13] << 8); bits = b[14] | (b[15] << 8); // Compute and assign value
            if (tag == 0xFFFE && want == 40) tag = b[24] | (b[25] << 8);                 // WAVE_FORMAT_EXTENSIBLE sub-format
            pcm = (tag == 1); // Compute and assign value
        } else if (!std::memcmp(c, "data", 4)) { // Execute statement
            if (!pcm || channels < 1 || channels > 8 || (bits != 8 && bits != 16 && bits != 24) || blockAlign != channels * bits / 8) return false; // Conditional branch check
            u64 off = pos + 8, avail = fileSize > off ? fileSize - off : 0; // Compute and assign value
            a.dataOff = off; a.dataLen = (size == 0 || size > avail) ? avail : size; a.channels = channels; a.width = bits / 8; // Compute and assign value
            return a.dataLen >= (u64)a.frame(); // Return result to caller
        } // Close scope / block
        pos += 8 + size + (size & 1); // Compute and assign value
    } // Close scope / block
    return false; // Return result to caller
} // Close scope / block
// ------------------------------------------------------------------ long-distance duplicate finder
// Rolling hash over a 32-byte window; content-defined anchors (1 in 2^k positions) are remembered in a hash table.
// A hit is verified byte by byte, then extended backwards and forwards. The table is bounded (256 MB), so any
// file size works; the file is read in 16 MB steps and verified through two small block caches.
class FileCache { // Declare data structure / class
public: // Member access specifier
    FileCache(const fs::path& p, u64 size) : size_(size), blk_(kBlk) { ok_ = f_.open(p, "rb"); } // Compute and assign value
    bool ok() const { return ok_; } // Execute statement
    const u8* get(u64 pos, size_t& avail) { // Execute statement
        if (pos >= size_) { avail = 0; return nullptr; } // Conditional branch check
        if (pos < off_ || pos >= off_ + len_) { // Conditional branch check
            off_ = pos & ~(u64)(kBlk - 1); // Compute and assign value
            if (seek64(f_.f, off_) != 0) { len_ = 0; avail = 0; return nullptr; } // Conditional branch check
            len_ = std::fread(blk_.data(), 1, kBlk, f_.f); // Buffered file I/O operation
            if (pos >= off_ + len_) { avail = 0; return nullptr; } // Conditional branch check
        } // Close scope / block
        avail = (size_t)(off_ + len_ - pos); return blk_.data() + (pos - off_); // Compute and assign value
    } // Close scope / block
    int at(u64 pos) { size_t a; const u8* p = get(pos, a); return p ? (int)*p : -1; } // Compute and assign value
private: // Member access specifier
    static constexpr size_t kBlk = (size_t)1 << 22; // Define helper function or constant
    File f_; bool ok_ = false; u64 size_, off_ = ~(u64)0; size_t len_ = 0; std::vector<u8> blk_; // Allocate / manage dynamic buffer
}; // Close scope / block
inline u64 common_forward(FileCache& a, u64 pa, FileCache& b, u64 pb, u64 maxLen) { // Define helper function or constant
    u64 total = 0; // Compute and assign value
    while (total < maxLen) { // Loop while condition holds
        size_t aa, ab; const u8* x = a.get(pa + total, aa); const u8* y = b.get(pb + total, ab); // Compute and assign value
        if (!x || !y) break; // Conditional branch check
        size_t m = (size_t)std::min<u64>(std::min<u64>(aa, ab), maxLen - total), k = 0; // Compute and assign value
        while (k < m && x[k] == y[k]) k++; // Loop while condition holds
        total += k; if (k < m) break; // Compute and assign value
    } // Close scope / block
    return total; // Return result to caller
} // Close scope / block
// returns false on cancel/read error. cancelled is set when prog returned false.
inline bool find_duplicates(const fs::path& src, u64 n, u64 tableBudget, const std::function<bool(double)>& prog, // Define helper function or constant
                            std::vector<Match>& out, u64& matched, bool& cancelled) { // Allocate / manage dynamic buffer
    cancelled = false; out.clear(); matched = 0; // Compute and assign value
    const int k = std::max(3, ceil_log2(n) - 25); // Compute and assign value
    const u64 minm = std::max<u64>(16, (u64)1 << k); // 16B min match (catches 16-255B screen/video blocks)
    u64 anchors = (n >> k) + 1, cap = (u64)1 << 16; // Compute and assign value
    while (cap < anchors + anchors / 4 && cap < ((u64)1 << 26)) cap <<= 1; // Loop while condition holds
    while (cap * 12 > tableBudget && cap > ((u64)1 << 16)) cap >>= 1; // Loop while condition holds
    std::vector<u64> tpos((size_t)cap, 0); std::vector<u32> tchk((size_t)cap, 0); // Allocate / manage dynamic buffer
    const u64 B = 0x100000001B3ull; u64 Bpow = 1; for (int i = 0; i < 32; i++) Bpow *= B; // Compute and assign value
    const int shift = 64 - k; // Compute and assign value

    // Fast in-RAM path for files <= 256 MB: zero random FileCache seeks + 8-byte word matching
    if (n <= ((u64)256 << 20)) { // Conditional branch check
        File in; if (!in.open(src, "rb")) return false; // Execute statement
        std::vector<u8> mem((size_t)n); // Allocate / manage dynamic buffer
        size_t got = 0; // Compute and assign value
        while (got < (size_t)n) { // Loop while condition holds
            size_t r = std::fread(mem.data() + got, 1, std::min<size_t>((size_t)16 << 20, (size_t)n - got), in.f); // Buffered file I/O operation
            if (r == 0) return false; // Conditional branch check
            got += r; // Compute and assign value
        } // Close scope / block
        const u8* ptr = mem.data(); // Compute and assign value
        u64 Bpow16 = 1; for (int i = 0; i < 16; i++) Bpow16 *= B; // Compute and assign value
        u64 pos = 0, litStart = 0, h = 0, lastDist = 0; int fill = 0; // Compute and assign value
        u64 nextProg = (u64)16 << 20; // Compute and assign value
        while (pos < n) { // Loop while condition holds
            if (pos >= nextProg) { // Conditional branch check
                if (prog && !prog((double)pos / (double)n)) { cancelled = true; return false; } // Conditional branch check
                nextProg = pos + ((u64)16 << 20); // Compute and assign value
            } // Close scope / block
            // Fast repeat-distance continuation: catches inter-frame duplicates at the same stride without waiting for an anchor
            if (lastDist && pos >= lastDist && pos + 16 <= n) { // Conditional branch check
                u64 a0, b0, a1, b1; // Execute statement
                std::memcpy(&a0, ptr + pos, 8); std::memcpy(&b0, ptr + pos - lastDist, 8); // Fast raw memory copy
                if (a0 == b0) { // Conditional branch check
                    std::memcpy(&a1, ptr + pos + 8, 8); std::memcpy(&b1, ptr + pos - lastDist + 8, 8); // Fast raw memory copy
                    if (a1 == b1) { // Conditional branch check
                        u64 maxf = n - pos, fw = 16; // Compute and assign value
                        while (fw + 8 <= maxf) { // Loop while condition holds
                            u64 ax, bx; std::memcpy(&ax, ptr + pos + fw, 8); std::memcpy(&bx, ptr + pos - lastDist + fw, 8); // Fast raw memory copy
                            if (ax != bx) break; // Conditional branch check
                            fw += 8; // Compute and assign value
                        } // Close scope / block
                        while (fw < maxf && ptr[pos + fw] == ptr[pos - lastDist + fw]) fw++; // Loop while condition holds
                        out.push_back(Match{pos - litStart, fw, lastDist}); // Append element to container
                        matched += fw; litStart = pos + fw; // Compute and assign value
                        pos = litStart; fill = 0; h = 0; // Compute and assign value
                        continue; // Advance to next loop iteration
                    } // Close scope / block
                } // Close scope / block
            } // Close scope / block
            u8 c = ptr[pos]; // Compute and assign value
            if (fill < 16) { h = h * B + c; fill++; pos++; if (fill < 16) continue; } // Conditional branch check
            else { h = h * B + c - (u64)ptr[pos - 16] * Bpow16; pos++; } // Fallback branch
            u64 mix = h * 0x9E3779B97F4A7C15ull; // Compute and assign value
            if ((mix >> shift) != 0) continue; // Conditional branch check
            u64 p = pos; // Compute and assign value
            size_t s0 = (size_t)((mix >> 8) & (cap - 1)); // Compute and assign value
            size_t s1 = s0 ^ 1u; // Compute and assign value
            u32 chk = (u32)((h * 0xC2B2AE3D27D4EB4Full) >> 32) | 1u; // Compute and assign value
            u64 bestLen = 0, bestStart = 0, bestDist = 0; // Compute and assign value
            for (int w = 0; w < 2; w++) { // Loop iteration
                size_t sl = (w == 0) ? s0 : s1; // Compute and assign value
                u64 q = tpos[sl]; // Compute and assign value
                if (q && tchk[sl] == chk) { // Conditional branch check
                    u64 dist = p - q; // Compute and assign value
                    u64 maxb = std::min<u64>(std::min<u64>(p - litStart, q), (u64)1 << 20); // Compute and assign value
                    u64 bk = 0; // Compute and assign value
                    while (bk + 8 <= maxb) { // Loop while condition holds
                        u64 a, b; std::memcpy(&a, ptr + p - 8 - bk, 8); std::memcpy(&b, ptr + q - 8 - bk, 8); // Fast raw memory copy
                        if (a != b) break; // Conditional branch check
                        bk += 8; // Compute and assign value
                    } // Close scope / block
                    while (bk < maxb && ptr[p - 1 - bk] == ptr[q - 1 - bk]) bk++; // Loop while condition holds
                    u64 maxf = n - p, fw = 0; // Compute and assign value
                    while (fw + 8 <= maxf) { // Loop while condition holds
                        u64 a, b; std::memcpy(&a, ptr + p + fw, 8); std::memcpy(&b, ptr + q + fw, 8); // Fast raw memory copy
                        if (a != b) break; // Conditional branch check
                        fw += 8; // Compute and assign value
                    } // Close scope / block
                    while (fw < maxf && ptr[p + fw] == ptr[q + fw]) fw++; // Loop while condition holds
                    u64 len = bk + fw, start = p - bk; // Compute and assign value
                    if (len >= minm && start + len >= p && len > bestLen) { // Conditional branch check
                        bestLen = len; bestStart = start; bestDist = dist; // Compute and assign value
                    } // Close scope / block
                } // Close scope / block
            } // Close scope / block
            if (bestLen >= minm) { // Conditional branch check
                out.push_back(Match{bestStart - litStart, bestLen, bestDist}); // Append element to container
                matched += bestLen; litStart = bestStart + bestLen; // Compute and assign value
                lastDist = bestDist; // Compute and assign value
                pos = litStart; fill = 0; h = 0; // Compute and assign value
                continue; // Advance to next loop iteration
            } // Close scope / block
            tpos[s1] = tpos[s0]; tchk[s1] = tchk[s0]; // Compute and assign value
            tpos[s0] = p; tchk[s0] = chk; // Compute and assign value
        } // Close scope / block
        return true; // Return result to caller
    } // Close scope / block

    FileCache cs(src, n), cd(src, n); File in; // Execute statement
    if (!cs.ok() || !cd.ok() || !in.open(src, "rb")) return false; // Conditional branch check
    const size_t kBuf = (size_t)16 << 20; std::vector<u8> buf(kBuf); // Allocate / manage dynamic buffer
    u64 pos = 0, litStart = 0, bufStart = 0, bufEnd = 0, h = 0; int fill = 0; u8 win[32]; unsigned wi = 0; // Compute and assign value
    while (pos < n) { // Loop while condition holds
        if (pos < bufStart || pos >= bufEnd) { // Conditional branch check
            if (prog && !prog((double)pos / (double)n)) { cancelled = true; return false; } // Conditional branch check
            if (seek64(in.f, pos) != 0) return false; // Conditional branch check
            size_t got = std::fread(buf.data(), 1, (size_t)std::min<u64>(kBuf, n - pos), in.f); if (got == 0) return false; // Buffered file I/O operation
            bufStart = pos; bufEnd = pos + got; // Compute and assign value
        } // Close scope / block
        size_t i = (size_t)(pos - bufStart), end = (size_t)(bufEnd - bufStart); bool jumped = false; // Compute and assign value
        for (; i < end; i++) { // Loop iteration
            u8 c = buf[i]; // Compute and assign value
            if (fill < 32) { h = h * B + c; win[wi] = c; wi = (wi + 1) & 31; if (++fill < 32) continue; } // Conditional branch check
            else { u8 o = win[wi]; win[wi] = c; wi = (wi + 1) & 31; h = h * B + c - (u64)o * Bpow; } // Fallback branch
            u64 mix = h * 0x9E3779B97F4A7C15ull; // Compute and assign value
            if ((mix >> shift) != 0) continue; // Conditional branch check
            u64 p = bufStart + i + 1; // Compute and assign value
            size_t slot = (size_t)((mix >> 8) & (cap - 1)); u32 chk = (u32)((h * 0xC2B2AE3D27D4EB4Full) >> 32) | 1u; // Compute and assign value
            u64 q = tpos[slot]; // Compute and assign value
            if (q && tchk[slot] == chk) { // Conditional branch check
                u64 dist = p - q, maxb = std::min<u64>(std::min<u64>(p - litStart, q), (u64)1 << 20), bk = 0; // Compute and assign value
                while (bk < maxb && cd.at(p - 1 - bk) == cs.at(q - 1 - bk)) bk++; // Loop while condition holds
                u64 fw = common_forward(cd, p, cs, q, n - p); // Compute and assign value
                u64 len = bk + fw, start = p - bk;                                     // len > dist = repeating run / pattern
                if (len >= minm && start + len >= p) { // Conditional branch check
                    out.push_back(Match{start - litStart, len, dist}); matched += len; litStart = start + len; // Append element to container
                    pos = litStart; fill = 0; h = 0; wi = 0; jumped = true; break; // Compute and assign value
                } // Close scope / block
            } // Close scope / block
            tpos[slot] = p; tchk[slot] = chk; // Compute and assign value
        } // Close scope / block
        if (!jumped) pos = bufEnd; // Conditional branch check
    } // Close scope / block
    return true; // Return result to caller
} // Close scope / block
struct CrcJob { fs::path path; u32 crc = 0; bool ok = true; }; // Declare data structure / class
inline void crc_file(CrcJob* j) { // Define helper function or constant
    File f; if (!f.open(j->path, "rb")) { j->ok = false; return; } // Compute and assign value
    std::vector<u8> b((size_t)4 << 20); size_t g; // Allocate / manage dynamic buffer
    while ((g = std::fread(b.data(), 1, b.size(), f.f)) > 0) j->crc = crc32_update(j->crc, b.data(), g); // Loop while condition holds
} // Close scope / block
// literal stream = the file without the matched ranges
struct LitRun { u64 off, len, cum; }; // Declare data structure / class
inline std::vector<LitRun> make_runs(const std::vector<Match>& m, u64 n) { // Allocate / manage dynamic buffer
    std::vector<LitRun> r; u64 cursor = 0, cum = 0; // Allocate / manage dynamic buffer
    for (const Match& x : m) { if (x.gap) { r.push_back(LitRun{cursor, x.gap, cum}); cum += x.gap; } cursor += x.gap + x.len; } // Loop iteration
    if (cursor < n) r.push_back(LitRun{cursor, n - cursor, cum}); // Conditional branch check
    return r; // Return result to caller
} // Close scope / block
inline bool read_literals(FILE* f, const std::vector<LitRun>& runs, u64 vstart, size_t L, u8* dst) { // Allocate / manage dynamic buffer
    if (runs.empty()) return L == 0; // Conditional branch check
    size_t lo = 0, hi = runs.size(); // Compute and assign value
    while (lo + 1 < hi) { size_t mid = (lo + hi) / 2; if (runs[mid].cum <= vstart) lo = mid; else hi = mid; } // Loop while condition holds
    u64 v = vstart; size_t need = L; // Compute and assign value
    for (size_t ri = lo; need && ri < runs.size(); ri++) { // Loop iteration
        const LitRun& r = runs[ri]; u64 skip = v - r.cum, take = std::min<u64>(r.len - skip, need); // Compute and assign value
        if (seek64(f, r.off + skip) != 0 || std::fread(dst, 1, (size_t)take, f) != (size_t)take) return false; // Conditional branch check
        dst += take; v += take; need -= (size_t)take; // Compute and assign value
    } // Close scope / block
    return need == 0; // Return result to caller
} // Close scope / block
// Output side: writes literal runs and executes the copy records in order. Matches within the last 64 MB are served
// from an in-memory window; older ones are read back from the output file.
class OutWriter { // Declare data structure / class
public: // Member access specifier
    OutWriter(FILE* out, const fs::path& tmp, const std::vector<Match>* m, Counters& cn) : out_(out), tmp_(tmp), m_(m), cn_(cn) { // Allocate / manage dynamic buffer
        if (m_) { ring_.resize(kRing); chunk_.resize(kChunk); gapLeft_ = m_->empty() ? ~(u64)0 : (*m_)[0].gap; } // Conditional branch check
    } // Close scope / block
    void append(const u8* p, size_t L) { // Execute statement
        if (!m_) { write(p, L); return; } // Conditional branch check
        size_t i = 0; // Compute and assign value
        for (;;) { // Loop iteration
            applyReady(); // Execute statement
            if (i >= L) break; // Conditional branch check
            size_t take = (size_t)std::min<u64>(gapLeft_, (u64)(L - i)); // Compute and assign value
            write(p + i, take); i += take; if (gapLeft_ != ~(u64)0) gapLeft_ -= take; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    void finish() { if (m_) applyReady(); } // Execute statement
    u64 pos() const { return pos_; } // Execute statement
    u32 crc() const { return crc_; } // Execute statement
private: // Member access specifier
    static constexpr size_t kRing = (size_t)64 << 20, kChunk = (size_t)4 << 20; // Define helper function or constant
    void applyReady() { // Execute statement
        while (mi_ < m_->size() && gapLeft_ == 0) { copyMatch((*m_)[mi_]); mi_++; gapLeft_ = mi_ < m_->size() ? (*m_)[mi_].gap : ~(u64)0; } // Loop while condition holds
    } // Close scope / block
    void write(const u8* p, size_t L) { // Execute statement
        if (!L) return; // Conditional branch check
        if (std::fwrite(p, 1, L, out_) != L) throw Failure{Status::DiskFull, "write failed (disk full?)"}; // Conditional branch check
        if (m_) { // Conditional branch check
            crc_ = crc32_update(crc_, p, L); // Update / verify CRC-32 integrity checksum
            size_t at = (size_t)(pos_ & (kRing - 1)), first = std::min(L, kRing - at); // Compute and assign value
            std::memcpy(ring_.data() + at, p, first); if (first < L) std::memcpy(ring_.data(), p + first, L - first); // Fast raw memory copy
        } // Close scope / block
        pos_ += L; // Compute and assign value
    } // Close scope / block
    void readBack(u64 src, size_t c, u64 dist, u8* dst) {                 // c bytes starting at output offset src
        if (dist <= kRing) { // Conditional branch check
            size_t at = (size_t)(src & (kRing - 1)), first = std::min(c, kRing - at); // Compute and assign value
            std::memcpy(dst, ring_.data() + at, first); if (first < c) std::memcpy(dst + first, ring_.data(), c - first); // Fast raw memory copy
        } else { // Execute statement
            if (std::fflush(out_) != 0) throw Failure{Status::DiskFull, "write failed (disk full?)"}; // Conditional branch check
            if (!rd_.f && !rd_.open(tmp_, "rb")) throw Failure{Status::IoError, "cannot re-read the output file"}; // Conditional branch check
            if (seek64(rd_.f, src) != 0 || std::fread(dst, 1, c, rd_.f) != c) throw Failure{Status::IoError, "cannot re-read the output file"}; // Conditional branch check
        } // Close scope / block
    } // Close scope / block
    void copyMatch(const Match& m) { // Execute statement
        if (m.dist > pos_) throw Failure{Status::Corrupt, "file is damaged (bad copy record)"}; // Conditional branch check
        u64 left = m.len; // Compute and assign value
        while (left) { // Loop while condition holds
            size_t c = (size_t)std::min<u64>(left, kChunk); // Compute and assign value
            if (m.dist >= c) readBack(pos_ - m.dist, c, m.dist, chunk_.data()); // Conditional branch check
            else {                                                       // overlapping copy: repeat the last `dist` bytes
                size_t d = (size_t)m.dist; readBack(pos_ - m.dist, d, m.dist, chunk_.data()); // Compute and assign value
                for (size_t i = d; i < c; i++) chunk_[i] = chunk_[i - d]; // Loop iteration
            } // Close scope / block
            write(chunk_.data(), c); cn_.out.fetch_add(c, std::memory_order_relaxed); left -= c; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    FILE* out_; fs::path tmp_; const std::vector<Match>* m_; Counters& cn_; File rd_; // Allocate / manage dynamic buffer
    std::vector<u8> ring_, chunk_; u64 pos_ = 0, gapLeft_ = 0; size_t mi_ = 0; u32 crc_ = 0; // Allocate / manage dynamic buffer
}; // Close scope / block

// ------------------------------------------------------------------ generator detector (MT19937)
// Files written by Python's random.randbytes / numpy RandomState.bytes are the raw 32-bit outputs of a Mersenne Twister.
// 624 outputs reveal its whole internal state, so the rest of the file can be regenerated exactly.
class Mt19937Stream { // Declare data structure / class
public: // Member access specifier
    explicit Mt19937Stream(const u32* first624) { for (int i = 0; i < 624; i++) mt_[i] = untemper(first624[i]); idx_ = 624; } // Compute and assign value
    u32 next() { if (idx_ == 624) { twist(); idx_ = 0; } return temper(mt_[idx_++]); } // Compute and assign value
private: // Member access specifier
    static u32 temper(u32 y) { y ^= y >> 11; y ^= (y << 7) & 0x9D2C5680u; y ^= (y << 15) & 0xEFC60000u; y ^= y >> 18; return y; } // Define helper function or constant
    static u32 untemper(u32 y) { // Define helper function or constant
        y ^= y >> 18; y ^= (y << 15) & 0xEFC60000u; // Compute and assign value
        u32 t = y; for (int i = 0; i < 4; i++) t = y ^ ((t << 7) & 0x9D2C5680u); y = t; // Compute and assign value
        t = y; for (int i = 0; i < 2; i++) t = y ^ (t >> 11); return t; // Compute and assign value
    } // Close scope / block
    void twist() { // Execute statement
        for (int i = 0; i < 624; i++) { // Loop iteration
            u32 y = (mt_[i] & 0x80000000u) | (mt_[(i + 1) % 624] & 0x7fffffffu); // Compute and assign value
            mt_[i] = mt_[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DFu : 0u); // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    u32 mt_[624]; int idx_; // Execute statement
}; // Close scope / block
inline u32 rd32le(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); } // Define helper function or constant
// true only if EVERY byte of the file matches the regenerated stream (the last 0-3 bytes are kept raw)
inline bool detect_mt19937(const fs::path& src, u64 n, Header& h) { // Define helper function or constant
    if (n < ((u64)100 << 10)) return false; // Conditional branch check
    File f; if (!f.open(src, "rb")) return false; // Execute statement
    std::vector<u8> head(624 * 4); if (std::fread(head.data(), 1, head.size(), f.f) != head.size()) return false; // Buffered file I/O operation
    u32 w[624]; for (int i = 0; i < 624; i++) w[i] = rd32le(&head[(size_t)i * 4]); // Compute and assign value
    Mt19937Stream mt(w); u32 crc = crc32_update(0, head.data(), head.size()); // Update / verify CRC-32 integrity checksum
    u64 words = n / 4, done = 624; std::vector<u8> buf((size_t)4 << 20); u64 compared = 0; // Allocate / manage dynamic buffer
    while (compared < words - 624) { // Loop while condition holds
        size_t cnt = (size_t)std::min<u64>(buf.size() / 4, words - 624 - compared); // Compute and assign value
        if (std::fread(buf.data(), 1, cnt * 4, f.f) != cnt * 4) return false; // Conditional branch check
        for (size_t i = 0; i < cnt; i++) if (rd32le(&buf[i * 4]) != mt.next()) return false; // Loop iteration
        crc = crc32_update(crc, buf.data(), cnt * 4); compared += cnt; done += cnt; // Update / verify CRC-32 integrity checksum
    } // Close scope / block
    size_t tl = (size_t)(n - words * 4); std::vector<u8> tail(tl); // Allocate / manage dynamic buffer
    if (tl && std::fread(tail.data(), 1, tl, f.f) != tl) return false; // Conditional branch check
    if (tl) crc = crc32_update(crc, tail.data(), tl); // Conditional branch check
    h.gen = true; h.genPrefix = head; h.genTail = tail; h.fileCrc = crc; return true; // Compute and assign value
} // Close scope / block

// ------------------------------------------------------------------ JPEG recompression (baseline / extended-sequential Huffman)
// The entropy-coded scan is decoded to quantised DCT coefficients, which are coded with a context-mixing model that looks at
// the neighbouring blocks; extraction re-creates the exact original scan bytes. The compressor re-encodes and compares before
// it accepts the method, so anything unusual (progressive, odd padding, non-canonical Huffman use) just falls back.
namespace jpg { // Open namespace scope
struct Huff { // Declare data structure / class
    bool present = false; u8 bits[17] = {}; u8 vals[256] = {}; int mincode[17] = {}, maxcode[18] = {}, valptr[17] = {}; u16 code[256] = {}; u8 len[256] = {}; // Compute and assign value
}; // Close scope / block
struct Comp { int id = 0, h = 1, v = 1, td = 0, ta = 0, bw = 0, bh = 0; std::vector<i16> coef; std::vector<u8> nz; }; // Declare data structure / class
struct Info { // Declare data structure / class
    int width = 0, height = 0, ncomp = 0, restart = 0, mcusX = 0, mcusY = 0, hmax = 1, vmax = 1; // Compute and assign value
    Comp c[4]; Huff dc[4], ac[4]; size_t scanStart = 0; // Compute and assign value
}; // Close scope / block
inline bool build(Huff& t) { // Define helper function or constant
    int code = 0, k = 0; // Compute and assign value
    for (int l = 1; l <= 16; l++) { // Loop iteration
        t.valptr[l] = k; t.mincode[l] = code; // Compute and assign value
        for (int i = 0; i < t.bits[l]; i++) { if (k >= 256 || code >= (1 << l)) return false; t.code[t.vals[k]] = (u16)code; t.len[t.vals[k]] = (u8)l; k++; code++; } // Loop iteration
        t.maxcode[l] = t.bits[l] ? code - 1 : -1; code <<= 1; // Compute and assign value
    } // Close scope / block
    t.present = true; return true; // Compute and assign value
} // Close scope / block
// parses the markers up to and including the SOS header; false for anything that is not baseline / extended sequential 8-bit
inline bool parse_header(const u8* d, size_t n, Info& j) { // Define helper function or constant
    if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return false; // Conditional branch check
    size_t pos = 2; bool haveSof = false; // Compute and assign value
    while (pos + 4 <= n) { // Loop while condition holds
        if (d[pos] != 0xFF) return false; // Conditional branch check
        while (pos + 1 < n && d[pos + 1] == 0xFF) pos++; // Loop while condition holds
        int m = d[pos + 1]; pos += 2; // Compute and assign value
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD8)) continue; // Conditional branch check
        if (m == 0xD9) return false; // Conditional branch check
        if (pos + 2 > n) return false; // Conditional branch check
        size_t len = ((size_t)d[pos] << 8) | d[pos + 1]; if (len < 2 || pos + len > n) return false; // Compute and assign value
        const u8* s = d + pos + 2; size_t sl = len - 2; // Compute and assign value
        if (m == 0xC0 || m == 0xC1) { // Conditional branch check
            if (haveSof || sl < 6 || s[0] != 8) return false; // Conditional branch check
            j.height = (s[1] << 8) | s[2]; j.width = (s[3] << 8) | s[4]; j.ncomp = s[5]; // Compute and assign value
            if (j.width < 1 || j.height < 1 || j.ncomp < 1 || j.ncomp > 4 || sl < (size_t)(6 + 3 * j.ncomp)) return false; // Conditional branch check
            for (int i = 0; i < j.ncomp; i++) { j.c[i].id = s[6 + 3 * i]; j.c[i].h = s[7 + 3 * i] >> 4; j.c[i].v = s[7 + 3 * i] & 15; if (j.c[i].h < 1 || j.c[i].h > 4 || j.c[i].v < 1 || j.c[i].v > 4) return false; } // Loop iteration
            haveSof = true; // Compute and assign value
        } else if ((m >= 0xC2 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) || m == 0xC8 || m == 0xCC) return false;      // progressive / lossless / arithmetic
        else if (m == 0xC4) { // Alternative conditional branch
            size_t o = 0; // Compute and assign value
            while (o + 17 <= sl) { // Loop while condition holds
                int tc = s[o] >> 4, th = s[o] & 15; if (tc > 1 || th > 3) return false; // Compute and assign value
                Huff& t = tc ? j.ac[th] : j.dc[th]; t = Huff(); size_t tot = 0; // Compute and assign value
                for (int l = 1; l <= 16; l++) { t.bits[l] = s[o + l]; tot += t.bits[l]; } // Loop iteration
                if (tot > 256 || o + 17 + tot > sl) return false; // Conditional branch check
                for (size_t i = 0; i < tot; i++) t.vals[i] = s[o + 17 + i]; // Loop iteration
                if (!build(t)) return false; // Conditional branch check
                o += 17 + tot; // Compute and assign value
            } // Close scope / block
        } else if (m == 0xDD) { if (sl < 2) return false; j.restart = (s[0] << 8) | s[1]; } // Compute and assign value
        else if (m == 0xDA) { // Alternative conditional branch
            if (!haveSof || sl < 1 || s[0] != j.ncomp || sl < (size_t)(1 + 2 * s[0] + 3)) return false; // Conditional branch check
            for (int i = 0; i < j.ncomp; i++) { // Loop iteration
                int cid = s[1 + 2 * i], td = s[2 + 2 * i] >> 4, ta = s[2 + 2 * i] & 15; if (cid != j.c[i].id || td > 3 || ta > 3) return false; // Compute and assign value
                if (!j.dc[td].present || !j.ac[ta].present) return false; // Conditional branch check
                j.c[i].td = td; j.c[i].ta = ta; // Compute and assign value
            } // Close scope / block
            size_t q = 1 + 2 * j.ncomp; if (s[q] != 0 || s[q + 1] != 63 || s[q + 2] != 0) return false; // Compute and assign value
            j.scanStart = pos + len; // Compute and assign value
            if (j.ncomp == 1 && (j.c[0].h != 1 || j.c[0].v != 1)) return false; // Conditional branch check
            for (int i = 0; i < j.ncomp; i++) { j.hmax = std::max(j.hmax, j.c[i].h); j.vmax = std::max(j.vmax, j.c[i].v); } // Loop iteration
            j.mcusX = (j.width + 8 * j.hmax - 1) / (8 * j.hmax); j.mcusY = (j.height + 8 * j.vmax - 1) / (8 * j.vmax); // Compute and assign value
            u64 total = 0; // Compute and assign value
            for (int i = 0; i < j.ncomp; i++) { j.c[i].bw = j.mcusX * j.c[i].h; j.c[i].bh = j.mcusY * j.c[i].v; total += (u64)j.c[i].bw * j.c[i].bh * 64; } // Loop iteration
            if (total > ((u64)200 << 20)) return false; // Conditional branch check
            for (int i = 0; i < j.ncomp; i++) { size_t nb = (size_t)j.c[i].bw * j.c[i].bh; j.c[i].coef.assign(nb * 64, 0); j.c[i].nz.assign(nb, 0); } // Loop iteration
            return true; // Return result to caller
        } // Close scope / block
        pos += len; // Compute and assign value
    } // Close scope / block
    return false; // Return result to caller
} // Close scope / block
struct BitReader { // Declare data structure / class
    const u8* d; size_t n, pos; u8 cur = 0; int nb = 0; bool err = false; // Compute and assign value
    int get() { // Execute statement
        if (nb == 0) { // Conditional branch check
            if (pos >= n) { err = true; return 0; } // Conditional branch check
            u8 b = d[pos]; if (b == 0xFF) { if (pos + 1 < n && d[pos + 1] == 0x00) pos += 2; else { err = true; return 0; } } else pos++; // Compute and assign value
            cur = b; nb = 8; // Compute and assign value
        } // Close scope / block
        nb--; return (cur >> nb) & 1; // Execute statement
    } // Close scope / block
    int bits(int k) { int v = 0; while (k--) v = (v << 1) | get(); return v; } // Compute and assign value
    bool pad_ok() { if (nb > 0) { int rem = cur & ((1 << nb) - 1); if (rem != (1 << nb) - 1) return false; nb = 0; } return true; } // Compute and assign value
}; // Close scope / block
inline int hdecode(BitReader& br, const Huff& t) { // Define helper function or constant
    int code = 0; // Compute and assign value
    for (int l = 1; l <= 16; l++) { code = (code << 1) | br.get(); if (t.maxcode[l] >= 0 && code <= t.maxcode[l] && code >= t.mincode[l]) return t.vals[t.valptr[l] + code - t.mincode[l]]; } // Loop iteration
    return -1; // Return result to caller
} // Close scope / block
inline int extend(int v, int s) { return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v; } // Define helper function or constant
inline bool decode_scan(const u8* d, size_t n, Info& j, size_t& scanEnd) { // Define helper function or constant
    BitReader br{d, n, j.scanStart}; int pred[4] = {0, 0, 0, 0}; size_t total = (size_t)j.mcusX * (size_t)j.mcusY, rst = 0; // Compute and assign value
    for (size_t m = 0; m < total; m++) { // Loop iteration
        if (j.restart && m && m % (size_t)j.restart == 0) { // Conditional branch check
            if (!br.pad_ok() || br.pos + 2 > n || d[br.pos] != 0xFF || d[br.pos + 1] != 0xD0 + (int)(rst & 7)) return false; // Conditional branch check
            br.pos += 2; rst++; for (int i = 0; i < 4; i++) pred[i] = 0; // Compute and assign value
        } // Close scope / block
        int mx = (int)(m % (size_t)j.mcusX), my = (int)(m / (size_t)j.mcusX); // Compute and assign value
        for (int ci = 0; ci < j.ncomp; ci++) { // Loop iteration
            Comp& c = j.c[ci]; // Compute and assign value
            for (int vy = 0; vy < c.v; vy++) for (int hx = 0; hx < c.h; hx++) { // Loop iteration
                i16* blk = &c.coef[((size_t)(my * c.v + vy) * (size_t)c.bw + (size_t)(mx * c.h + hx)) * 64]; // Compute and assign value
                int t = hdecode(br, j.dc[c.td]); if (t < 0 || t > 11) return false; // Compute and assign value
                int diff = t ? extend(br.bits(t), t) : 0; pred[ci] += diff; if (pred[ci] < -32000 || pred[ci] > 32000) return false; blk[0] = (i16)pred[ci]; // Compute and assign value
                int k = 1; // Compute and assign value
                while (k < 64) { // Loop while condition holds
                    int rs = hdecode(br, j.ac[c.ta]); if (rs < 0) return false; // Compute and assign value
                    int r = rs >> 4, s = rs & 15; // Compute and assign value
                    if (s == 0) { if (r == 15) { k += 16; continue; } break; } // Conditional branch check
                    if (s > 10) return false; // Conditional branch check
                    k += r; if (k > 63) return false; blk[k] = (i16)extend(br.bits(s), s); k++; // Compute and assign value
                } // Close scope / block
                if (br.err) return false; // Conditional branch check
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    if (!br.pad_ok() || br.err) return false; // Conditional branch check
    scanEnd = br.pos; return true; // Compute and assign value
} // Close scope / block
struct BitWriter { // Declare data structure / class
    std::vector<u8>& out; u32 acc = 0; int n = 0; // Allocate / manage dynamic buffer
    explicit BitWriter(std::vector<u8>& o) : out(o) {} // Allocate / manage dynamic buffer
    void put(u32 code, int len) { // Execute statement
        acc = (acc << len) | code; n += len; // Compute and assign value
        while (n >= 8) { u8 b = (u8)(acc >> (n - 8)); out.push_back(b); if (b == 0xFF) out.push_back(0); n -= 8; } // Loop while condition holds
        acc &= (n ? ((1u << n) - 1) : 0u); // Compute and assign value
    } // Close scope / block
    void pad() { if (n) put((1u << (8 - n)) - 1, 8 - n); } // Execute statement
}; // Close scope / block
inline int bitlen(int v) { int n = 0; while (v) { n++; v >>= 1; } return n; } // Define helper function or constant
inline bool encode_scan(const Info& j, std::vector<u8>& out) { // Allocate / manage dynamic buffer
    BitWriter bw(out); int pred[4] = {0, 0, 0, 0}; size_t total = (size_t)j.mcusX * (size_t)j.mcusY, rst = 0; // Compute and assign value
    for (size_t m = 0; m < total; m++) { // Loop iteration
        if (j.restart && m && m % (size_t)j.restart == 0) { // Conditional branch check
            bw.pad(); out.push_back(0xFF); out.push_back((u8)(0xD0 + (rst & 7))); rst++; for (int i = 0; i < 4; i++) pred[i] = 0; // Append element to container
        } // Close scope / block
        int mx = (int)(m % (size_t)j.mcusX), my = (int)(m / (size_t)j.mcusX); // Compute and assign value
        for (int ci = 0; ci < j.ncomp; ci++) { // Loop iteration
            const Comp& c = j.c[ci]; const Huff& dc = j.dc[c.td]; const Huff& ac = j.ac[c.ta]; // Compute and assign value
            for (int vy = 0; vy < c.v; vy++) for (int hx = 0; hx < c.h; hx++) { // Loop iteration
                const i16* blk = &c.coef[((size_t)(my * c.v + vy) * (size_t)c.bw + (size_t)(mx * c.h + hx)) * 64]; // Compute and assign value
                int diff = blk[0] - pred[ci]; pred[ci] = blk[0]; int a = diff < 0 ? -diff : diff, t = bitlen(a); // Compute and assign value
                if (t > 11 || !dc.len[t]) return false; // Conditional branch check
                bw.put(dc.code[t], dc.len[t]); if (t) bw.put((u32)(diff < 0 ? diff + (1 << t) - 1 : diff), t); // Execute statement
                int r = 0; // Compute and assign value
                for (int k = 1; k < 64; k++) { // Loop iteration
                    int v = blk[k]; if (v == 0) { r++; continue; } // Compute and assign value
                    while (r > 15) { if (!ac.len[0xF0]) return false; bw.put(ac.code[0xF0], ac.len[0xF0]); r -= 16; } // Loop while condition holds
                    int av = v < 0 ? -v : v, s = bitlen(av), sym = (r << 4) | s; // Compute and assign value
                    if (s > 10 || !ac.len[sym]) return false; // Conditional branch check
                    bw.put(ac.code[sym], ac.len[sym]); bw.put((u32)(v < 0 ? v + (1 << s) - 1 : v), s); r = 0; // Compute and assign value
                } // Close scope / block
                if (r > 0) { if (!ac.len[0]) return false; bw.put(ac.code[0], ac.len[0]); } // Conditional branch check
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    bw.pad(); return true; // Execute statement
} // Close scope / block

// ---- coefficient model: three context-selected counters per decision, mixed by a small integer mixer
class Model { // Declare data structure / class
public: // Member access specifier
    Model() : t1_((size_t)1 << 20, kInit), t2_((size_t)1 << 20, kInit), t3_((size_t)1 << 20, kInit), mw_((size_t)kSel * 4, 22000) {} // Execute statement
    Enc* e = nullptr; Dec* d = nullptr; // Compute and assign value
    int bit(int y, u32 c1, u32 c2, u32 c3, u32 sel) { // Execute statement
        const Tables& T = tables(); // Compute and assign value
        u32* s1 = &t1_[hmix(c1, 0x51ED27u) >> 12]; u32* s2 = &t2_[hmix(c2, 0x2545F49u) >> 12]; u32* s3 = &t3_[hmix(c3, 0x7F4A7C1u) >> 12]; // Compute and assign value
        int st[4] = {T.stretch_[clampi((int)(*s1 >> 20), 4, 4091)], T.stretch_[clampi((int)(*s2 >> 20), 4, 4091)], T.stretch_[clampi((int)(*s3 >> 20), 4, 4091)], 128}; // Compute and assign value
        i32* w = &mw_[(size_t)(sel % kSel) * 4]; // Compute and assign value
        i64 dot = 0; for (int i = 0; i < 4; i++) dot += (i64)w[i] * st[i]; // Compute and assign value
        int p = clampi(squash(clampi((int)(dot >> 16), -2047, 2047)), 1, 4095); // Compute and assign value
        int yy; if (e) { e->bit(y, p); yy = y; } else yy = d->bit(p); // Compute and assign value
        int err = ((yy << 12) - p) * 6; for (int i = 0; i < 4; i++) w[i] += (st[i] * err + 0x8000) >> 16; // Compute and assign value
        sm_up(s1, yy, 255); sm_up(s2, yy, 255); sm_up(s3, yy, 255); // Execute statement
        return yy; // Return result to caller
    } // Close scope / block
private: // Member access specifier
    static constexpr int kSel = 4096; // Define helper function or constant
    std::vector<u32> t1_, t2_, t3_; std::vector<i32> mw_; // Allocate / manage dynamic buffer
}; // Close scope / block
inline u32 H(u32 a, u32 b = 0, u32 c = 0, u32 d = 0, u32 e = 0) { return hmix(hmix(hmix(hmix(a + 1, b), c), d), e); } // Define helper function or constant
inline int qb(int x) { return x < 4 ? x : 2 + bitlen(x); } // Define helper function or constant
// codes one block (both directions); coef values are read when encoding and written when decoding
inline void code_block(Model& M, Info& j, int ci, int bx, int by, int& prevNz, int& prevDcMag) { // Define helper function or constant
    Comp& c = j.c[ci]; const bool enc = M.e != nullptr; const int ct = ci ? 1 : 0, bw = c.bw; // Compute and assign value
    i16* blk = &c.coef[((size_t)by * (size_t)bw + (size_t)bx) * 64]; // Compute and assign value
    const i16* L = bx > 0 ? blk - 64 : nullptr; const i16* A = by > 0 ? blk - (size_t)bw * 64 : nullptr; const i16* AL = (bx > 0 && by > 0) ? blk - (size_t)bw * 64 - 64 : nullptr; // Compute and assign value
    // ---- DC: residual against a gradient prediction from the left / above blocks
    int pred, gb = 15; // Compute and assign value
    if (L && A) { int l = L[0], a = A[0], al = AL[0], g = l + a - al, lo = std::min(l, a), hi = std::max(l, a); pred = g < lo ? lo : (g > hi ? hi : g); gb = bitlen(std::abs(l - a)); } // Conditional branch check
    else if (L) pred = L[0]; else if (A) pred = A[0]; else pred = 0; // Alternative conditional branch
    int e = enc ? blk[0] - pred : 0, ae = e < 0 ? -e : e, s = bitlen(ae); // Compute and assign value
    int zero = M.bit(ae == 0, H(20, ct, gb), H(21, ct, prevDcMag), H(22, ct, gb, prevDcMag), 20 * 64 + (u32)ct); // Compute and assign value
    int val = 0; // Compute and assign value
    if (!zero) { // Conditional branch check
        int sg = M.bit(e < 0, H(23, ct, gb, L && A ? (L[0] > A[0]) : 2), H(24, ct, prevDcMag), H(25, ct), 21 * 64 + (u32)ct); // Compute and assign value
        int sz = 1; for (; sz < 12; sz++) { int more = M.bit(s > sz, H(26, ct, gb, sz), H(27, ct, prevDcMag, sz), H(28, ct, sz), 22 * 64 + (u32)sz); if (!more) break; } // Compute and assign value
        int mag = 1; for (int b = sz - 2; b >= 0; b--) { int y = M.bit((ae >> b) & 1, H(29, sz, b), H(30, ct, sz, b, mag & 3), H(31, b), 23 * 64 + (u32)b); mag = (mag << 1) | y; } // Compute and assign value
        val = sg ? -mag : mag; // Compute and assign value
    } // Close scope / block
    if (!enc) blk[0] = (i16)(pred + val); // Conditional branch check
    prevDcMag = bitlen(std::abs(val)); // Compute and assign value
    // ---- number of non-zero AC coefficients (6-bit tree), then the coefficients in zig-zag order
    const size_t bi = (size_t)by * (size_t)bw + (size_t)bx; // Compute and assign value
    int nl = L ? c.nz[bi - 1] : -1, na = A ? c.nz[bi - (size_t)bw] : -1, cnt = (nl >= 0) + (na >= 0), avg = cnt ? (std::max(nl, 0) + std::max(na, 0) + cnt / 2) / cnt : 0; // Compute and assign value
    int nzv = 0; if (enc) for (int k = 1; k < 64; k++) nzv += blk[k] != 0; // Compute and assign value
    int node = 1; // Compute and assign value
    for (int b = 5; b >= 0; b--) { // Loop iteration
        int y = M.bit((nzv >> b) & 1, H(1, ct, node, avg, cnt), H(2, ct, node, nl + 1, na + 1), H(3, ct, node, prevNz), 1 * 64 + (u32)(5 - b)); // Compute and assign value
        node = (node << 1) | y; // Compute and assign value
    } // Close scope / block
    nzv = node & 63; prevNz = nzv; c.nz[bi] = (u8)nzv; // Compute and assign value
    int remain = nzv; // Compute and assign value
    for (int k = 1; k < 64 && remain > 0; k++) { // Loop iteration
        int v = enc ? blk[k] : 0, p1 = k >= 2 ? std::abs(blk[k - 1]) : 0, p2 = k >= 3 ? std::abs(blk[k - 2]) : 0; // Compute and assign value
        int ml = L ? std::abs(L[k]) : -1, ma = A ? std::abs(A[k]) : -1, nbm = (ml >= 0 && ma >= 0) ? (ml + ma + 1) / 2 : std::max(std::max(ml, ma), 0); // Compute and assign value
        int nzf; // Execute statement
        if (remain == 64 - k) nzf = 1;                                           // every position left must be non-zero
        else nzf = M.bit(v != 0, H(40, ct, (u32)k, qb(nbm), (ml >= 0) + (ma >= 0)), H(41, ct, (u32)k, std::min(p1, 7), std::min(p2, 7)), H(42, ct, (u32)k, std::min(remain, 15), std::min((64 - k) / 4, 15)), 2 * 64 + (u32)k); // Fallback branch
        if (!nzf) { continue; } // Conditional branch check
        remain--; // Execute statement
        int av = v < 0 ? -v : v, sv = bitlen(av), mag; // Compute and assign value
        int sz = 1; for (; sz < 10; sz++) { int more = M.bit(sv > sz, H(50, ct, (u32)k, (u32)sz, qb(nbm)), H(51, ct, (u32)k, (u32)sz, bitlen(p1)), H(52, ct, (u32)sz, bitlen(nbm), std::min(remain, 7)), 3 * 64 + (u32)std::min(k, 63)); if (!more) break; } // Compute and assign value
        mag = 1; for (int b = sz - 2; b >= 0; b--) { int y = M.bit((av >> b) & 1, H(53, (u32)sz, (u32)b, b == sz - 2 ? (u32)k : 0u), H(54, ct, (u32)sz, (u32)b), H(55, (u32)b, qb(nbm)), 4 * 64 + (u32)b); mag = (mag << 1) | y; } // Compute and assign value
        int sl = L ? (L[k] > 0 ? 1 : (L[k] < 0 ? 2 : 0)) : 3, sa = A ? (A[k] > 0 ? 1 : (A[k] < 0 ? 2 : 0)) : 3; // Compute and assign value
        int neg = M.bit(v < 0, H(56, ct, (u32)k, sl, sa), H(57, (u32)k, sl), H(58, (u32)k, sa), 5 * 64 + (u32)k); // Compute and assign value
        if (!enc) blk[k] = (i16)(neg ? -mag : mag); // Conditional branch check
    } // Close scope / block
} // Close scope / block
inline void code_all(Model& M, Info& j) { // Define helper function or constant
    int prevNz[4] = {0, 0, 0, 0}, prevDc[4] = {0, 0, 0, 0}; // Compute and assign value
    size_t total = (size_t)j.mcusX * (size_t)j.mcusY; // Compute and assign value
    for (size_t m = 0; m < total; m++) { // Loop iteration
        int mx = (int)(m % (size_t)j.mcusX), my = (int)(m / (size_t)j.mcusX); // Compute and assign value
        for (int ci = 0; ci < j.ncomp; ci++) { Comp& c = j.c[ci]; for (int vy = 0; vy < c.v; vy++) for (int hx = 0; hx < c.h; hx++) code_block(M, j, ci, mx * c.h + hx, my * c.v + vy, prevNz[ci], prevDc[ci]); } // Loop iteration
    } // Close scope / block
} // Close scope / block
}  // namespace jpg

// payload = [varint hdrLen][varint tailLen][varint blobLen][blob stream][coefficient stream]; blob = header bytes + tail bytes
inline bool jpeg_try(const u8* d, size_t n, std::vector<u8>& payload) { // Allocate / manage dynamic buffer
    jpg::Info j; if (!jpg::parse_header(d, n, j)) return false; // Execute statement
    size_t scanEnd = 0; if (!jpg::decode_scan(d, n, j, scanEnd)) return false; // Compute and assign value
    std::vector<u8> re; re.reserve(scanEnd - j.scanStart + 16); // Allocate / manage dynamic buffer
    if (!jpg::encode_scan(j, re) || re.size() != scanEnd - j.scanStart || std::memcmp(re.data(), d + j.scanStart, re.size()) != 0) return false;   // must be bit-exact
    std::vector<u8> blob(d, d + j.scanStart); blob.insert(blob.end(), d + scanEnd, d + n); // Allocate / manage dynamic buffer
    if (blob.size() > kSegment) return false; // Conditional branch check
    std::vector<u8> blobOut; Counters dummy; encode_segment(blob.data(), blob.size(), 20, Hint(), blobOut, dummy); // Allocate / manage dynamic buffer
    std::vector<u8> coefOut; Enc enc(coefOut); jpg::Model* M = new jpg::Model(); M->e = &enc; jpg::code_all(*M, j); delete M; enc.finish(); // Allocate / manage dynamic buffer
    payload.clear(); put_varint(payload, j.scanStart); put_varint(payload, n - scanEnd); put_varint(payload, blobOut.size()); // Encode / decode variable-length integer
    payload.insert(payload.end(), blobOut.begin(), blobOut.end()); payload.insert(payload.end(), coefOut.begin(), coefOut.end()); // Execute statement
    return true; // Return result to caller
} // Close scope / block
inline bool jpeg_restore(const std::vector<u8>& payload, std::vector<u8>& file) { // Allocate / manage dynamic buffer
    const u8* p = payload.data(); const u8* end = p + payload.size(); u64 v[3]; // Compute and assign value
    for (int i = 0; i < 3; i++) { v[i] = 0; int sh = 0; for (;;) { if (p >= end || sh > 56) return false; u8 c = *p++; v[i] |= (u64)(c & 127) << sh; if (!(c & 128)) break; sh += 7; } } // Loop iteration
    u64 hdrLen = v[0], tailLen = v[1], blobLen = v[2]; // Compute and assign value
    if (blobLen > (u64)(end - p) || hdrLen + tailLen > kSegment || hdrLen < 4) return false; // Conditional branch check
    std::vector<u8> blob((size_t)(hdrLen + tailLen)); jpg::Info j; // Allocate / manage dynamic buffer
    try { decode_segment(p, (size_t)blobLen, blob.data(), blob.size(), 20, Hint(), nullptr); } catch (const Failure&) { return false; } // Begin exception-guarded block
    if (!jpg::parse_header(blob.data(), (size_t)hdrLen, j) || j.scanStart != hdrLen) return false; // Conditional branch check
    const u8* cp = p + blobLen; Dec dec(cp, (size_t)(end - cp)); jpg::Model* M = new jpg::Model(); M->d = &dec; jpg::code_all(*M, j); delete M; // Compute and assign value
    std::vector<u8> scan; if (!jpg::encode_scan(j, scan)) return false; // Allocate / manage dynamic buffer
    file.assign(blob.begin(), blob.begin() + (size_t)hdrLen); file.insert(file.end(), scan.begin(), scan.end()); file.insert(file.end(), blob.begin() + (size_t)hdrLen, blob.end()); // Execute statement
    return true; // Return result to caller
} // Close scope / block

// ------------------------------------------------------------------ deflate streams: exact re-creation of zlib output
// inflate()  : own decoder (no zlib needed).   Def : a faithful port of zlib's deflate (levels 1-9, default strategy), used to
// prove that a stream was written by zlib with some (level, memLevel, windowBits) and to rebuild it bit for bit on extraction.
namespace zl { // Open namespace scope
struct Huff { std::vector<u16> tab; int bits = 0; };           // entry = symbol << 4 | length, indexed by the next `bits` input bits
inline bool build_huff(const u8* lens, int n, Huff& h) { // Define helper function or constant
    int cnt[16] = {0}; for (int i = 0; i < n; i++) cnt[lens[i]]++; // Compute and assign value
    int maxl = 0, total = 0; for (int l = 1; l < 16; l++) if (cnt[l]) { maxl = l; total += cnt[l]; } // Compute and assign value
    if (maxl == 0) { h.bits = 1; h.tab.assign(2, 0); return true; }           // no codes: every lookup fails
    int left = 1; for (int l = 1; l < 16; l++) { left <<= 1; left -= cnt[l]; if (left < 0) return false; } // Compute and assign value
    if (left > 0 && !(total == 1 && maxl == 1)) return false;                  // incomplete (only a single 1-bit code is legal)
    unsigned next[17] = {0}; { unsigned c = 0; for (int l = 1; l <= 15; l++) { c = (c + (l > 1 ? (unsigned)cnt[l - 1] : 0u)) << 1; next[l] = c; } } // Compute and assign value
    h.bits = maxl; h.tab.assign((size_t)1 << maxl, 0); // Compute and assign value
    for (int s = 0; s < n; s++) { // Loop iteration
        int l = lens[s]; if (!l) continue; // Compute and assign value
        unsigned c = next[l]++, r = 0; for (int i = 0; i < l; i++) { r = (r << 1) | (c & 1); c >>= 1; } // Compute and assign value
        for (unsigned j = r; j < ((unsigned)1 << maxl); j += (1u << l)) h.tab[j] = (u16)((s << 4) | l); // Loop iteration
    } // Close scope / block
    return true; // Return result to caller
} // Close scope / block
struct BitIn { // Declare data structure / class
    const u8* p; const u8* end; const u8* start; u64 buf = 0; int cnt = 0; bool bad = false; // Compute and assign value
    BitIn(const u8* b, size_t n) : p(b), end(b + n), start(b) {} // Execute statement
    inline void fill() { while (cnt <= 56 && p < end) { buf |= (u64)*p++ << cnt; cnt += 8; } } // Define helper function or constant
    inline u32 get(int n) { if (n == 0) return 0; if (cnt < n) fill(); if (cnt < n) { bad = true; return 0; } u32 v = (u32)(buf & (((u64)1 << n) - 1)); buf >>= n; cnt -= n; return v; } // Define helper function or constant
    inline int sym(const Huff& h) { // Define helper function or constant
        if (cnt < 15) fill(); // Conditional branch check
        u16 e = h.tab[(size_t)(buf & (((u64)1 << h.bits) - 1))]; int l = e & 15; // Compute and assign value
        if (l == 0 || l > cnt) { bad = true; return -1; } // Conditional branch check
        buf >>= l; cnt -= l; return e >> 4; // Compute and assign value
    } // Close scope / block
}; // Close scope / block
// Decodes one complete deflate stream. used = number of input bytes it occupies (last partial byte included).
inline bool inflate_raw(const u8* in, size_t n, size_t& used, std::vector<u8>& out, size_t maxOut, std::vector<u32>* storedLens = nullptr, bool* sawCoded = nullptr) { // Allocate / manage dynamic buffer
    static const u16 lbase[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258}; // Define helper function or constant
    static const u8 lext[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0}; // Define helper function or constant
    static const u16 dbase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577}; // Define helper function or constant
    static const u8 dext[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13}; // Define helper function or constant
    static const u8 corder[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15}; // Define helper function or constant
    static Huff fixL, fixD; static bool fixInit = false; // Define helper function or constant
    if (!fixInit) { u8 l[288]; for (int i = 0; i < 144; i++) l[i] = 8; for (int i = 144; i < 256; i++) l[i] = 9; for (int i = 256; i < 280; i++) l[i] = 7; for (int i = 280; i < 288; i++) l[i] = 8; // Conditional branch check
                    build_huff(l, 288, fixL); u8 d[32]; for (int i = 0; i < 32; i++) d[i] = 5; build_huff(d, 32, fixD); fixInit = true; } // Compute and assign value
    BitIn br(in, n); out.clear(); // Execute statement
    for (;;) { // Loop iteration
        u32 fin = br.get(1), type = br.get(2); if (br.bad) return false; // Compute and assign value
        if (type == 0) { // Conditional branch check
            br.buf >>= (br.cnt & 7); br.cnt -= (br.cnt & 7); // Compute and assign value
            u32 len = br.get(16), nlen = br.get(16); if (br.bad || (len ^ 0xffff) != nlen) return false; // Compute and assign value
            if (out.size() + len > maxOut) return false; // Conditional branch check
            if (storedLens) { if (storedLens->size() >= ((size_t)1 << 20)) return false; storedLens->push_back(len); } // Conditional branch check
            size_t o = out.size(); out.resize(o + len); // Compute and assign value
            for (u32 i = 0; i < len; i++) { out[o + i] = (u8)br.get(8); if (br.bad) return false; } // Loop iteration
        } else if (type == 1 || type == 2) { // Compute and assign value
            if (sawCoded) *sawCoded = true; // Conditional branch check
            Huff dl, dd; const Huff* L = &fixL; const Huff* D = &fixD; // Compute and assign value
            if (type == 2) { // Conditional branch check
                u32 hlit = br.get(5) + 257, hdist = br.get(5) + 1, hclen = br.get(4) + 4; if (br.bad || hlit > 286 || hdist > 30) return false; // Compute and assign value
                u8 cl[19] = {0}; for (u32 i = 0; i < hclen; i++) cl[corder[i]] = (u8)br.get(3); if (br.bad) return false; // Compute and assign value
                Huff ch; if (!build_huff(cl, 19, ch)) return false; // Execute statement
                u8 lens[320] = {0}; u32 i = 0; // Compute and assign value
                while (i < hlit + hdist) { // Loop while condition holds
                    int s = br.sym(ch); if (s < 0) return false; // Compute and assign value
                    if (s < 16) lens[i++] = (u8)s; // Conditional branch check
                    else { // Fallback branch
                        u32 rep; u8 val = 0; // Compute and assign value
                        if (s == 16) { if (i == 0) return false; val = lens[i - 1]; rep = 3 + br.get(2); } // Conditional branch check
                        else if (s == 17) rep = 3 + br.get(3); else rep = 11 + br.get(7); // Alternative conditional branch
                        if (br.bad || i + rep > hlit + hdist) return false; // Conditional branch check
                        while (rep--) lens[i++] = val; // Loop while condition holds
                    } // Close scope / block
                } // Close scope / block
                if (lens[256] == 0) return false; // Conditional branch check
                if (!build_huff(lens, (int)hlit, dl) || !build_huff(lens + hlit, (int)hdist, dd)) return false; // Conditional branch check
                L = &dl; D = &dd; // Compute and assign value
            } // Close scope / block
            for (;;) { // Loop iteration
                int s = br.sym(*L); if (s < 0) return false; // Compute and assign value
                if (s < 256) { if (out.size() >= maxOut) return false; out.push_back((u8)s); } // Conditional branch check
                else if (s == 256) break; // Alternative conditional branch
                else { // Fallback branch
                    s -= 257; if (s >= 29) return false; // Compute and assign value
                    u32 len = lbase[s] + br.get(lext[s]); int ds = br.sym(*D); if (ds < 0 || ds >= 30) return false; // Compute and assign value
                    u32 dist = dbase[ds] + br.get(dext[ds]); if (br.bad || dist > out.size() || out.size() + len > maxOut) return false; // Compute and assign value
                    size_t o = out.size(); out.resize(o + len); u8* q = out.data() + o; const u8* src = q - dist; // Compute and assign value
                    for (u32 k = 0; k < len; k++) q[k] = src[k]; // Loop iteration
                } // Close scope / block
            } // Close scope / block
        } else return false; // Execute statement
        if (fin) break; // Conditional branch check
    } // Close scope / block
    used = (size_t)(br.p - br.start) - (size_t)(br.cnt / 8); // Compute and assign value
    return true; // Return result to caller
} // Close scope / block

// ---- zlib deflate port
enum { MIN_MATCH = 3, MAX_MATCH = 258, MIN_LOOKAHEAD = MAX_MATCH + MIN_MATCH + 1, TOO_FAR = 4096, LENGTH_CODES = 29, LITERALS = 256, // Define enumeration constants
       L_CODES = LITERALS + 1 + LENGTH_CODES, D_CODES = 30, BL_CODES = 19, HEAP_SIZE = 2 * L_CODES + 1, MAX_BITS = 15, MAX_BL_BITS = 7, // Compute and assign value
       END_BLOCK = 256, REP_3_6 = 16, REPZ_3_10 = 17, REPZ_11_138 = 18 }; // Compute and assign value
struct CT { u16 fc, dl; };                       // fc = Freq / Code, dl = Dad / Len (a union in zlib: kept aliased on purpose)
static const int kExtraL[LENGTH_CODES] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0}; // Define helper function or constant
static const int kExtraD[D_CODES] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13}; // Define helper function or constant
static const int kExtraBL[BL_CODES] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7}; // Define helper function or constant
static const u8 kBlOrder[BL_CODES] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15}; // Define helper function or constant
inline unsigned bi_reverse(unsigned code, int len) { unsigned r = 0; do { r |= code & 1; code >>= 1; r <<= 1; } while (--len > 0); return r >> 1; } // Define helper function or constant
inline void gen_codes(CT* tree, int maxCode, const u16* blCount) { // Define helper function or constant
    u16 next[MAX_BITS + 1]; unsigned code = 0; // Compute and assign value
    for (int bits = 1; bits <= MAX_BITS; bits++) { code = (code + blCount[bits - 1]) << 1; next[bits] = (u16)code; } // Loop iteration
    for (int n = 0; n <= maxCode; n++) { int len = tree[n].dl; if (len == 0) continue; tree[n].fc = (u16)bi_reverse(next[len]++, len); } // Loop iteration
} // Close scope / block
struct Tabs { // Declare data structure / class
    u8 lengthCode[256]; u8 distCode[512]; int baseLength[LENGTH_CODES]; int baseDist[D_CODES]; CT sl[L_CODES + 2]; CT sd[D_CODES]; // Execute statement
    Tabs() { // Execute statement
        int length = 0, code, n; for (code = 0; code < LENGTH_CODES - 1; code++) { baseLength[code] = length; for (n = 0; n < (1 << kExtraL[code]); n++) lengthCode[length++] = (u8)code; } // Compute and assign value
        lengthCode[length - 1] = (u8)code; // Compute and assign value
        int dist = 0; for (code = 0; code < 16; code++) { baseDist[code] = dist; for (n = 0; n < (1 << kExtraD[code]); n++) distCode[dist++] = (u8)code; } // Compute and assign value
        dist >>= 7; // Compute and assign value
        for (; code < D_CODES; code++) { baseDist[code] = dist << 7; for (n = 0; n < (1 << (kExtraD[code] - 7)); n++) distCode[256 + dist++] = (u8)code; } // Loop iteration
        u16 bl[MAX_BITS + 1] = {0}; std::memset(sl, 0, sizeof sl); // Initialize memory buffer
        n = 0; while (n <= 143) { sl[n++].dl = 8; bl[8]++; } while (n <= 255) { sl[n++].dl = 9; bl[9]++; } while (n <= 279) { sl[n++].dl = 7; bl[7]++; } while (n <= 287) { sl[n++].dl = 8; bl[8]++; } // Compute and assign value
        gen_codes(sl, L_CODES + 1, bl); // Execute statement
        for (n = 0; n < D_CODES; n++) { sd[n].dl = 5; sd[n].fc = (u16)bi_reverse((unsigned)n, 5); } // Loop iteration
    } // Close scope / block
}; // Close scope / block
inline const Tabs& tabs() { static Tabs t; return t; } // Define helper function or constant

struct TreeDesc { CT* dyn; int maxCode; const CT* stat; const int* extra; int base, elems, maxLen; }; // Declare data structure / class
struct Cfg { int good, lazy, nice, chain; bool slow; }; // Declare data structure / class
static const Cfg kCfg[10] = {{0,0,0,0,false},{4,4,8,4,false},{4,5,16,8,false},{4,6,32,32,false},{4,4,16,16,true},{8,16,32,32,true},{8,16,128,128,true},{8,32,128,256,true},{32,128,258,1024,true},{32,258,258,4096,true}}; // Define helper function or constant

struct Def { // Declare data structure / class
    const Tabs& T = tabs(); // Compute and assign value
    int level, memLevel, wBits, strategy; unsigned wSize, wMask, hashBits, hashSize, hashMask, hashShift, litBufSize; // Execute statement
    std::vector<u8> window; std::vector<u16> prev, head; // Allocate / manage dynamic buffer
    const u8* nextIn = nullptr; size_t availIn = 0; // Compute and assign value
    unsigned insH = 0; long blockStart = 0; unsigned matchLength = 0, prevMatch = 0, strStart = 0, matchStart = 0, lookahead = 0, prevLength = 0, insert = 0; // Compute and assign value
    int matchAvailable = 0; unsigned maxChain = 0, maxLazy = 0, goodMatch = 0, niceMatch = 0; size_t highWater = 0, windowSize = 0; // Compute and assign value
    CT dynL[HEAP_SIZE], dynD[2 * D_CODES + 1], blTree[2 * BL_CODES + 1]; // Execute statement
    TreeDesc lDesc, dDesc, blDesc; u16 blCount[MAX_BITS + 1]; int heap[2 * L_CODES + 1]; int heapLen = 0, heapMax = 0; u8 depth[2 * L_CODES + 1]; // Compute and assign value
    std::vector<u16> dBuf; std::vector<u8> lBuf; unsigned lastLit = 0; u64 optLen = 0, staticLen = 0; unsigned matches = 0; // Allocate / manage dynamic buffer
    u64 biBuf = 0; int biValid = 0; // Compute and assign value
    // output (match mode compares against `expect` while producing and drops confirmed bytes)
    std::vector<u8> out; const u8* expect = nullptr; size_t expectLen = 0, emitted = 0; bool bad = false; // Allocate / manage dynamic buffer

    Def(int lev, int memLev, int wb, int strat) : level(lev), memLevel(memLev), wBits(wb), strategy(strat) { // Execute statement
        wSize = 1u << wb; wMask = wSize - 1; hashBits = (unsigned)memLev + 7; hashSize = 1u << hashBits; hashMask = hashSize - 1; hashShift = (hashBits + MIN_MATCH - 1) / MIN_MATCH; // Compute and assign value
        window.assign((size_t)wSize * 2 + 8, 0); prev.assign(wSize, 0); head.assign(hashSize, 0); // Execute statement
        litBufSize = 1u << (memLev + 6); dBuf.assign(litBufSize, 0); lBuf.assign(litBufSize, 0); // Compute and assign value
        windowSize = (size_t)2 * wSize; // Compute and assign value
        const Cfg& c = kCfg[lev]; maxLazy = (unsigned)c.lazy; goodMatch = (unsigned)c.good; niceMatch = (unsigned)c.nice; maxChain = (unsigned)c.chain; // Compute and assign value
        lDesc = {dynL, 0, T.sl, kExtraL, LITERALS + 1, L_CODES, MAX_BITS}; dDesc = {dynD, 0, T.sd, kExtraD, 0, D_CODES, MAX_BITS}; blDesc = {blTree, 0, nullptr, kExtraBL, 0, BL_CODES, MAX_BL_BITS}; // Compute and assign value
        strStart = 0; blockStart = 0; lookahead = 0; insert = 0; matchLength = prevLength = MIN_MATCH - 1; matchAvailable = 0; insH = 0; // Compute and assign value
        initBlock(); // Execute statement
    } // Close scope / block
    unsigned maxDist() const { return wSize - MIN_LOOKAHEAD; } // Execute statement
    // ---- bit output
    inline void putByte(u8 b) { out.push_back(b); } // Append element to container
    inline void sendBits(unsigned v, int len) { biBuf |= (u64)v << biValid; biValid += len; while (biValid >= 8) { putByte((u8)biBuf); biBuf >>= 8; biValid -= 8; } } // Define helper function or constant
    inline void windup() { if (biValid > 0) putByte((u8)biBuf); biBuf = 0; biValid = 0; } // Define helper function or constant
    inline void sendCode(int c, const CT* t) { sendBits(t[c].fc, t[c].dl); } // Define helper function or constant
    void check() { // Execute statement
        if (!expect) return; // Conditional branch check
        if (emitted + out.size() > expectLen || (!out.empty() && std::memcmp(out.data(), expect + emitted, out.size()) != 0)) { bad = true; return; } // Conditional branch check
        emitted += out.size(); out.clear(); // Compute and assign value
    } // Close scope / block
    // ---- trees
    void initBlock() { // Execute statement
        for (int n = 0; n < L_CODES; n++) dynL[n].fc = 0; // Loop iteration
        for (int n = 0; n < D_CODES; n++) dynD[n].fc = 0; // Loop iteration
        for (int n = 0; n < BL_CODES; n++) blTree[n].fc = 0; // Loop iteration
        dynL[END_BLOCK].fc = 1; optLen = staticLen = 0; lastLit = matches = 0; // Compute and assign value
    } // Close scope / block
    inline bool smaller(const CT* t, int n, int m) const { return t[n].fc < t[m].fc || (t[n].fc == t[m].fc && depth[n] <= depth[m]); } // Define helper function or constant
    void pqdown(const CT* t, int k) { // Execute statement
        int v = heap[k], j = k << 1; // Compute and assign value
        while (j <= heapLen) { if (j < heapLen && smaller(t, heap[j + 1], heap[j])) j++; if (smaller(t, v, heap[j])) break; heap[k] = heap[j]; k = j; j <<= 1; } // Loop while condition holds
        heap[k] = v; // Compute and assign value
    } // Close scope / block
    void genBitlen(TreeDesc& d) { // Execute statement
        CT* tree = d.dyn; int maxCode = d.maxCode; const CT* stree = d.stat; const int* extra = d.extra; int base = d.base, maxLength = d.maxLen; // Compute and assign value
        int h, n, m, bits, xbits; int overflow = 0; // Compute and assign value
        for (bits = 0; bits <= MAX_BITS; bits++) blCount[bits] = 0; // Loop iteration
        tree[heap[heapMax]].dl = 0; // Compute and assign value
        for (h = heapMax + 1; h < HEAP_SIZE; h++) { // Loop iteration
            n = heap[h]; bits = tree[tree[n].dl].dl + 1; // Compute and assign value
            if (bits > maxLength) { bits = maxLength; overflow++; } // Conditional branch check
            tree[n].dl = (u16)bits; // Compute and assign value
            if (n > maxCode) continue; // Conditional branch check
            blCount[bits]++; xbits = 0; if (n >= base) xbits = extra[n - base]; // Compute and assign value
            u16 f = tree[n].fc; optLen += (u64)f * (unsigned)(bits + xbits); if (stree) staticLen += (u64)f * (unsigned)(stree[n].dl + xbits); // Compute and assign value
        } // Close scope / block
        if (overflow == 0) return; // Conditional branch check
        do { bits = maxLength - 1; while (blCount[bits] == 0) bits--; blCount[bits]--; blCount[bits + 1] += 2; blCount[maxLength]--; overflow -= 2; } while (overflow > 0); // Compute and assign value
        for (bits = maxLength; bits != 0; bits--) { // Loop iteration
            n = blCount[bits]; // Compute and assign value
            while (n != 0) { m = heap[--h]; if (m > maxCode) continue; if ((unsigned)tree[m].dl != (unsigned)bits) { optLen += ((u64)bits - tree[m].dl) * tree[m].fc; tree[m].dl = (u16)bits; } n--; } // Loop while condition holds
        } // Close scope / block
    } // Close scope / block
    void buildTree(TreeDesc& d) { // Execute statement
        CT* tree = d.dyn; const CT* stree = d.stat; int elems = d.elems; int n, m, maxCode = -1, node; // Compute and assign value
        heapLen = 0; heapMax = HEAP_SIZE; // Compute and assign value
        for (n = 0; n < elems; n++) { if (tree[n].fc != 0) { heap[++heapLen] = maxCode = n; depth[n] = 0; } else tree[n].dl = 0; } // Loop iteration
        while (heapLen < 2) { node = heap[++heapLen] = (maxCode < 2 ? ++maxCode : 0); tree[node].fc = 1; depth[node] = 0; optLen--; if (stree) staticLen -= stree[node].dl; } // Loop while condition holds
        d.maxCode = maxCode; // Compute and assign value
        for (n = heapLen / 2; n >= 1; n--) pqdown(tree, n); // Loop iteration
        node = elems; // Compute and assign value
        do { // Execute statement
            n = heap[1]; heap[1] = heap[heapLen--]; pqdown(tree, 1); // Compute and assign value
            m = heap[1]; // Compute and assign value
            heap[--heapMax] = n; heap[--heapMax] = m; // Compute and assign value
            tree[node].fc = (u16)(tree[n].fc + tree[m].fc); // Compute and assign value
            depth[node] = (u8)((depth[n] >= depth[m] ? depth[n] : depth[m]) + 1); // Compute and assign value
            tree[n].dl = tree[m].dl = (u16)node; // Compute and assign value
            heap[1] = node++; pqdown(tree, 1); // Compute and assign value
        } while (heapLen >= 2); // Compute and assign value
        heap[--heapMax] = heap[1]; // Compute and assign value
        genBitlen(d); gen_codes(tree, maxCode, blCount); // Execute statement
    } // Close scope / block
    void scanTree(CT* tree, int maxCode) { // Execute statement
        int prevlen = -1, curlen, nextlen = tree[0].dl, count = 0, maxCount = 7, minCount = 4; // Compute and assign value
        if (nextlen == 0) { maxCount = 138; minCount = 3; } // Conditional branch check
        tree[maxCode + 1].dl = 0xffff; // Compute and assign value
        for (int n = 0; n <= maxCode; n++) { // Loop iteration
            curlen = nextlen; nextlen = tree[n + 1].dl; // Compute and assign value
            if (++count < maxCount && curlen == nextlen) continue; // Conditional branch check
            else if (count < minCount) blTree[curlen].fc = (u16)(blTree[curlen].fc + count); // Alternative conditional branch
            else if (curlen != 0) { if (curlen != prevlen) blTree[curlen].fc++; blTree[REP_3_6].fc++; } // Alternative conditional branch
            else if (count <= 10) blTree[REPZ_3_10].fc++; else blTree[REPZ_11_138].fc++; // Alternative conditional branch
            count = 0; prevlen = curlen; // Compute and assign value
            if (nextlen == 0) { maxCount = 138; minCount = 3; } else if (curlen == nextlen) { maxCount = 6; minCount = 3; } else { maxCount = 7; minCount = 4; } // Conditional branch check
        } // Close scope / block
    } // Close scope / block
    void sendTree(CT* tree, int maxCode) { // Execute statement
        int prevlen = -1, curlen, nextlen = tree[0].dl, count = 0, maxCount = 7, minCount = 4; // Compute and assign value
        if (nextlen == 0) { maxCount = 138; minCount = 3; } // Conditional branch check
        for (int n = 0; n <= maxCode; n++) { // Loop iteration
            curlen = nextlen; nextlen = tree[n + 1].dl; // Compute and assign value
            if (++count < maxCount && curlen == nextlen) continue; // Conditional branch check
            else if (count < minCount) { do { sendCode(curlen, blTree); } while (--count != 0); } // Alternative conditional branch
            else if (curlen != 0) { if (curlen != prevlen) { sendCode(curlen, blTree); count--; } sendCode(REP_3_6, blTree); sendBits((unsigned)(count - 3), 2); } // Alternative conditional branch
            else if (count <= 10) { sendCode(REPZ_3_10, blTree); sendBits((unsigned)(count - 3), 3); } // Alternative conditional branch
            else { sendCode(REPZ_11_138, blTree); sendBits((unsigned)(count - 11), 7); } // Fallback branch
            count = 0; prevlen = curlen; // Compute and assign value
            if (nextlen == 0) { maxCount = 138; minCount = 3; } else if (curlen == nextlen) { maxCount = 6; minCount = 3; } else { maxCount = 7; minCount = 4; } // Conditional branch check
        } // Close scope / block
    } // Close scope / block
    int buildBlTree() { // Execute statement
        scanTree(dynL, lDesc.maxCode); scanTree(dynD, dDesc.maxCode); buildTree(blDesc); // Execute statement
        int mb; for (mb = BL_CODES - 1; mb >= 3; mb--) if (blTree[kBlOrder[mb]].dl != 0) break; // Compute and assign value
        optLen += 3 * ((u64)mb + 1) + 5 + 5 + 4; return mb; // Compute and assign value
    } // Close scope / block
    void sendAllTrees(int lcodes, int dcodes, int blcodes) { // Execute statement
        sendBits((unsigned)(lcodes - 257), 5); sendBits((unsigned)(dcodes - 1), 5); sendBits((unsigned)(blcodes - 4), 4); // Execute statement
        for (int r = 0; r < blcodes; r++) sendBits(blTree[kBlOrder[r]].dl, 3); // Loop iteration
        sendTree(dynL, lcodes - 1); sendTree(dynD, dcodes - 1); // Execute statement
    } // Close scope / block
    inline int dCode(unsigned dist) const { return dist < 256 ? T.distCode[dist] : T.distCode[256 + (dist >> 7)]; } // Define helper function or constant
    void compressBlock(const CT* ltree, const CT* dtree) { // Execute statement
        unsigned lx = 0; // Compute and assign value
        if (lastLit != 0) do { // Conditional branch check
            unsigned dist = dBuf[lx]; int lc = lBuf[lx++]; // Compute and assign value
            if (dist == 0) sendCode(lc, ltree); // Conditional branch check
            else { // Fallback branch
                unsigned code = T.lengthCode[lc]; sendCode((int)code + LITERALS + 1, ltree); // Compute and assign value
                int extra = kExtraL[code]; if (extra != 0) { lc -= T.baseLength[code]; sendBits((unsigned)lc, extra); } // Compute and assign value
                dist--; code = (unsigned)dCode(dist); sendCode((int)code, dtree); // Compute and assign value
                extra = kExtraD[code]; if (extra != 0) { dist -= (unsigned)T.baseDist[code]; sendBits(dist, extra); } // Compute and assign value
            } // Close scope / block
        } while (lx < lastLit); // Execute statement
        sendCode(END_BLOCK, ltree); // Execute statement
    } // Close scope / block
    void storedBlock(const u8* buf, u64 len, int last) { // Execute statement
        sendBits((unsigned)(0 << 1) + (unsigned)last, 3); windup(); // Execute statement
        putByte((u8)len); putByte((u8)(len >> 8)); putByte((u8)~len); putByte((u8)(~len >> 8)); // Execute statement
        out.insert(out.end(), buf, buf + len); // Execute statement
    } // Close scope / block
    void flushBlockTrees(const u8* buf, u64 storedLen, int last) { // Execute statement
        u64 optLenb, staticLenb; int maxBl = 0; // Compute and assign value
        buildTree(lDesc); buildTree(dDesc); maxBl = buildBlTree(); // Compute and assign value
        optLenb = (optLen + 3 + 7) >> 3; staticLenb = (staticLen + 3 + 7) >> 3; if (staticLenb <= optLenb) optLenb = staticLenb; // Compute and assign value
        if (storedLen + 4 <= optLenb && buf != nullptr) storedBlock(buf, storedLen, last); // Conditional branch check
        else if (staticLenb == optLenb) { sendBits((unsigned)((1 << 1) + last), 3); compressBlock(T.sl, T.sd); } // Alternative conditional branch
        else { sendBits((unsigned)((2 << 1) + last), 3); sendAllTrees(lDesc.maxCode + 1, dDesc.maxCode + 1, maxBl + 1); compressBlock(dynL, dynD); } // Fallback branch
        initBlock(); if (last) windup(); // Execute statement
    } // Close scope / block
    inline bool tallyLit(u8 c) { dBuf[lastLit] = 0; lBuf[lastLit++] = c; dynL[c].fc++; return lastLit == litBufSize - 1; } // Define helper function or constant
    inline bool tallyDist(unsigned dist, unsigned lenMinus3) { // Define helper function or constant
        dBuf[lastLit] = (u16)dist; lBuf[lastLit++] = (u8)lenMinus3; matches++; dist--; // Compute and assign value
        dynL[T.lengthCode[lenMinus3] + LITERALS + 1].fc++; dynD[dCode(dist)].fc++; return lastLit == litBufSize - 1; // Compute and assign value
    } // Close scope / block
    void flushBlock(int last) { // Execute statement
        flushBlockTrees(blockStart >= 0 ? &window[(size_t)blockStart] : nullptr, (u64)((long)strStart - blockStart), last); // Compute and assign value
        blockStart = (long)strStart; check(); // Compute and assign value
    } // Close scope / block
    // ---- matching
    inline void updateHash(unsigned c) { insH = ((insH << hashShift) ^ c) & hashMask; } // Define helper function or constant
    inline unsigned insertString(unsigned str) { updateHash(window[str + (MIN_MATCH - 1)]); unsigned h = head[insH]; prev[str & wMask] = (u16)h; head[insH] = (u16)str; return h; } // Define helper function or constant
    void slideHash() { // Execute statement
        for (unsigned i = 0; i < hashSize; i++) { unsigned m = head[i]; head[i] = (u16)(m >= wSize ? m - wSize : 0); } // Loop iteration
        for (unsigned i = 0; i < wSize; i++) { unsigned m = prev[i]; prev[i] = (u16)(m >= wSize ? m - wSize : 0); } // Loop iteration
    } // Close scope / block
    void fillWindow() { // Execute statement
        unsigned more, wsize = wSize; // Compute and assign value
        do { // Execute statement
            more = (unsigned)(windowSize - (size_t)lookahead - (size_t)strStart); // Compute and assign value
            if (strStart >= wsize + maxDist()) { // Conditional branch check
                std::memcpy(window.data(), window.data() + wsize, (size_t)wsize - more); // Fast raw memory copy
                matchStart -= wsize; strStart -= wsize; blockStart -= (long)wsize; slideHash(); more += wsize; // Compute and assign value
            } // Close scope / block
            if (availIn == 0) break; // Conditional branch check
            unsigned len = (unsigned)std::min<size_t>(availIn, more); // Compute and assign value
            std::memcpy(window.data() + strStart + lookahead, nextIn, len); nextIn += len; availIn -= len; lookahead += len; // Fast raw memory copy
            if (lookahead + insert >= MIN_MATCH) { // Conditional branch check
                unsigned str = strStart - insert; insH = window[str]; updateHash(window[str + 1]); // Compute and assign value
                while (insert) { updateHash(window[str + MIN_MATCH - 1]); prev[str & wMask] = head[insH]; head[insH] = (u16)str; str++; insert--; if (lookahead + insert < MIN_MATCH) break; } // Loop while condition holds
            } // Close scope / block
        } while (lookahead < MIN_LOOKAHEAD && availIn != 0); // Compute and assign value
        if (highWater < windowSize) { // Conditional branch check
            size_t curr = (size_t)strStart + lookahead, init; // Compute and assign value
            if (highWater < curr) { init = windowSize - curr; if (init > MAX_MATCH) init = MAX_MATCH; std::memset(window.data() + curr, 0, init); highWater = curr + init; } // Conditional branch check
            else if (highWater < curr + MAX_MATCH) { init = curr + MAX_MATCH - highWater; if (init > windowSize - highWater) init = windowSize - highWater; std::memset(window.data() + highWater, 0, init); highWater += init; } // Alternative conditional branch
        } // Close scope / block
    } // Close scope / block
    unsigned longestMatch(unsigned curMatch) { // Execute statement
        unsigned chain = maxChain; const u8* scan = window.data() + strStart; int bestLen = (int)prevLength; int nice = (int)niceMatch; // Compute and assign value
        unsigned limit = strStart > maxDist() ? strStart - maxDist() : 0; const u16* pv = prev.data(); unsigned wm = wMask; // Compute and assign value
        u8 scanEnd1 = scan[bestLen - 1], scanEnd = scan[bestLen]; // Compute and assign value
        if (prevLength >= goodMatch) chain >>= 2; // Conditional branch check
        if ((unsigned)nice > lookahead) nice = (int)lookahead; // Conditional branch check
        do { // Execute statement
            const u8* match = window.data() + curMatch; // Compute and assign value
            if (match[bestLen] != scanEnd || match[bestLen - 1] != scanEnd1 || match[0] != scan[0] || match[1] != scan[1]) continue; // Conditional branch check
            int len = 3; while (len < MAX_MATCH && scan[len] == match[len]) len++;       // scan[2] == match[2] is guaranteed by the hash
            if (len > bestLen) { matchStart = curMatch; bestLen = len; if (len >= nice) break; scanEnd1 = scan[bestLen - 1]; scanEnd = scan[bestLen]; } // Conditional branch check
        } while ((curMatch = pv[curMatch & wm]) > limit && --chain != 0); // Compute and assign value
        if ((unsigned)bestLen <= lookahead) return (unsigned)bestLen; // Conditional branch check
        return lookahead; // Return result to caller
    } // Close scope / block
    void deflateFast() { // Execute statement
        for (;;) { // Loop iteration
            if (bad) return; // Conditional branch check
            if (lookahead < MIN_LOOKAHEAD) { fillWindow(); if (lookahead == 0) break; } // Conditional branch check
            unsigned hashHead = 0; // Compute and assign value
            if (lookahead >= MIN_MATCH) hashHead = insertString(strStart); // Conditional branch check
            if (hashHead != 0 && strStart - hashHead <= maxDist()) matchLength = longestMatch(hashHead); // Conditional branch check
            bool bflush; // Execute statement
            if (matchLength >= MIN_MATCH) { // Conditional branch check
                bflush = tallyDist(strStart - matchStart, matchLength - MIN_MATCH); // Compute and assign value
                lookahead -= matchLength; // Compute and assign value
                if (matchLength <= maxLazy && lookahead >= MIN_MATCH) { // Conditional branch check
                    matchLength--; // Execute statement
                    do { strStart++; insertString(strStart); } while (--matchLength != 0); // Compute and assign value
                    strStart++; // Execute statement
                } else { strStart += matchLength; matchLength = 0; insH = window[strStart]; updateHash(window[strStart + 1]); } // Compute and assign value
            } else { bflush = tallyLit(window[strStart]); lookahead--; strStart++; } // Compute and assign value
            if (bflush) flushBlock(0); // Conditional branch check
        } // Close scope / block
        insert = strStart < MIN_MATCH - 1 ? strStart : MIN_MATCH - 1; // Compute and assign value
        flushBlock(1); // Execute statement
    } // Close scope / block
    void deflateSlow() { // Execute statement
        for (;;) { // Loop iteration
            if (bad) return; // Conditional branch check
            if (lookahead < MIN_LOOKAHEAD) { fillWindow(); if (lookahead == 0) break; } // Conditional branch check
            unsigned hashHead = 0; // Compute and assign value
            if (lookahead >= MIN_MATCH) hashHead = insertString(strStart); // Conditional branch check
            prevLength = matchLength; prevMatch = matchStart; matchLength = MIN_MATCH - 1; // Compute and assign value
            if (hashHead != 0 && prevLength < maxLazy && strStart - hashHead <= maxDist()) { // Conditional branch check
                matchLength = longestMatch(hashHead); // Compute and assign value
                if (matchLength <= 5 && (strategy == 1 || (matchLength == MIN_MATCH && strStart - matchStart > TOO_FAR))) matchLength = MIN_MATCH - 1; // Conditional branch check
            } // Close scope / block
            if (prevLength >= MIN_MATCH && matchLength <= prevLength) { // Conditional branch check
                unsigned maxInsert = strStart + lookahead - MIN_MATCH; // Compute and assign value
                bool bflush = tallyDist(strStart - 1 - prevMatch, prevLength - MIN_MATCH); // Compute and assign value
                lookahead -= prevLength - 1; prevLength -= 2; // Compute and assign value
                do { if (++strStart <= maxInsert) insertString(strStart); } while (--prevLength != 0); // Compute and assign value
                matchAvailable = 0; matchLength = MIN_MATCH - 1; strStart++; // Compute and assign value
                if (bflush) flushBlock(0); // Conditional branch check
            } else if (matchAvailable) { // Execute statement
                bool bflush = tallyLit(window[strStart - 1]); // Compute and assign value
                if (bflush) flushBlock(0); // Conditional branch check
                strStart++; lookahead--; // Execute statement
            } else { matchAvailable = 1; strStart++; lookahead--; } // Compute and assign value
        } // Close scope / block
        if (matchAvailable) { tallyLit(window[strStart - 1]); matchAvailable = 0; } // Conditional branch check
        insert = strStart < MIN_MATCH - 1 ? strStart : MIN_MATCH - 1; // Compute and assign value
        flushBlock(1); // Execute statement
    } // Close scope / block
    void deflateHuff() { // Execute statement
        for (;;) { // Loop iteration
            if (bad) return; // Conditional branch check
            if (lookahead == 0) { fillWindow(); if (lookahead == 0) break; } // Conditional branch check
            matchLength = 0; bool bflush = tallyLit(window[strStart]); lookahead--; strStart++; // Compute and assign value
            if (bflush) flushBlock(0); // Conditional branch check
        } // Close scope / block
        insert = 0; flushBlock(1); // Compute and assign value
    } // Close scope / block
    void deflateRle() { // Execute statement
        for (;;) { // Loop iteration
            if (bad) return; // Conditional branch check
            if (lookahead <= MAX_MATCH) { fillWindow(); if (lookahead == 0) break; } // Conditional branch check
            matchLength = 0; // Compute and assign value
            if (lookahead >= MIN_MATCH && strStart > 0) { // Conditional branch check
                const u8* w = window.data() + strStart; u8 pv = w[-1]; // Compute and assign value
                if (pv == w[0] && pv == w[1] && pv == w[2]) { unsigned len = 3; while (len < MAX_MATCH && pv == w[len]) len++; matchLength = len; if (matchLength > lookahead) matchLength = lookahead; } // Conditional branch check
            } // Close scope / block
            bool bflush; // Execute statement
            if (matchLength >= MIN_MATCH) { bflush = tallyDist(1, matchLength - MIN_MATCH); lookahead -= matchLength; strStart += matchLength; matchLength = 0; } // Conditional branch check
            else { bflush = tallyLit(window[strStart]); lookahead--; strStart++; } // Fallback branch
            if (bflush) flushBlock(0); // Conditional branch check
        } // Close scope / block
        insert = 0; flushBlock(1); // Compute and assign value
    } // Close scope / block
    void run(const u8* in, size_t n) { nextIn = in; availIn = n; if (strategy == 2) deflateHuff(); else if (strategy == 3) deflateRle(); else if (kCfg[level].slow) deflateSlow(); else deflateFast(); } // Compute and assign value
}; // Close scope / block
// Re-creates the raw deflate stream of `in`. expect != null: returns true only if the result equals expect (stops at the first difference).
// expect == null: the stream is appended to *result.
inline bool deflate_exact(const u8* in, size_t n, int level, int memLevel, int wBits, int strategy, const u8* expect, size_t expectLen, std::vector<u8>* result) { // Allocate / manage dynamic buffer
    if (level < 1 || level > 9 || memLevel < 1 || memLevel > 9 || wBits < 9 || wBits > 15 || strategy < 0 || strategy > 3) return false; // Conditional branch check
    std::unique_ptr<Def> d(new Def(level, memLevel, wBits, strategy)); d->expect = expect; d->expectLen = expectLen; d->run(in, n); // Compute and assign value
    if (expect) { d->check(); return !d->bad && d->emitted == expectLen; } // Conditional branch check
    if (result) result->insert(result->end(), d->out.begin(), d->out.end()); // Conditional branch check
    return true; // Return result to caller
} // Close scope / block
}  // namespace zl


// ------------------------------------------------------------------ deflate transform: inflate every stream zlib can reproduce exactly
// The file is scanned for zlib / gzip / ZIP-entry / PNG-IDAT streams. For each one that is inflated and then re-deflated by zl::Def
// to exactly the original bytes, the stream is replaced by its inflated data and a small record (level, memLevel, windowBits ...).
// The result (a normal byte stream, much more compressible) goes through the usual 3P model; extraction undoes it.
namespace zl { // Open namespace scope
inline u32 adler32(const u8* p, size_t n) { // Define helper function or constant
    u32 a = 1, b = 0; while (n) { size_t k = std::min<size_t>(n, 5552); n -= k; while (k--) { a += *p++; b += a; } a %= 65521; b %= 65521; } return (b << 16) | a; // Compute and assign value
} // Close scope / block
struct DefRec {                                     // one replaced stream
    u64 gap = 0;                                    // untouched bytes since the previous record (original coordinates)
    u64 rawLen = 0;                                 // inflated size
    u8 wrap = 0;                                    // 0 raw deflate, 1 zlib (2-byte header + adler32), 2 raw deflate + gzip trailer (crc32, isize)
    u8 zh0 = 0, zh1 = 0, level = 6, memLevel = 8, wbits = 15, strategy = 0; // Compute and assign value
    std::vector<u32> sb;                            // strategy 4: lengths of the stored blocks
    u8 pf = 0, pbpp = 0; u64 pstride = 0, prows = 0; std::vector<u8> filt;   // PNG pixels: stored unfiltered, one filter type per row kept here
    std::vector<u64> partLen, joinLen;              // PNG: one stream split over several IDAT chunks (all parts but the last, and the bytes between them)
}; // Close scope / block
struct PngInfo { int stride = 0, bpp = 0; bool pix = false; u64 phase = 0; };   // row geometry handed to the model (pix: unfiltered pixels, phase: stream offset of the first pixel byte)
struct Params { int level = 0, memLevel = 8, wbits = 15, strategy = 0; std::vector<u32> sb; };   // strategy 4: stored blocks only, sb = their lengths
struct Memo { Params last; bool have = false; int fails = 0, hits = 0; }; // Declare data structure / class

constexpr size_t kMaxRaw = (size_t)256 << 20; // Define helper function or constant
constexpr u64 kMaxX = (u64)3 << 30; // Define helper function or constant
constexpr size_t kMaxRecs = 200000; // Define helper function or constant

inline bool zlib_header_ok(const u8* p) { return (p[0] & 15) == 8 && (p[0] >> 4) <= 7 && ((p[0] << 8) | p[1]) % 31 == 0 && !(p[1] & 0x20); } // Define helper function or constant

inline u8 paeth(int a, int b, int c) { int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c); return (u8)((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c)); } // Define helper function or constant
// filtered scanlines (1 type byte + stride bytes per row) <-> pixel rows + filter types. Both directions are exact.
inline bool png_unfilter(const u8* f, size_t rows, size_t stride, size_t bpp, u8* pix, u8* filt) { // Define helper function or constant
    for (size_t r = 0; r < rows; r++) { // Loop iteration
        const u8* in = f + r * (stride + 1); u8* cur = pix + r * stride; const u8* up = r ? cur - stride : nullptr; u8 t = in[0]; filt[r] = t; in++; // Compute and assign value
        if (t > 4) return false; // Conditional branch check
        for (size_t i = 0; i < stride; i++) { // Loop iteration
            int a = i >= bpp ? cur[i - bpp] : 0, b = up ? up[i] : 0, c = (up && i >= bpp) ? up[i - bpp] : 0, x = in[i]; // Compute and assign value
            switch (t) { case 1: x += a; break; case 2: x += b; break; case 3: x += (a + b) >> 1; break; case 4: x += paeth(a, b, c); break; default: break; } // Multi-way dispatch switch
            cur[i] = (u8)x; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    return true; // Return result to caller
} // Close scope / block
inline void png_refilter(const u8* pix, size_t rows, size_t stride, size_t bpp, const u8* filt, u8* f) { // Define helper function or constant
    for (size_t r = 0; r < rows; r++) { // Loop iteration
        const u8* cur = pix + r * stride; const u8* up = r ? cur - stride : nullptr; u8 t = filt[r]; u8* o = f + r * (stride + 1); *o++ = t; // Compute and assign value
        for (size_t i = 0; i < stride; i++) { // Loop iteration
            int a = i >= bpp ? cur[i - bpp] : 0, b = up ? up[i] : 0, c = (up && i >= bpp) ? up[i - bpp] : 0, x = cur[i]; // Compute and assign value
            switch (t) { case 1: x -= a; break; case 2: x -= b; break; case 3: x -= (a + b) >> 1; break; case 4: x -= paeth(a, b, c); break; default: break; } // Multi-way dispatch switch
            o[i] = (u8)x; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
} // Close scope / block
inline bool build_stored(const u8* raw, size_t n, const std::vector<u32>& lens, std::vector<u8>& out) { // Allocate / manage dynamic buffer
    size_t at = 0; // Compute and assign value
    for (size_t i = 0; i < lens.size(); i++) { // Loop iteration
        u32 l = lens[i]; if (l > 65535 || (size_t)l > n - at) return false; // Compute and assign value
        out.push_back(i + 1 == lens.size() ? 1 : 0); out.push_back((u8)l); out.push_back((u8)(l >> 8)); out.push_back((u8)~l); out.push_back((u8)(~l >> 8)); // Append element to container
        out.insert(out.end(), raw + at, raw + at + l); at += l; // Compute and assign value
    } // Close scope / block
    return at == n; // Return result to caller
} // Close scope / block
// p: start of the region (zlib header / deflate body). Returns true when the region is a complete stream that can be rebuilt exactly.
inline bool probe(const u8* p, size_t avail, int wrap, std::vector<u8>& raw, size_t& regionLen, Params& P, Memo& memo, bool& parsedOk, bool preferFiltered = false) { // Allocate / manage dynamic buffer
    parsedOk = false; // Compute and assign value
    size_t hdr = wrap == 1 ? 2 : 0, used = 0; // Compute and assign value
    if (avail < hdr + 8) return false; // Conditional branch check
    if (wrap == 1 && !zlib_header_ok(p)) return false; // Conditional branch check
    std::vector<u32> sl; bool coded = false; // Allocate / manage dynamic buffer
    if (!inflate_raw(p + hdr, avail - hdr, used, raw, kMaxRaw, &sl, &coded)) return false; // Conditional branch check
    size_t trail = wrap == 1 ? 4 : wrap == 2 ? 8 : 0; // Compute and assign value
    if (hdr + used + trail > avail) return false; // Conditional branch check
    const u8* t = p + hdr + used; // Compute and assign value
    if (wrap == 1) { u32 a = ((u32)t[0] << 24) | ((u32)t[1] << 16) | ((u32)t[2] << 8) | t[3]; if (a != adler32(raw.data(), raw.size())) return false; } // Conditional branch check
    if (wrap == 2) { if (rd32le(t) != crc32(raw.data(), raw.size()) || rd32le(t + 4) != (u32)raw.size()) return false; } // Conditional branch check
    regionLen = hdr + used + trail; parsedOk = true; // Compute and assign value
    if (regionLen < 64 || raw.size() < 32) return false; // Conditional branch check
    if (raw.size() > (size_t)regionLen * 512 + ((size_t)1 << 20)) return false;                 // absurd ratios: not worth the model time
    const u8* body = p + hdr; size_t bodyLen = used; // Compute and assign value
    std::vector<std::pair<int, int>> cand;                            // (level, strategy) in the order they are tried
    std::vector<int> wbs; const int mems[2] = {8, 9}; // Allocate / manage dynamic buffer
    auto addLevel = [&](int lv) { if (preferFiltered) { cand.push_back({lv, 1}); cand.push_back({lv, 0}); } else { cand.push_back({lv, 0}); cand.push_back({lv, 1}); } }; // Append element to container
    if (wrap == 1) { // Conditional branch check
        int fl = p[1] >> 6; wbs.push_back((p[0] >> 4) + 8); // Append element to container
        if (wbs[0] < 9) return false; // Conditional branch check
        if (fl == 0) { addLevel(1); cand.push_back({1, 2}); cand.push_back({1, 3}); } // Conditional branch check
        else if (fl == 1) { for (int lv = 2; lv <= 5; lv++) addLevel(lv); } // Alternative conditional branch
        else if (fl == 2) addLevel(6); // Alternative conditional branch
        else { for (int lv = 7; lv <= 9; lv++) addLevel(lv); } // Fallback branch
    } else { wbs.push_back(15); for (int lv : {6, 9, 1, 4, 5, 7, 8, 2, 3}) addLevel(lv); cand.push_back({1, 2}); cand.push_back({1, 3}); } // Append element to container
    auto tryP = [&](int lv, int st, int ml, int wb) { return deflate_exact(raw.data(), raw.size(), lv, ml, wb, st, body, bodyLen, nullptr); }; // Compute and assign value
    if (memo.have && memo.last.wbits == wbs[0] && std::find(cand.begin(), cand.end(), std::make_pair(memo.last.level, memo.last.strategy)) != cand.end() && tryP(memo.last.level, memo.last.strategy, memo.last.memLevel, memo.last.wbits)) { P = memo.last; return true; } // Conditional branch check
    for (auto& c : cand) for (int ml : mems) for (int wb : wbs) { // Loop iteration
        if (memo.have && c.first == memo.last.level && c.second == memo.last.strategy && ml == memo.last.memLevel && wb == memo.last.wbits) continue; // Conditional branch check
        if (tryP(c.first, c.second, ml, wb)) { P.level = c.first; P.strategy = c.second; P.memLevel = ml; P.wbits = wb; P.sb.clear(); memo.last = P; memo.have = true; return true; } // Conditional branch check
    } // Close scope / block
    if (!coded && !sl.empty() && (wrap != 1 || (p[1] >> 6) == 0)) {                               // level 0: nothing but stored blocks
        std::vector<u8> st; if (build_stored(raw.data(), raw.size(), sl, st) && st.size() == bodyLen && std::memcmp(st.data(), body, bodyLen) == 0) { // Allocate / manage dynamic buffer
            P.level = 1; P.strategy = 4; P.memLevel = 8; P.wbits = wbs[0]; P.sb = sl; return true; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    return false; // Return result to caller
} // Close scope / block

inline void put_rec(std::vector<u8>& v, const DefRec& r) { // Allocate / manage dynamic buffer
    put_varint(v, r.gap); put_varint(v, r.rawLen); v.push_back((u8)(r.wrap | (r.pf << 4))); if (r.wrap == 1) { v.push_back(r.zh0); v.push_back(r.zh1); } // Append element to container
    v.push_back((u8)(r.level | (r.memLevel << 4))); v.push_back((u8)(r.wbits | (r.strategy << 4))); // Append element to container
    if (r.strategy == 4) { std::vector<std::pair<u32, u64>> runs; for (u32 l : r.sb) { if (!runs.empty() && runs.back().first == l) runs.back().second++; else runs.push_back({l, 1}); } put_varint(v, runs.size()); for (auto& q : runs) { put_varint(v, q.first); put_varint(v, q.second); } } // Conditional branch check
    if (r.pf) { put_varint(v, r.pstride); v.push_back(r.pbpp); put_varint(v, r.prows); v.insert(v.end(), r.filt.begin(), r.filt.end()); } // Conditional branch check
    put_varint(v, r.partLen.size() + 1); for (size_t i = 0; i < r.partLen.size(); i++) { put_varint(v, r.partLen[i]); put_varint(v, r.joinLen[i]); } // Encode / decode variable-length integer
} // Close scope / block
inline bool get_varint_mem(const u8*& p, const u8* e, u64& v) { v = 0; int sh = 0; for (;;) { if (p >= e || sh > 56) return false; u8 c = *p++; v |= (u64)(c & 127) << sh; if (!(c & 128)) return true; sh += 7; } } // Encode / decode variable-length integer
inline bool parse_recs(const std::vector<u8>& t, u64 count, std::vector<DefRec>& recs) { // Allocate / manage dynamic buffer
    const u8* p = t.data(); const u8* e = p + t.size(); recs.clear(); if (count > kMaxRecs) return false; // Compute and assign value
    for (u64 i = 0; i < count; i++) { // Loop iteration
        DefRec r; u64 np = 0; // Compute and assign value
        if (!get_varint_mem(p, e, r.gap) || !get_varint_mem(p, e, r.rawLen) || p >= e) return false; // Conditional branch check
        r.wrap = *p++; r.pf = r.wrap >> 4; r.wrap &= 15; if (r.wrap > 2 || r.pf > 1) return false; // Compute and assign value
        if (r.wrap == 1) { if (e - p < 2) return false; r.zh0 = *p++; r.zh1 = *p++; } // Conditional branch check
        if (e - p < 2) return false; // Conditional branch check
        u8 lm = *p++; u8 wsb = *p++; r.level = lm & 15; r.memLevel = lm >> 4; r.wbits = wsb & 15; r.strategy = wsb >> 4; // Compute and assign value
        if (r.level < 1 || r.level > 9 || r.memLevel < 1 || r.memLevel > 9 || r.wbits < 9 || r.wbits > 15 || r.strategy > 4 || r.rawLen > kMaxRaw) return false; // Conditional branch check
        if (r.strategy == 4) { // Conditional branch check
            u64 nr = 0; if (!get_varint_mem(p, e, nr) || nr < 1 || nr > ((u64)1 << 20)) return false; // Encode / decode variable-length integer
            for (u64 k = 0; k < nr; k++) { u64 l, c; if (!get_varint_mem(p, e, l) || !get_varint_mem(p, e, c) || l > 65535 || c < 1 || r.sb.size() + c > ((u64)1 << 20)) return false; r.sb.insert(r.sb.end(), (size_t)c, (u32)l); } // Loop iteration
        } // Close scope / block
        if (r.pf) { // Conditional branch check
            if (!get_varint_mem(p, e, r.pstride) || p >= e) return false; // Conditional branch check
            r.pbpp = *p++; if (!get_varint_mem(p, e, r.prows)) return false; // Encode / decode variable-length integer
            if (r.pstride < 1 || r.pstride > ((u64)1 << 26) || r.prows < 1 || r.pbpp < 1 || r.pbpp > 8 || r.prows > kMaxRaw || r.pstride * r.prows != r.rawLen || (u64)(e - p) < r.prows) return false; // Conditional branch check
            r.filt.assign(p, p + r.prows); p += r.prows; for (u8 ft : r.filt) if (ft > 4) return false; // Compute and assign value
        } // Close scope / block
        if (!get_varint_mem(p, e, np) || np < 1 || np > 100000) return false; // Conditional branch check
        for (u64 k = 0; k + 1 < np; k++) { u64 a, b; if (!get_varint_mem(p, e, a) || !get_varint_mem(p, e, b) || b > 4096) return false; r.partLen.push_back(a); r.joinLen.push_back(b); } // Loop iteration
        recs.push_back(std::move(r)); // Append element to container
    } // Close scope / block
    return p == e; // Return result to caller
} // Close scope / block
inline u32 be32(const u8* p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3]; } // Define helper function or constant
inline u32 le16(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8); } // Define helper function or constant

// Scans d[0..n), writes the transformed stream to xf. Returns 1 = at least one stream replaced, 0 = nothing found, -1 = cancelled.
inline int transform(const u8* d, size_t n, FILE* xf, std::vector<DefRec>& recs, u64& xTotal, PngInfo& pi, const std::function<bool(double)>& prog) { // Allocate / manage dynamic buffer
    recs.clear(); xTotal = 0; pi = PngInfo(); Memo memo; size_t pos = 0; std::vector<u8> raw, cat; bool giveUp = false; // Allocate / manage dynamic buffer
    auto put = [&](const u8* p, size_t k) { if (k && std::fwrite(p, 1, k, xf) != k) throw Failure{Status::DiskFull, "cannot write the temporary file (disk full?)"}; xTotal += k; }; // Buffered file I/O operation
    std::vector<u8> pix; DefRec png; bool pngPix = true;                                      // png.pf != 0: pixel form of the current PNG stream
    auto accept = [&](size_t start, size_t end, const Params& P, int wrap, const u8* zh, const std::vector<std::pair<size_t, size_t>>* parts) { // Allocate / manage dynamic buffer
        const std::vector<u8>& body = png.pf ? pix : raw; // Allocate / manage dynamic buffer
        DefRec r; r.gap = start - pos; r.rawLen = body.size(); if (png.pf) { r.pf = 1; r.pstride = png.pstride; r.pbpp = png.pbpp; r.prows = png.prows; r.filt = png.filt; } r.wrap = (u8)wrap; if (wrap == 1) { r.zh0 = zh[0]; r.zh1 = zh[1]; } // Compute and assign value
        r.level = (u8)P.level; r.memLevel = (u8)P.memLevel; r.wbits = (u8)P.wbits; r.strategy = (u8)P.strategy; r.sb = P.sb; // Compute and assign value
        if (png.pf && !pi.stride) { pi.stride = (int)png.pstride; pi.bpp = png.pbpp; pi.pix = pngPix; pi.phase = xTotal + (start - pos); } // Conditional branch check
        put(d + pos, start - pos); put(body.data(), body.size()); // Execute statement
        if (parts) { size_t total = 0; for (auto& q : *parts) total += q.second - q.first; (void)total; // Conditional branch check
            for (size_t i = 0; i + 1 < parts->size(); i++) { r.partLen.push_back((*parts)[i].second - (*parts)[i].first); size_t j = (*parts)[i + 1].first - (*parts)[i].second; r.joinLen.push_back(j); put(d + (*parts)[i].second, j); } } // Loop iteration
        recs.push_back(std::move(r)); pos = end; // Append element to container
    }; // Close scope / block
    auto room = [&]() { return recs.size() < kMaxRecs && xTotal + raw.size() + n < kMaxX; }; // Compute and assign value
    size_t nextProg = 0; // Compute and assign value
    auto tick = [&](size_t i) { if (i >= nextProg) { nextProg = i + ((size_t)4 << 20); if (prog && !prog((double)i / (double)n)) return false; } return true; }; // Compute and assign value

    if (n >= 8 && std::memcmp(d, "\x89PNG\r\n\x1a\n", 8) == 0) {                         // PNG: the IDAT chunks hold one zlib stream
        size_t off = 8; // Compute and assign value
        while (off + 12 <= n && !giveUp) { // Loop while condition holds
            u32 len = be32(d + off); bool isIdat = std::memcmp(d + off + 4, "IDAT", 4) == 0; // Compute and assign value
            if ((u64)off + 12 + len > n) break; // Conditional branch check
            if (!isIdat) { off += 12 + (size_t)len; continue; } // Conditional branch check
            std::vector<std::pair<size_t, size_t>> parts; cat.clear(); size_t o = off; // Allocate / manage dynamic buffer
            while (o + 12 <= n && std::memcmp(d + o + 4, "IDAT", 4) == 0) { // Loop while condition holds
                u32 l = be32(d + o); if ((u64)o + 12 + l > n) break; // Compute and assign value
                parts.push_back({o + 8, o + 8 + (size_t)l}); cat.insert(cat.end(), d + o + 8, d + o + 8 + l); o += 12 + (size_t)l; // Append element to container
            } // Close scope / block
            off = o; // Compute and assign value
            if (!tick(off)) return -1; // Conditional branch check
            size_t regionLen = 0; Params P; bool parsed = false; // Compute and assign value
            if (probe(cat.data(), cat.size(), 1, raw, regionLen, P, memo, parsed, true) && regionLen == cat.size() && room()) { // Conditional branch check
                png = DefRec(); pix.clear(); // Compute and assign value
                if (n >= 33 && be32(d + 8) == 13 && std::memcmp(d + 12, "IHDR", 4) == 0) {            // pixel form: undo the row filters (needs a plain, non-interlaced image)
                    u64 w = be32(d + 16), hh = be32(d + 20); int depth = d[24], ct = d[25], il = d[28]; // Compute and assign value
                    int ch = ct == 0 ? 1 : ct == 2 ? 3 : ct == 3 ? 1 : ct == 4 ? 2 : ct == 6 ? 4 : 0; // Compute and assign value
                    if (ch && il == 0 && w > 0 && hh > 0 && w < ((u64)1 << 24) && hh < ((u64)1 << 24) && (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)) { // Conditional branch check
                        u64 bits = w * (u64)ch * (u64)depth, stride = (bits + 7) / 8; size_t bpp = std::max<size_t>(1, (size_t)(ch * depth / 8)); // Compute and assign value
                        if (stride < ((u64)1 << 26) && hh * (stride + 1) == raw.size()) { // Conditional branch check
                            pix.resize((size_t)(hh * stride)); png.filt.resize((size_t)hh); // Execute statement
                            if (png_unfilter(raw.data(), (size_t)hh, (size_t)stride, bpp, pix.data(), png.filt.data())) { // Conditional branch check
                                // which form does the model like best?  A: filtered bytes   B: filtered bytes + row geometry   C: unfiltered pixels + row geometry
                                auto trial = [&](const u8* data, size_t L, int st, int bp, bool px = false) { Hint hn; hn.stride = st; hn.bpp = bp; hn.pix = px; std::vector<u8> o; Counters cn; encode_segment(data, L, 20, hn, o, cn); return std::min(o.size(), L); }; // Allocate / manage dynamic buffer
                                size_t rowsT = (size_t)std::min<u64>(hh, std::max<u64>(1, ((u64)(P3_FAST_TRIALS ? 64 : 768) << 10) / (stride + 1))); // Compute and assign value
                                size_t A = trial(raw.data(), rowsT * (size_t)(stride + 1), 0, 0), B = trial(raw.data(), rowsT * (size_t)(stride + 1), (int)stride + 1, (int)bpp), C = trial(pix.data(), rowsT * (size_t)stride, (int)stride, (int)bpp, true), D = trial(pix.data(), rowsT * (size_t)stride, (int)stride, (int)bpp, false); // Compute and assign value
                                if (!pi.stride) { // Conditional branch check
                                    size_t P = std::min(C, D); // Compute and assign value
                                    if (P + P / 50 < A && P <= B) { png.pf = 1; png.pstride = stride; png.pbpp = (u8)bpp; png.prows = hh; pngPix = C <= D; } // Conditional branch check
                                    else if (B + B / 50 < A) { pi.stride = (int)stride + 1; pi.bpp = (int)bpp; } // Alternative conditional branch
                                } // Close scope / block
                            } // Close scope / block
                        } // Close scope / block
                    } // Close scope / block
                } // Close scope / block
                accept(parts.front().first, parts.back().second, P, 1, cat.data(), &parts); // Execute statement
                png = DefRec(); pix.clear(); // Compute and assign value
            } // Close scope / block
        } // Close scope / block
    } else { // Execute statement
        size_t i = 0; // Compute and assign value
        while (i + 8 < n && !giveUp) { // Loop while condition holds
            if (!tick(i)) return -1; // Conditional branch check
            u8 b = d[i]; size_t regionLen = 0; Params P; bool parsed = false; int wrap = -1; size_t bodyOff = i; // Compute and assign value
            if (b == 'P' && d[i + 1] == 'K' && d[i + 2] == 3 && d[i + 3] == 4 && i + 30 <= n) { // Conditional branch check
                u32 flags = le16(d + i + 6), method = le16(d + i + 8), nl = le16(d + i + 26), el = le16(d + i + 28); // Compute and assign value
                if (method == 8 && !(flags & 1) && i + 30 + nl + el < n) { wrap = 0; bodyOff = i + 30 + nl + el; } // Conditional branch check
                else { i += 4; continue; } // Fallback branch
            } else if (b == 0x1f && d[i + 1] == 0x8b && d[i + 2] == 8) { // Compute and assign value
                u32 fl = d[i + 3]; size_t q = i + 10; // Compute and assign value
                if (fl & 4) { if (q + 2 > n) { i++; continue; } q += 2 + le16(d + q); } // Conditional branch check
                if (fl & 8) { while (q < n && d[q]) q++; q++; } // Conditional branch check
                if (fl & 16) { while (q < n && d[q]) q++; q++; } // Conditional branch check
                if (fl & 2) q += 2; // Conditional branch check
                if (q >= n) { i++; continue; } // Conditional branch check
                wrap = 2; bodyOff = q; // Compute and assign value
            } else if ((b & 15) == 8 && (b >> 4) <= 7 && zlib_header_ok(d + i)) { wrap = 1; bodyOff = i; } // Compute and assign value
            else { i++; continue; } // Fallback branch
            bool ok = probe(d + bodyOff, n - bodyOff, wrap, raw, regionLen, P, memo, parsed); // Compute and assign value
            if (ok && room()) { accept(bodyOff, bodyOff + regionLen, P, wrap, d + bodyOff, nullptr); memo.fails = 0; memo.hits++; i = bodyOff + regionLen; } // Conditional branch check
            else if (parsed) { if (regionLen >= 256 && !memo.hits && ++memo.fails >= 24) giveUp = true; i = bodyOff + regionLen; } // Alternative conditional branch
            else i = (wrap == 0 || wrap == 2) ? i + 4 : i + 1; // Fallback branch
        } // Close scope / block
    } // Close scope / block
    if (recs.empty()) return 0; // Conditional branch check
    put(d + pos, n - pos); // Execute statement
    return 1; // Return result to caller
} // Close scope / block

// Rebuilds the original file from the transformed stream. emit() receives the output in order.
inline void rebuild(FILE* xin, u64 xTotal, const std::vector<DefRec>& recs, const std::function<void(const u8*, size_t)>& emit) { // Allocate / manage dynamic buffer
    u64 consumed = 0; std::vector<u8> raw, body, buf((size_t)1 << 20); // Allocate / manage dynamic buffer
    auto rd = [&](u8* p, size_t k) { if (k && std::fread(p, 1, k, xin) != k) throw Failure{Status::Corrupt, "file is truncated or damaged"}; consumed += k; }; // Buffered file I/O operation
    auto copy = [&](u64 k) { while (k) { size_t c = (size_t)std::min<u64>(k, buf.size()); rd(buf.data(), c); emit(buf.data(), c); k -= c; } }; // Compute and assign value
    for (const DefRec& r : recs) { // Loop iteration
        if (consumed + r.gap + r.rawLen > xTotal) throw Failure{Status::Corrupt, "file is damaged (stream table)"}; // Conditional branch check
        copy(r.gap); // Execute statement
        raw.resize((size_t)r.rawLen); rd(raw.data(), raw.size()); // Execute statement
        if (r.pf) { std::vector<u8> fl((size_t)(r.prows * (r.pstride + 1))); png_refilter(raw.data(), (size_t)r.prows, (size_t)r.pstride, r.pbpp, r.filt.data(), fl.data()); raw.swap(fl); } // Conditional branch check
        u64 joinTotal = 0; for (u64 j : r.joinLen) joinTotal += j; // Compute and assign value
        if (consumed + joinTotal > xTotal) throw Failure{Status::Corrupt, "file is damaged (stream table)"}; // Conditional branch check
        std::vector<u8> joins((size_t)joinTotal); rd(joins.data(), joins.size()); // Allocate / manage dynamic buffer
        body.clear(); // Execute statement
        if (r.wrap == 1) { body.push_back(r.zh0); body.push_back(r.zh1); } // Conditional branch check
        if (r.strategy == 4) { if (!build_stored(raw.data(), raw.size(), r.sb, body)) throw Failure{Status::Corrupt, "file is damaged (stored blocks)"}; } // Conditional branch check
        else if (!deflate_exact(raw.data(), raw.size(), r.level, r.memLevel, r.wbits, r.strategy, nullptr, 0, &body)) throw Failure{Status::Corrupt, "file is damaged (stream parameters)"}; // Alternative conditional branch
        if (r.wrap == 1) { u32 a = adler32(raw.data(), raw.size()); for (int s = 24; s >= 0; s -= 8) body.push_back((u8)(a >> s)); } // Conditional branch check
        if (r.wrap == 2) { u32 c = crc32(raw.data(), raw.size()), l = (u32)raw.size(); for (int s = 0; s < 32; s += 8) body.push_back((u8)(c >> s)); for (int s = 0; s < 32; s += 8) body.push_back((u8)(l >> s)); } // Conditional branch check
        size_t at = 0, jat = 0; // Compute and assign value
        for (size_t k = 0; k < r.partLen.size(); k++) { // Loop iteration
            if (r.partLen[k] > body.size() - at) throw Failure{Status::Corrupt, "file is damaged (stream split)"}; // Conditional branch check
            emit(body.data() + at, (size_t)r.partLen[k]); at += (size_t)r.partLen[k]; // Compute and assign value
            emit(joins.data() + jat, (size_t)r.joinLen[k]); jat += (size_t)r.joinLen[k]; // Compute and assign value
        } // Close scope / block
        emit(body.data() + at, body.size() - at); // Execute statement
    } // Close scope / block
    if (consumed > xTotal) throw Failure{Status::Corrupt, "file is damaged"}; // Conditional branch check
    copy(xTotal - consumed); // Execute statement
} // Close scope / block
}  // namespace zl

inline Hint make_hint(const Header& h, u64 segStart, size_t L, int audioCode) { // Define helper function or constant
    Hint t; t.stride = h.stride; t.bpp = h.bpp; // Compute and assign value
    if (h.defl && h.dStride) {                                                // PNG pixels
        t.stride = h.dStride; t.bpp = h.dBpp; // Compute and assign value
        if (h.dPix) { t.pix = true; u64 bp = (u64)h.dBpp; t.phase = (int)((segStart % bp + bp - h.dPhase % bp) % bp); } // Conditional branch check
    } // Close scope / block
    if (h.grid.on()) { t.stride = h.grid.stride; t.bpp = h.grid.width; } // Conditional branch check
    if (h.audio.on() && audioCode > 0) { // Conditional branch check
        size_t ts, te; audio_region(h.audio, segStart, L, ts, te); // Execute statement
        if (te > ts) { t.ats = ts; t.ate = te; t.channels = h.audio.channels; t.width = h.audio.width; } // Conditional branch check
    } // Close scope / block
    return t; // Return result to caller
} // Close scope / block
inline void remove_quiet(const fs::path& p) { std::error_code ec; fs::remove(p, ec); } // Define helper function or constant
struct TmpGuard { fs::path p; ~TmpGuard() { if (!p.empty()) { std::error_code ec; fs::remove(p, ec); } } }; // Declare data structure / class
inline Result fail(Status s, const std::string& m) { Result r; r.status = s; r.message = m; return r; } // Define helper function or constant

// shared state of one parallel run
struct Shared { // Declare data structure / class
    sys::Mutex m; sys::CondVar cv; // Execute statement
    u64 nextWrite = 0; unsigned alive = 0; bool failed = false; Status st = Status::Ok; std::string msg; // Compute and assign value
    std::atomic<u64> nextSeg{0}; Counters cn; bool userCancel = false; // Compute and assign value
    void error(Status s, const std::string& text) { sys::Lock l(m); if (!failed) { failed = true; st = s; msg = text; } cn.cancel = true; cv.notify_all(); } // Compute and assign value
}; // Close scope / block
// Memory per worker: ~24 MB (fast), ~64 MB (20-bit), ~120 MB (21-bit), ~200 MB (22-bit).
inline u64 per_thread_bytes(int tbMax) { // Define helper function or constant
    if (tbMax == 1) return (u64)24 << 20; // Conditional branch check
    if (tbMax <= 20) return (u64)64 << 20; // Conditional branch check
    if (tbMax == 21) return (u64)120 << 20; // Conditional branch check
    return (u64)200 << 20; // Return result to caller
} // Close scope / block
inline unsigned pick_threads(const Options& o, u64 segments, int tbMax) { // Define helper function or constant
    unsigned t = o.threads ? o.threads : std::min(sys::hardware_threads(), 16u); // Compute and assign value
    u64 budget = o.memoryBudget ? o.memoryBudget : ((u64)2 << 30); // Compute and assign value
    u64 cap = budget / per_thread_bytes(tbMax); t = (unsigned)std::min<u64>(t, std::max<u64>(1, cap)); // Compute and assign value
    if (t > segments) t = (unsigned)std::max<u64>(1, segments); // Conditional branch check
    return std::max(1u, t); // Return result to caller
} // Close scope / block
// compression: cache-aware table sizing to prevent DRAM-channel thrashing at high thread counts
inline void plan(const Options& o, u64 segments, unsigned& threads, int& tbMax) { // Define helper function or constant
    unsigned want = o.threads ? o.threads : std::min(sys::hardware_threads(), 16u); // Compute and assign value
    if (want > segments) want = (unsigned)std::max<u64>(1, segments); // Conditional branch check
    u64 budget = o.memoryBudget ? o.memoryBudget : ((u64)2 << 30); // Compute and assign value
    if (o.fast) { tbMax = 1; threads = (unsigned)std::min<u64>(want, std::max<u64>(1, budget / per_thread_bytes(1))); return; } // Conditional branch check
    int idealTb = (want >= 8) ? 20 : ((want >= 4) ? 21 : 22); // Compute and assign value
    for (int tb = idealTb; tb >= 19; tb--) { // Loop iteration
        if ((u64)want * per_thread_bytes(tb) <= budget) { threads = want; tbMax = tb; return; } // Conditional branch check
    } // Close scope / block
    tbMax = 20; threads = (unsigned)std::min<u64>(want, std::max<u64>(1, budget / per_thread_bytes(20))); // Compute and assign value
} // Close scope / block
// runs `worker` on n threads and polls progress on the calling thread until all are done
inline void run_workers(Shared& sh, unsigned n, u64 total, const std::function<void()>& worker, const ProgressFn& progress) { // Define helper function or constant
    std::vector<std::unique_ptr<sys::Thread>> th; // Allocate / manage dynamic buffer
    { sys::Lock l(sh.m); sh.alive = n; } // Compute and assign value
    for (unsigned i = 0; i < n; i++) { // Loop iteration
        th.emplace_back(new sys::Thread()); // Append element to container
        bool ok = th.back()->start([&sh, &worker] { // Compute and assign value
            try { worker(); } catch (const Cancelled&) {} catch (const Failure& f) { sh.error(f.status, f.msg); } // Begin exception-guarded block
            catch (const std::bad_alloc&) { sh.error(Status::OutOfMemory, "not enough memory"); } // Handle caught exception
            catch (const std::exception& e) { sh.error(Status::IoError, e.what()); } // Handle caught exception
            sys::Lock l(sh.m); sh.alive--; sh.cv.notify_all(); // Execute statement
        }); // Close scope / block
        if (!ok) { sys::Lock l(sh.m); sh.alive--; if (i == 0) { sh.failed = true; sh.st = Status::OutOfMemory; sh.msg = "cannot start worker threads"; } } // Conditional branch check
    } // Close scope / block
    for (;;) { // Loop iteration
        { sys::Lock l(sh.m); if (sh.alive == 0) break; sh.cv.wait_ms(sh.m, 100); } // Compute and assign value
        double f = total ? (double)sh.cn.done.load() / (double)total : 1.0; // Compute and assign value
        if (progress && !progress(f > 1.0 ? 1.0 : f, sh.cn.out.load())) { sys::Lock l(sh.m); sh.userCancel = true; sh.cn.cancel = true; sh.cv.notify_all(); } // Conditional branch check
    } // Close scope / block
    for (auto& t : th) t->join(); // Loop iteration
} // Close scope / block

}  // namespace detail

namespace detail { // Open namespace scope
// Encoder-side choice for one segment: 0 = plain bytes, 1..3 = audio presets 0..2. Trial-codes a sample of the
// PCM region each way (about 128 KB each) and keeps the smallest. Decoding never needs to repeat this.
inline int choose_audio_code(const Header& h, const u8* seg, size_t L, u64 segStart, int tbMax) { // Define helper function or constant
    size_t ts, te; audio_region(h.audio, segStart, L, ts, te); // Execute statement
    if (te <= ts) return 0; // Conditional branch check
    size_t frame = (size_t)h.audio.frame(), span = te - ts, want = std::min<size_t>(span, (size_t)(P3_FAST_TRIALS ? 32 : 128) << 10); // Compute and assign value
    want -= want % frame; if (want < frame * 256) return 0; // Compute and assign value
    size_t from = ts + ((span - want) / 2 / frame) * frame;           // sample from the middle of the region
    if (tbMax == 1) { // Conditional branch check
        // Fast mode: pick best LMS preset via split-byte order-0 entropy (< 0.3 ms instead of 4 full encodes)
        const size_t w = (size_t)h.audio.width, chs = (size_t)h.audio.channels, nSamp = want / w; // Compute and assign value
        auto splitEnt = [&](const u8* buf) { // Compute and assign value
            double total = 0.0; // Compute and assign value
            std::vector<u32> hiAcc(nSamp, 0); // Allocate / manage dynamic buffer
            for (int b = (int)w - 1; b >= 0; b--) { // Loop iteration
                u32 c[16][256] = {}; u32 ct[16] = {}; u32 avg[8] = {}; // Compute and assign value
                for (size_t k = 0; k < nSamp; k++) { // Loop iteration
                    size_t ch = k % chs; // Compute and assign value
                    u8 v = buf[k * w + (size_t)b]; // Compute and assign value
                    u32 bl = fast::fast_bitlen(avg[ch] >> 4); // Compute and assign value
                    u32 cx; // Execute statement
                    if (b == (int)w - 1) cx = std::min<u32>(15u, bl * 2u + (avg[ch] >= 16u ? 1u : 0u)); // Conditional branch check
                    else { u32 hi = hiAcc[k]; cx = (hi == 0) ? std::min<u32>(11u, bl) : (hi == 1 ? 12u : (hi == 2 ? 13u : (hi < 8 ? 14u : 15u))); } // Fallback branch
                    c[cx][v]++; ct[cx]++; // Execute statement
                    u32 curMag = (hiAcc[k] << 8) | (u32)v; // Compute and assign value
                    hiAcc[k] = curMag; // Compute and assign value
                    avg[ch] = (avg[ch] * 3u + (curMag << 4) + 2u) >> 2; // Compute and assign value
                } // Close scope / block
                double bits = 0.0; // Compute and assign value
                for (int cx = 0; cx < 16; cx++) if (ct[cx]) { // Loop iteration
                    double s = 0.0; // Compute and assign value
                    for (int i = 0; i < 256; i++) if (c[cx][i]) s += (double)c[cx][i] * std::log2((double)c[cx][i]); // Loop iteration
                    bits += (double)ct[cx] * std::log2((double)ct[cx]) - s; // Compute and assign value
                } // Close scope / block
                total += bits / (double)nSamp; // Compute and assign value
            } // Close scope / block
            return total; // Return result to caller
        }; // Close scope / block
        const double plainE = splitEnt(seg + from); // Compute and assign value
        double bestE = plainE; int code = 0; // Compute and assign value
        std::vector<u8> work(want); // Allocate / manage dynamic buffer
        for (int p = 0; p < 3; p++) { // Loop iteration
            std::memcpy(work.data(), seg + from, want); // Fast raw memory copy
            AudioCodec(h.audio.channels, h.audio.width, p).forward(work.data(), want / frame); // Execute statement
            double e = splitEnt(work.data()); // Compute and assign value
            if (e < bestE) { bestE = e; code = p + 1; } // Conditional branch check
        } // Close scope / block
        if (bestE + 0.12 >= plainE) code = 0; // Conditional branch check
        return code; // Return result to caller
    } // Close scope / block
    std::vector<u8> work(seg + from, seg + from + want), out; Counters dummy; // Allocate / manage dynamic buffer
    Hint none; encode_segment(work.data(), want, tbMax, none, out, dummy); const size_t plain = out.size(); // Compute and assign value
    size_t best = (size_t)-1; int code = 0; // Compute and assign value
    for (int p = 0; p < 3; p++) { // Loop iteration
        std::memcpy(work.data(), seg + from, want); // Fast raw memory copy
        AudioCodec(h.audio.channels, h.audio.width, p).forward(work.data(), want / frame); // Execute statement
        Hint hint; hint.ats = 0; hint.ate = want; hint.channels = h.audio.channels; hint.width = h.audio.width; // Compute and assign value
        encode_segment(work.data(), want, tbMax, hint, out, dummy); // Execute statement
        if (out.size() < best) { best = out.size(); code = p + 1; } // Conditional branch check
    } // Close scope / block
    if (best + best / 64 >= plain) code = 0;          // plain bytes unless audio mode is clearly (1.5%) better
    return code; // Return result to caller
} // Close scope / block

// Headerless data: looks for (a) 16-bit PCM audio, (b) a record / image-row size, (c) a grid transform (word differences or
// gradient residuals). Cheap statistics on a 2 MB sample decide which candidates are worth a trial; each is trial-coded on a
// 256 KB slice against plain coding. If several beat plain, the finalists are re-tested head-to-head on a larger slice.
// Fast contiguous-strip SAD helper (AVX2 _mm256_sad_epu8 when available, auto-vectorized scalar fallback)
inline u64 sad_strip(const u8* a, const u8* b, size_t len) { // Define helper function or constant
    u64 acc = 0; // Compute and assign value
    size_t i = 0; // Compute and assign value
#if P3_AVX2_MIXER // Conditional compilation check
    __m256i vsum = _mm256_setzero_si256(); // Compute and assign value
    for (; i + 32 <= len; i += 32) { // Loop iteration
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i)); // Compute and assign value
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i)); // Compute and assign value
        vsum = _mm256_add_epi64(vsum, _mm256_sad_epu8(va, vb)); // Compute and assign value
    } // Close scope / block
    alignas(32) u64 tmp[4]; // Execute statement
    _mm256_store_si256((__m256i*)tmp, vsum); // Execute statement
    acc += tmp[0] + tmp[1] + tmp[2] + tmp[3]; // Compute and assign value
#endif // End conditional compilation guard
    for (; i < len; i++) { // Loop iteration
        int dv = (int)a[i] - (int)b[i]; // Compute and assign value
        acc += (u64)(dv < 0 ? -dv : dv); // Compute and assign value
    } // Close scope / block
    return acc; // Return result to caller
} // Close scope / block

inline void detect_structure(const fs::path& src, u64 n, Header& h, bool execOnly = false, int kTb = 22) { // Define helper function or constant
    if (n < ((u64)512 << 10)) return; // Conditional branch check
    const size_t S = (size_t)std::min<u64>(n, (u64)2 << 20); // Compute and assign value
    std::vector<u8> smp(S); const u64 from = ((n - S) / 2) & ~(u64)7; // Allocate / manage dynamic buffer
    { File f; if (!f.open(src, "rb") || seek64(f.f, from) != 0 || std::fread(smp.data(), 1, S, f.f) != S) return; } // Buffered file I/O operation

    // Fast early-out for encrypted / already-compressed noise (order-0 & delta-1 entropy both near 8.0)
    if (!execOnly && S >= 32768) { // Conditional branch check
        const u8* q = smp.data() + (S / 2 - 16384); // Compute and assign value
        double h0 = block_entropy(q, 32768); // Compute and assign value
        if (h0 > 7.93) { // Conditional branch check
            u32 dc[256] = {}; // Compute and assign value
            for (size_t i = 1; i < 32768; i++) dc[(u8)(q[i] - q[i - 1])]++; // Loop iteration
            double ds = 0; // Compute and assign value
            for (int i = 0; i < 256; i++) if (dc[i]) ds += (double)dc[i] * std::log2((double)dc[i]); // Loop iteration
            double h1 = std::log2(32767.0) - ds / 32767.0; // Compute and assign value
            if (h1 > 7.93) return; // Conditional branch check
        } // Close scope / block
    } // Close scope / block

    // Reuse a single Model across all trial encodes (avoids repeated 20+ MB calloc/free churn)
    const int trialTb = (kTb == 1) ? 1 : std::min(kTb, 18); // Compute and assign value
    std::unique_ptr<Model> trialModel; // Execute statement
    if (trialTb != 1) trialModel.reset(new Model()); // Conditional branch check

    struct Cand { int kind = -1; Audio a; int stride = 0, bpp = 0; Grid g; int ek = 0; size_t sz = (size_t)-1; };   // kind: 0 audio, 1 stride hint, 2 grid, 3 executable filter
    auto build = [&](const Cand& c, size_t off, size_t len, std::vector<u8>& data, Hint& hint) { // Allocate / manage dynamic buffer
        data.assign(smp.begin() + off, smp.begin() + off + len); hint = Hint(); // Compute and assign value
        if (c.kind == 0) { // Conditional branch check
            size_t frame = (size_t)c.a.frame(), usable = (len / frame) * frame; data.resize(usable); // Compute and assign value
            AudioCodec(c.a.channels, 2, 1).forward(data.data(), usable / frame); // Execute statement
            hint.ats = 0; hint.ate = usable; hint.channels = c.a.channels; hint.width = 2; // Compute and assign value
        } else if (c.kind == 1) { hint.stride = c.stride; hint.bpp = c.bpp; } // Compute and assign value
        else if (c.kind == 3) { exec_filter(data.data(), len, from + off, c.ek, true); } // Alternative conditional branch
        else { // Fallback branch
            size_t ts, te; if (grid_region(c.g, from + off, len, ts, te)) GridCodec(c.g).forward(data.data() + ts, te - ts); // Execute statement
            hint.stride = c.g.stride; hint.bpp = c.g.width; // Compute and assign value
        } // Close scope / block
    }; // Close scope / block
    auto trialOf = [&](const Cand& c, size_t off, size_t len) { // Compute and assign value
        std::vector<u8> d, o; Hint hint; Counters dummy; build(c, off, len, d, hint); // Allocate / manage dynamic buffer
        encode_segment(d.data(), d.size(), trialTb, hint, o, dummy, trialModel.get()); return o.size(); // Execute statement
    }; // Close scope / block
    const size_t T = std::min<size_t>(S, (size_t)(P3_FAST_TRIALS ? 32 : 256) << 10), tOff = ((S - T) / 2) & ~(size_t)7; // Compute and assign value
    Cand plainC; plainC.kind = 1;                                  // stride 0 / bpp 0 == plain
    size_t plain = 0; bool havePlain = false; // Compute and assign value
    auto plainSize = [&]() { if (!havePlain) { plain = trialOf(plainC, tOff, T); havePlain = true; } return plain; }; // Compute and assign value
    Cand bestAudio, bestStr, bestGrid, bestExec;                    // best of each kind (vs plain, 2% margin)

    // (a) headerless 16-bit PCM, mono or stereo: samples must be much smoother than they are large
    for (int ch = 1; ch <= (execOnly ? 0 : 2); ch++) { // Loop iteration
        size_t fb = (size_t)(2 * ch), frames = std::min<size_t>(S / fb, 40000) - 1; double d = 0, a = 0; // Compute and assign value
        for (size_t t = 1; t <= frames; t++) { // Loop iteration
            int cur = (int)(i16)(smp[t * fb] | (smp[t * fb + 1] << 8)), prv = (int)(i16)(smp[(t - 1) * fb] | (smp[(t - 1) * fb + 1] << 8)); // Compute and assign value
            d += std::abs(cur - prv); a += std::abs(cur); // Compute and assign value
        } // Close scope / block
        if (a > 0 && d / a < 0.8) { // Conditional branch check
            Cand c; c.kind = 0; c.a.dataOff = 0; c.a.dataLen = n; c.a.channels = ch; c.a.width = 2; // Compute and assign value
            c.sz = trialOf(c, tOff, T); size_t ref = plainSize(); // Compute and assign value
            if (c.sz + c.sz / 32 < ref && c.sz < bestAudio.sz) bestAudio = c; // Conditional branch check
        } // Close scope / block
    } // Close scope / block
    // (d) executable code: 8 stripes of 64 KB spread over the whole file (code is rarely in the middle); the stripe with the most
    //     branch instructions (x86 E8/E9, ARM64 BL, ARM32 BL) is trial-coded with and without the filter
    { // Open scope / block
        const size_t SS = (size_t)(P3_FAST_TRIALS ? 48 : 256) << 10; const int ns = (int)std::min<u64>(8, n / SS); // Compute and assign value
        size_t bestHits[4] = {0, 0, 0, 0}; u64 bestOff[4] = {0, 0, 0, 0}; // Compute and assign value
        File f2; std::vector<u8> st(SS); // Allocate / manage dynamic buffer
        if (ns >= 1 && f2.open(src, "rb")) { // Conditional branch check
            for (int k = 0; k < ns; k++) { // Loop iteration
                u64 off = (ns == 1 ? 0 : (u64)k * (n - SS) / (u64)(ns - 1)) & ~(u64)7; // Compute and assign value
                if (seek64(f2.f, off) != 0 || std::fread(st.data(), 1, SS, f2.f) != SS) break; // Conditional branch check
                size_t x86 = 0, a64 = 0, a32 = 0; // Compute and assign value
                for (size_t i = 0; i + 5 <= SS; i++) if ((st[i] == 0xE8 || st[i] == 0xE9) && (st[i + 4] == 0 || st[i + 4] == 0xFF)) x86++; // Loop iteration
                for (size_t i = 0; i + 4 <= SS; i += 4) { if ((st[i + 3] >> 2) == 0x25) a64++; if (st[i + 3] == 0xEB) a32++; } // Loop iteration
                size_t hs[4] = {0, x86, a64, a32}; // Compute and assign value
                for (int q = 1; q <= 3; q++) if (hs[q] > bestHits[q]) { bestHits[q] = hs[q]; bestOff[q] = off; } // Loop iteration
            } // Close scope / block
            for (int q = 1; q <= 3; q++) { // Loop iteration
                if (bestHits[q] < SS / 2500) continue;                            // ~100 hits per stripe
                if (seek64(f2.f, bestOff[q]) != 0 || std::fread(st.data(), 1, SS, f2.f) != SS) break; // Conditional branch check
                std::vector<u8> o1, o2, w = st; Counters dummy; Hint none; // Allocate / manage dynamic buffer
                encode_segment(st.data(), SS, trialTb, none, o1, dummy, trialModel.get()); // Execute statement
                exec_filter(w.data(), SS, bestOff[q], q, true); // Execute statement
                encode_segment(w.data(), SS, trialTb, none, o2, dummy, trialModel.get()); // Execute statement
                if (o2.size() + o2.size() / 66 < o1.size() && o2.size() < bestExec.sz) { bestExec = Cand(); bestExec.kind = 3; bestExec.ek = q; bestExec.sz = o2.size(); } // Conditional branch check
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    if (execOnly) { if (bestExec.kind >= 0) h.exec = bestExec.ek; return; } // Conditional branch check

    // (b) record length / image row length: contiguous-strip SIMD SAD across 16 x 128-byte strips (2048 B total)
    //     Replaces 147M scattered scalar lookups with cache-friendly SIMD strips (~40-80x faster).
    const size_t maxLag = std::min<size_t>(S / 4, 49152), minLag = 8; // Compute and assign value
    const size_t kStripLen = 128, kStrips = 16; // Compute and assign value
    std::vector<double> sad(maxLag + 1, 0.0); // Allocate / manage dynamic buffer
    if (S > maxLag + kStripLen * 2) { // Conditional branch check
        size_t first = maxLag, avail = S - first - kStripLen; // Compute and assign value
        size_t stepS = std::max<size_t>(kStripLen, avail / kStrips); // Compute and assign value
        size_t sOff[kStrips]; size_t nStrips = 0; // Compute and assign value
        for (size_t i = first; i + kStripLen <= S && nStrips < kStrips; i += stepS) sOff[nStrips++] = i; // Loop iteration
        const double invN = 1.0 / (double)(nStrips * kStripLen); // Compute and assign value
        const u8* base = smp.data(); // Compute and assign value
        for (size_t L = 1; L <= maxLag; L++) { // Loop iteration
            u64 acc = 0; // Compute and assign value
            for (size_t s = 0; s < nStrips; s++) acc += sad_strip(base + sOff[s], base + sOff[s] - L, kStripLen); // Loop iteration
            sad[L] = (double)acc * invN; // Compute and assign value
        } // Close scope / block
    } // Close scope / block
    std::vector<double> srt(sad.begin() + 1, sad.end()); std::nth_element(srt.begin(), srt.begin() + srt.size() / 2, srt.end()); // Allocate / manage dynamic buffer
    const double median = srt[srt.size() / 2]; // Compute and assign value
    std::vector<size_t> cand; // Allocate / manage dynamic buffer
    { // Open scope / block
        std::vector<std::pair<double, size_t>> order; // Allocate / manage dynamic buffer
        for (size_t L = minLag + 3; L + 3 <= maxLag; L++) { // Loop iteration
            if (!(sad[L] < 0.8 * median)) continue; // Conditional branch check
            double nb = (sad[L - 3] + sad[L - 2] + sad[L + 2] + sad[L + 3]) / 4.0, prom = (nb - sad[L]) / (nb > 0 ? nb : 1.0); // Compute and assign value
            if (prom >= 0.03 && sad[L] <= sad[L - 1] && sad[L] <= sad[L + 1]) order.push_back({prom, L}); // Conditional branch check
        } // Close scope / block
        std::sort(order.begin(), order.end(), [](const std::pair<double, size_t>& x, const std::pair<double, size_t>& y) { return x.first > y.first; }); // Execute statement
        for (auto& pr : order) { size_t L = pr.second; bool tooClose = false; for (size_t c : cand) if ((L > c ? L - c : c - L) <= 4) tooClose = true; if (!tooClose) cand.push_back(L); if (cand.size() == 2) break; } // Loop iteration
    } // Close scope / block
    if (!cand.empty() && trialTb != 1) { // Conditional branch check
        int bppA = 1; double bs = sad[1]; for (int k = 2; k <= 4; k++) if (sad[k] < bs) { bs = sad[k]; bppA = k; } // Compute and assign value
        std::vector<int> bpps; bpps.push_back(bppA); if (bppA != 1) bpps.push_back(1); // Allocate / manage dynamic buffer
        for (size_t L : cand) for (int bp : bpps) { // Loop iteration
            Cand c; c.kind = 1; c.stride = (int)L; c.bpp = bp; c.sz = trialOf(c, tOff, T); size_t ref = plainSize(); // Compute and assign value
            if (c.sz + c.sz / 50 < ref && c.sz < bestStr.sz) bestStr = c; // Conditional branch check
        } // Close scope / block
    } // Close scope / block
    // (c) grid transform, screened by order-0 entropy (top 2 configurations get a trial)
    { // Open scope / block
        auto h0 = [](const u8* p, size_t len) { u32 c[256]; std::memset(c, 0, sizeof c); for (size_t i = 0; i < len; i++) c[p[i]]++; double e = 0; for (int i = 0; i < 256; i++) if (c[i]) e -= (double)c[i] * std::log2((double)c[i] / (double)len); return e / (double)len; }; // Initialize memory buffer
        std::vector<int> strides; for (size_t L : cand) strides.push_back((int)L); strides.push_back(2); strides.push_back(4); // Allocate / manage dynamic buffer
        struct Cfg { Grid g; double ent; }; std::vector<Cfg> cfgs; // Declare data structure / class
        for (int St : strides) for (int w : {1, 2, 4}) { // Loop iteration
            if (St % w || St < w) continue; // Conditional branch check
            for (int mode = 0; mode < 2; mode++) { // Loop iteration
                if (mode == 1 && St == w) continue; // Conditional branch check
                for (int zz = 0; zz < 2; zz++) { // Loop iteration
                    Grid g; g.stride = St; g.width = w; g.mode = mode; g.zz = zz; // Compute and assign value
                    size_t ts, te; if (!grid_region(g, from + tOff, T, ts, te) || te - ts < (size_t)St * 8) continue; // Execute statement
                    std::vector<u8> w2(smp.begin() + tOff, smp.begin() + tOff + T); GridCodec(g).forward(w2.data() + ts, te - ts); // Allocate / manage dynamic buffer
                    cfgs.push_back({g, h0(w2.data() + ts, te - ts)}); // Append element to container
                } // Close scope / block
            } // Close scope / block
        } // Close scope / block
        if (!cfgs.empty()) { // Conditional branch check
            double hPlain = h0(smp.data() + tOff, T); // Compute and assign value
            std::sort(cfgs.begin(), cfgs.end(), [](const Cfg& a, const Cfg& b) { return a.ent < b.ent; }); // Execute statement
            for (size_t i = 0; i < cfgs.size() && i < 2; i++) { // Loop iteration
                if (cfgs[i].ent > 0.97 * hPlain) break; // Conditional branch check
                Cand c; c.kind = 2; c.g = cfgs[i].g; c.sz = trialOf(c, tOff, T); size_t ref = plainSize(); // Compute and assign value
                if (c.sz + c.sz / 50 < ref && c.sz < bestGrid.sz) bestGrid = c; // Conditional branch check
            } // Close scope / block
        } // Close scope / block
    } // Close scope / block
    std::vector<Cand> fin; if (bestAudio.kind >= 0) fin.push_back(bestAudio); if (bestStr.kind >= 0) fin.push_back(bestStr); if (bestGrid.kind >= 0) fin.push_back(bestGrid); // Allocate / manage dynamic buffer
    if (bestExec.kind >= 0) { h.exec = bestExec.ek; return; }     // code files: the filter is chosen on its own stripes
    if (fin.empty()) return; // Conditional branch check
    size_t pick = 0; // Compute and assign value
    if (fin.size() > 1) {                                           // head-to-head on a larger slice
        const size_t big = std::min<size_t>(S, (size_t)(P3_FAST_TRIALS ? 96 : 768) << 10), bOff = ((S - big) / 2) & ~(size_t)7; // Compute and assign value
        size_t bestSz = (size_t)-1; // Compute and assign value
        for (size_t i = 0; i < fin.size(); i++) { size_t z = trialOf(fin[i], bOff, big); if (z < bestSz) { bestSz = z; pick = i; } } // Loop iteration
    } // Close scope / block
    const Cand& w = fin[pick]; // Compute and assign value
    if (w.kind == 0) h.audio = w.a; else if (w.kind == 1) { h.stride = w.stride; h.bpp = w.bpp; } else if (w.kind == 3) h.exec = w.ek; else h.grid = w.g; // Conditional branch check
} // Close scope / block
}  // namespace detail

// ------------------------------------------------------------------ public functions
// Compress src into <outDir>/<stem>.3p. The original name + extension are stored in the metadata.
inline Result compress_file(const fs::path& src, const Options& opt, const ProgressFn& progress) { // Define helper function or constant
    using namespace detail; // Type alias or namespace import
    fs::path tmp; TmpGuard xfGuard; // Execute statement
    try { // Begin exception-guarded block
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.clear(); // Execute statement
        g_prof.modeName = "compress"; // Compute and assign value
        u64 _tAll0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        std::error_code ec; // Execute statement
        u64 n = (u64)fs::file_size(src, ec); if (ec) return fail(Status::IoError, "cannot read file size: " + ec.message()); // Compute and assign value
        Header h; h.name = to_utf8(src.filename()); h.total = n; // Compute and assign value
        if (h.name.empty() || h.name.size() > kMaxName) return fail(Status::IoError, "file name is empty or too long"); // Conditional branch check
        { std::string e = to_utf8(src.extension()); for (char& c : e) c = (char)std::tolower((unsigned char)c); if (e == extension()) return fail(Status::BadFormat, "already a .3p file"); } // Compute and assign value
#ifdef P3_PROFILE // Guard if macro is defined
        u64 _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        { File f; if (!f.open(src, "rb")) return fail(Status::IoError, "cannot open the source file"); // Execute statement
          if (!detect_wav(f.f, n, h.audio)) { h.audio = Audio(); seek64(f.f, 0); u8 head[64]; size_t got = std::fread(head, 1, sizeof head, f.f); detect_geometry(head, got, h.stride, h.bpp); } } // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preWavBmpUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard

        // long-distance duplicates (not for WAV / BMP: their audio / pixel modelling relies on the exact byte layout)
        u64 streamLen = n; std::vector<LitRun> runs; double base = 0.0; fs::path dataPath = src; // Allocate / manage dynamic buffer
        if (opt.detect && !h.audio.on() && !h.stride && n >= 2048 && n <= ((u64)256 << 20)) {          // baseline JPEG: coefficient recompression
            File jf; u8 sig[3] = {0, 0, 0}; // Compute and assign value
            if (jf.open(src, "rb") && std::fread(sig, 1, 3, jf.f) == 3 && sig[0] == 0xFF && sig[1] == 0xD8 && sig[2] == 0xFF) { // Conditional branch check
                std::vector<u8> jd((size_t)n); seek64(jf.f, 0); // Allocate / manage dynamic buffer
                if (std::fread(jd.data(), 1, jd.size(), jf.f) == jd.size()) { // Conditional branch check
                    std::vector<u8> payload; // Allocate / manage dynamic buffer
                    if (jpeg_try(jd.data(), jd.size(), payload) && payload.size() + 64 < (size_t)n) { h.jpeg = true; h.jpegPayload = std::move(payload); h.fileCrc = crc32(jd.data(), jd.size()); streamLen = 0; } // Conditional branch check
                } // Close scope / block
            } // Close scope / block
        } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preJpegUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        if (opt.detect && !h.jpeg && !h.audio.on() && !h.stride && detect_mt19937(src, n, h)) streamLen = 0;      // whole file regenerable: no frames at all
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preMtUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        if (opt.detect && !h.jpeg && !h.gen && !h.audio.on() && !h.stride && n >= 64 && n <= ((u64)256 << 20)) {      // zlib / gzip / ZIP / PNG / PDF streams: magic-byte gated
            u8 sig[4] = {0, 0, 0, 0}; size_t gotSig = 0; // Compute and assign value
            { File jf; if (jf.open(src, "rb")) gotSig = std::fread(sig, 1, 4, jf.f); } // Buffered file I/O operation
            bool maybeDefl = (gotSig >= 2 && ( // Compute and assign value
                (gotSig >= 4 && sig[0] == 0x89 && sig[1] == 'P' && sig[2] == 'N' && sig[3] == 'G') || // Compute and assign value
                (gotSig >= 4 && sig[0] == 'P'  && sig[1] == 'K' && sig[2] == 3   && sig[3] == 4)   || // Compute and assign value
                (gotSig >= 3 && sig[0] == 0x1f && sig[1] == 0x8b && sig[2] == 8)                   || // Compute and assign value
                (gotSig >= 4 && sig[0] == '%'  && sig[1] == 'P' && sig[2] == 'D' && sig[3] == 'F') || // Compute and assign value
                zl::zlib_header_ok(sig) // Execute statement
            )); // Execute statement
            std::vector<u8> all; bool got = false; // Allocate / manage dynamic buffer
            if (maybeDefl) { // Conditional branch check
                all.resize((size_t)n); // Execute statement
                File jf; if (jf.open(src, "rb")) got = std::fread(all.data(), 1, all.size(), jf.f) == all.size(); // Buffered file I/O operation
            } // Close scope / block
            fs::path xdir = opt.outDir.empty() ? src.parent_path() : opt.outDir; if (xdir.empty()) xdir = "."; // Compute and assign value
            File xf; fs::path xp = unique_path(xdir, to_utf8(src.stem()) + ".xf.tmp"); zl::PngInfo pinfo; // Compute and assign value
            if (got && xf.open(xp, "wb")) { // Conditional branch check
                xfGuard.p = xp; std::vector<zl::DefRec> recs; u64 xt = 0; int rc = 0; // Allocate / manage dynamic buffer
                try { rc = zl::transform(all.data(), all.size(), xf.f, recs, xt, pinfo, [&](double f) { return !progress || progress(0.03 * f, 0); }); } // Begin exception-guarded block
                catch (const Failure& fl) { xf.close(); return fail(fl.status, fl.msg); } // Handle caught exception
                if (std::fflush(xf.f) != 0) rc = 0; // Conditional branch check
                xf.close(); // Execute statement
                if (rc < 0) return fail(Status::Cancelled, "cancelled"); // Conditional branch check
                std::vector<u8> tbl; if (rc == 1) { for (const zl::DefRec& r : recs) zl::put_rec(tbl, r); if (tbl.size() > kSegment / 2) rc = 0; } // Allocate / manage dynamic buffer
                if (rc == 1) { // Conditional branch check
                    std::vector<u8> pay; Counters dummy; encode_segment(tbl.data(), tbl.size(), 20, Hint(), pay, dummy); // Allocate / manage dynamic buffer
                    bool modeled = pay.size() < tbl.size(); // Compute and assign value
                    h.defl = true; h.deflFlag = modeled ? 1 : 0; h.deflTbl = modeled ? pay : tbl; h.deflRaw = tbl.size(); h.deflCount = recs.size(); h.xTotal = xt; h.dStride = pinfo.stride; h.dBpp = pinfo.bpp; h.dPix = pinfo.pix ? 1 : 0; h.dPhase = pinfo.phase; h.fileCrc = crc32(all.data(), all.size()); // Update / verify CRC-32 integrity checksum
                    if (opt.verify) { // Conditional branch check
                        File vf; u32 c = 0; u64 cnt = 0; if (!vf.open(xp, "rb")) return fail(Status::IoError, "cannot reopen the temporary file"); // Compute and assign value
                        std::vector<zl::DefRec> recs2; if (!zl::parse_recs(tbl, recs.size(), recs2)) return fail(Status::Corrupt, "verification failed: the stream table does not parse"); // Allocate / manage dynamic buffer
                        try { zl::rebuild(vf.f, xt, recs2, [&](const u8* q, size_t k) { c = crc32_update(c, q, k); cnt += k; }); } catch (const Failure& fl) { return fail(fl.status, fl.msg); } // Begin exception-guarded block
                        if (cnt != n || c != h.fileCrc) return fail(Status::Corrupt, "verification failed: the deflate streams could not be rebuilt exactly"); // Conditional branch check
                    } // Close scope / block
                    dataPath = xp; streamLen = xt; base = 0.03; // Compute and assign value
                } else { remove_quiet(xp); xfGuard.p.clear(); } // Execute statement
            } // Close scope / block
        } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preDeflateUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        bool allowDedup = opt.dedup && n >= (opt.fast ? ((u64)8 << 20) : ((u64)2 << 20)); // Compute and assign value
        if (allowDedup && opt.fast) { // Conditional branch check
            // Skip dedup in Fast mode on very low-entropy files (H0 < 2.0) to prevent rolling-hash collision thrashing
            File df; u8 dsmp[16384]; // Execute statement
            if (df.open(src, "rb")) { // Conditional branch check
                size_t gotD = std::fread(dsmp, 1, sizeof dsmp, df.f); // Buffered file I/O operation
                if (gotD >= 4096 && block_entropy(dsmp, gotD) < 2.0) allowDedup = false; // Conditional branch check
            } // Close scope / block
        } // Close scope / block
        if (!h.gen && !h.jpeg && !h.defl && !h.audio.on() && !h.stride && allowDedup) { // Conditional branch check
            CrcJob job; job.path = src; sys::Thread crcThread; crcThread.start([&job] { crc_file(&job); }); // Compute and assign value
            std::vector<Match> found; u64 matched = 0; bool cancelled = false; // Allocate / manage dynamic buffer
            bool ok = find_duplicates(src, n, (u64)256 << 20, [&](double f) { return !progress || progress(0.03 * f, 0); }, found, matched, cancelled); // Compute and assign value
            crcThread.join(); // Execute statement
            if (!ok) return fail(cancelled ? Status::Cancelled : Status::IoError, cancelled ? "cancelled" : "could not read the source file"); // Conditional branch check
            base = 0.03; // Compute and assign value
            if (job.ok && matched >= std::max<u64>(n / 100, (u64)65536)) { // Conditional branch check
                h.dedup = true; h.matches = std::move(found); h.fileCrc = job.crc; streamLen = n - matched; runs = make_runs(h.matches, n); // Compute and assign value
            } // Close scope / block
        } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preDedupUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        if (!opt.fast) { // Conditional branch check
            if (!h.gen && !h.jpeg && !h.defl && !h.dedup && !h.audio.on() && !h.stride && opt.detect) detect_structure(src, n, h, false, 22); // Conditional branch check
            else if (h.dedup && opt.detect) detect_structure(src, n, h, true, 22);            // copy records + executable filter
        } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preDetectUs = prof_now_us() - _tP0; // Compute and assign value
        _tP0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        Options effOpt = opt; // Compute and assign value
        if (h.stride || h.grid.on() || (h.defl && h.dStride)) effOpt.fast = false; // Conditional branch check
        unsigned wantThreads = effOpt.threads ? effOpt.threads : std::min(sys::hardware_threads(), 16u); // Compute and assign value
        double segSampleH0 = 5.0; // Sample entropy for adaptive segment sizing
        if (effOpt.fast && !h.dedup && streamLen >= 16384u) { // Probe sample for entropy & byte-plane shuffle
            File sf; std::vector<u8> ssmp(65536); // 64 KB sample buffer
            if (sf.open(dataPath, "rb")) { // Open input stream for quick probe
                size_t gotS = std::fread(ssmp.data(), 1, ssmp.size(), sf.f); // Read up to 64 KB
                if (gotS >= 4096) { // Sufficient sample size
                    segSampleH0 = block_entropy(ssmp.data(), gotS); // Measure baseline H0
                    if (opt.detect && !h.gen && !h.jpeg && !h.defl && !h.audio.on() && !h.stride && !h.grid.on() && !h.exec && segSampleH0 > 1.5) { // Eligible for byte-plane shuffle
                        int bestStride = 0; double bestH0 = segSampleH0; // Track best stride candidate
                        std::vector<u8> trial(gotS); // Scratch buffer for stride trial
                        for (int cand : {2, 4, 8}) { // Test 16-bit, 32-bit, and 64-bit record widths
                            std::memcpy(trial.data(), ssmp.data(), gotS); // Copy raw sample
                            byte_plane_shuffle_delta(trial.data(), gotS, cand, true); // Apply forward shuffle+delta
                            double ch = block_entropy(trial.data(), gotS); // Measure post-transform H0
                            if (ch < bestH0) { bestH0 = ch; bestStride = cand; } // Keep lowest entropy stride
                        } // End stride trial loop
                        if (bestStride && bestH0 + 0.35 < segSampleH0) { h.bshuf = bestStride; segSampleH0 = bestH0; } // Accept only on >= 0.35 b/B drop
                    } // End byte-plane shuffle probe
                } // End sample size check
            } // End file open check
        } // End fast-mode sample probe
        h.segCode = choose_seg_code(streamLen, h.audio.on() && effOpt.fast ? std::max(1u, wantThreads / 2u) : wantThreads, effOpt.fast, segSampleH0, h.dedup); // Segment sizing configuration
        const size_t segSize = seg_size_from_code(h.segCode); // Segment sizing configuration
        u64 nseg = (streamLen + segSize - 1) / segSize; // Segment sizing configuration
        unsigned threads = 1; // Compute and assign value
        plan(effOpt, nseg, threads, h.tbMax); // Execute statement

#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.inBytes = n; g_prof.streamBytes = streamLen; // Compute and assign value
        g_prof.tbMax = h.tbMax; g_prof.segCode = h.segCode; g_prof.segSize = segSize; g_prof.nseg = nseg; // Segment sizing configuration
        g_prof.hwThreads = sys::hardware_threads(); g_prof.usedThreads = threads; g_prof.fastMode = (h.tbMax == 1); // Compute and assign value
        if (h.jpeg) g_prof.detectedPath = "JPEG-DCT"; // Conditional branch check
        else if (h.gen) g_prof.detectedPath = "MT19937"; // Alternative conditional branch
        else if (h.defl) g_prof.detectedPath = h.dStride ? "Deflate+PNG-Pix" : "Deflate-Inflated"; // Alternative conditional branch
        else if (h.audio.on()) g_prof.detectedPath = "WAV-PCM-Audio"; // Alternative conditional branch
        else if (h.stride) g_prof.detectedPath = "BMP-Image"; // Alternative conditional branch
        else if (h.grid.on()) g_prof.detectedPath = "Grid-Transform"; // Alternative conditional branch
        else if (h.dedup && h.exec) g_prof.detectedPath = "Dedup+ExecFilter"; // Alternative conditional branch
        else if (h.dedup) g_prof.detectedPath = "Dedup"; // Alternative conditional branch
        else if (h.exec) g_prof.detectedPath = "ExecFilter"; // Alternative conditional branch
        else g_prof.detectedPath = "Plain"; // Fallback branch
        g_prof.threads.resize(threads); // Execute statement
        for (unsigned ti = 0; ti < threads; ti++) g_prof.threads[ti].id = ti; // Loop iteration
        std::atomic<unsigned> profTid{0}; // Execute statement
#endif // End conditional compilation guard

        fs::path dir = opt.outDir.empty() ? src.parent_path() : opt.outDir; if (dir.empty()) dir = "."; // Compute and assign value
        fs::path finalPath = unique_path(dir, to_utf8(src.stem()) + extension()); // Compute and assign value
        tmp = finalPath; tmp += ".tmp"; // Compute and assign value
        File out; if (!out.open(tmp, "wb")) return fail(Status::IoError, "cannot create the output file (folder not writable?)"); // Execute statement
        std::vector<u8> hv = make_header(h); // Allocate / manage dynamic buffer
        if (std::fwrite(hv.data(), 1, hv.size(), out.f) != hv.size()) return fail(Status::IoError, "write failed"); // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.preHeaderUs = prof_now_us() - _tP0; // Compute and assign value
        u64 _tWorkWall0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard

        struct CompItem { std::vector<u8> fh, payload; }; // Declare data structure / class
        std::map<u64, CompItem> pending; // Execute statement
        bool draining = false; // Compute and assign value
        const size_t maxPending = std::max<size_t>((size_t)threads * 2u, 4u); // Compute and assign value
        Shared sh; u64 written = hv.size(); sh.cn.out = hv.size(); // Compute and assign value
        auto worker = [&]() { // Compute and assign value
#ifdef P3_PROFILE // Guard if macro is defined
            unsigned myTid = profTid.fetch_add(1, std::memory_order_relaxed); // Compute and assign value
#endif // End conditional compilation guard
            File f; if (!f.open(dataPath, "rb")) throw Failure{Status::IoError, "cannot open the source file"}; // Execute statement
            std::vector<u8> seg(segSize), payload, check; // Allocate / manage dynamic buffer
            std::unique_ptr<Model> threadModel; // Execute statement
            std::unique_ptr<fast::FastScratch> threadFast; // Execute statement
            if (h.tbMax != 1) threadModel.reset(new Model()); // Conditional branch check
            else threadFast.reset(new fast::FastScratch()); // Fallback branch
            for (;;) { // Loop iteration
                u64 idx = sh.nextSeg.fetch_add(1); // Compute and assign value
                if (idx >= nseg || sh.cn.cancel.load()) return; // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
                u64 _tSeg0 = prof_now_us(); // Compute and assign value
                u64 _t0 = _tSeg0; // Compute and assign value
#endif // End conditional compilation guard
                size_t L = (size_t)std::min<u64>(segSize, streamLen - idx * segSize); // Segment sizing configuration
                if (h.dedup ? !read_literals(f.f, runs, idx * segSize, L, seg.data()) : (seek64(f.f, idx * segSize) != 0 || std::fread(seg.data(), 1, L, f.f) != L)) throw Failure{Status::IoError, "could not read the source file (changed while reading?)"}; // Conditional branch check
                u32 crc = crc32(seg.data(), L); // Update / verify CRC-32 integrity checksum
#ifdef P3_PROFILE // Guard if macro is defined
                u64 _t1 = prof_now_us(); // Compute and assign value
                g_prof.wReadCrcUs.fetch_add(_t1 - _t0, std::memory_order_relaxed); // Execute statement
                _t0 = _t1; // Compute and assign value
#endif // End conditional compilation guard
                int code = h.audio.on() ? choose_audio_code(h, seg.data(), L, idx * segSize, h.tbMax) : 0; // Segment sizing configuration
                Hint hint = make_hint(h, idx * segSize, L, code); // Segment sizing configuration
                size_t gts = 0, gte = 0; bool gridOn = grid_region(h.grid, idx * segSize, L, gts, gte); // Segment sizing configuration
                if (hint.ate > hint.ats) AudioCodec(h.audio.channels, h.audio.width, code - 1).forward(seg.data() + hint.ats, (size_t)((hint.ate - hint.ats) / (u64)h.audio.frame())); // Conditional branch check
                if (gridOn) GridCodec(h.grid).forward(seg.data() + gts, gte - gts); // Conditional branch check
                if (h.exec) exec_filter(seg.data(), L, idx * segSize, h.exec, true); // Conditional branch check
                if (h.bshuf) { thread_local std::vector<u8> bsTmp; byte_plane_shuffle_delta(seg.data(), L, h.bshuf, true, &bsTmp); } // Reusable thread-local scratch
#ifdef P3_PROFILE // Guard if macro is defined
                _t1 = prof_now_us(); // Compute and assign value
                g_prof.wTransformUs.fetch_add(_t1 - _t0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
                encode_segment(seg.data(), L, h.tbMax, hint, payload, sh.cn, threadModel.get(), threadFast.get()); // Execute statement
                bool modeled = payload.size() < L; // Compute and assign value
                if (modeled && opt.verify) { // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
                    u64 _tV0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
                    check.resize(L); decode_segment(payload.data(), payload.size(), check.data(), L, h.tbMax, hint, nullptr, threadModel.get()); // Execute statement
                    if (hint.ate > hint.ats) AudioCodec(h.audio.channels, h.audio.width, code - 1).inverse(check.data() + hint.ats, (size_t)((hint.ate - hint.ats) / (u64)h.audio.frame())); // Conditional branch check
                    if (gridOn) GridCodec(h.grid).inverse(check.data() + gts, gte - gts); // Conditional branch check
                    if (h.exec) exec_filter(check.data(), L, idx * segSize, h.exec, false); // Conditional branch check
                    if (h.bshuf) { thread_local std::vector<u8> bsTmp; byte_plane_shuffle_delta(check.data(), L, h.bshuf, false, &bsTmp); } // Reusable thread-local scratch
                    if (crc32(check.data(), L) != crc) throw Failure{Status::Corrupt, "verification failed: decoded data differs from the original"}; // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
                    g_prof.wVerifyUs.fetch_add(prof_now_us() - _tV0, std::memory_order_relaxed); // Execute statement
#endif // End conditional compilation guard
                } // Close scope / block
                if (!modeled) { // Conditional branch check
                    if (hint.ate > hint.ats || gridOn || h.exec) { if (h.dedup ? !read_literals(f.f, runs, idx * segSize, L, seg.data()) : (seek64(f.f, idx * segSize) != 0 || std::fread(seg.data(), 1, L, f.f) != L)) throw Failure{Status::IoError, "could not read the source file"}; } // Conditional branch check
                    payload.assign(seg.begin(), seg.begin() + L); // Execute statement
                } // Close scope / block
                std::vector<u8> fh; put_varint(fh, ((u64)payload.size() << 3) | (u64)((modeled ? code : 0) << 1) | (modeled ? 1u : 0u)); // Allocate / manage dynamic buffer
                for (int i = 0; i < 4; i++) fh.push_back((u8)(crc >> (8 * i))); // Loop iteration

                // Non-blocking reorder queue: insert and immediately steal next segment unless draining head
#ifdef P3_PROFILE // Guard if macro is defined
                u64 _tQ0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
                bool doDrain = false; // Compute and assign value
                { // Open scope / block
                    sys::Lock l(sh.m); // Execute statement
                    while (idx != sh.nextWrite && pending.size() >= maxPending && !sh.cn.cancel.load()) sh.cv.wait(sh.m); // Loop while condition holds
                    if (sh.cn.cancel.load()) return; // Conditional branch check
                    CompItem item; item.fh = std::move(fh); item.payload.swap(payload); // Compute and assign value
                    pending.emplace(idx, std::move(item)); // Execute statement
                    if (!draining && pending.count(sh.nextWrite)) { draining = true; doDrain = true; } // Conditional branch check
                } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
                u64 _tQ1 = prof_now_us(); // Compute and assign value
                u64 _qWait = _tQ1 - _tQ0; // Compute and assign value
                g_prof.wQueueWaitUs.fetch_add(_qWait, std::memory_order_relaxed); // Execute statement
                u64 _wDrain = 0; // Compute and assign value
#endif // End conditional compilation guard
                while (doDrain) { // Loop while condition holds
#ifdef P3_PROFILE // Guard if macro is defined
                    u64 _tD0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
                    CompItem cur; // Execute statement
                    { // Open scope / block
                        sys::Lock l(sh.m); // Execute statement
                        auto it = pending.find(sh.nextWrite); // Compute and assign value
                        if (it == pending.end() || sh.cn.cancel.load()) { draining = false; sh.cv.notify_all(); break; } // Conditional branch check
                        cur = std::move(it->second); // Compute and assign value
                        pending.erase(it); // Execute statement
                    } // Close scope / block
                    if (std::fwrite(cur.fh.data(), 1, cur.fh.size(), out.f) != cur.fh.size() || // Conditional branch check
                        std::fwrite(cur.payload.data(), 1, cur.payload.size(), out.f) != cur.payload.size()) { // Buffered file I/O operation
                        sys::Lock l(sh.m); // Execute statement
                        draining = false; sh.failed = true; sh.st = Status::DiskFull; sh.msg = "write failed (disk full?)"; sh.cn.cancel = true; sh.cv.notify_all(); return; // Compute and assign value
                    } // Close scope / block
                    { // Open scope / block
                        sys::Lock l(sh.m); // Execute statement
                        written += cur.fh.size() + cur.payload.size(); // Compute and assign value
                        sh.nextWrite++; // Execute statement
                        sh.cv.notify_all(); // Execute statement
                    } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
                    _wDrain += prof_now_us() - _tD0; // Compute and assign value
#endif // End conditional compilation guard
                } // Close scope / block
#ifdef P3_PROFILE // Guard if macro is defined
                u64 _segDt = prof_now_us() - _tSeg0; // Compute and assign value
                g_prof.wWriteDrainUs.fetch_add(_wDrain, std::memory_order_relaxed); // Execute statement
                g_prof.wTotalActiveUs.fetch_add(_segDt, std::memory_order_relaxed); // Execute statement
                prof_update_minmax(g_prof.segMinUs, g_prof.segMaxUs, _segDt); // Execute statement
                g_prof.segSumUs.fetch_add(_segDt, std::memory_order_relaxed); // Execute statement
                g_prof.segDone.fetch_add(1, std::memory_order_relaxed); // Execute statement
                if (myTid < g_prof.threads.size()) { // Conditional branch check
                    g_prof.threads[myTid].segs++; // Execute statement
                    g_prof.threads[myTid].activeUs += _segDt; // Compute and assign value
                    g_prof.threads[myTid].waitUs += _qWait; // Compute and assign value
                    g_prof.threads[myTid].writeUs += _wDrain; // Compute and assign value
                } // Close scope / block
#endif // End conditional compilation guard
            } // Close scope / block
        }; // Close scope / block
        ProgressFn mapped = progress ? ProgressFn([&](double f, u64 o) { return progress(base + (1.0 - base) * f, o); }) : ProgressFn(); // Compute and assign value
        run_workers(sh, threads, streamLen, worker, mapped); // Execute statement
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.workerWallUs = prof_now_us() - _tWorkWall0; // Compute and assign value
        u64 _tFin0 = prof_now_us(); // Compute and assign value
#endif // End conditional compilation guard
        if (sh.failed) { out.close(); remove_quiet(tmp); return fail(sh.st, sh.msg); } // Conditional branch check
        if (sh.userCancel || sh.nextWrite != nseg) { out.close(); remove_quiet(tmp); return fail(Status::Cancelled, "cancelled"); } // Conditional branch check
        if (std::fflush(out.f) != 0) { out.close(); remove_quiet(tmp); return fail(Status::DiskFull, "write failed (disk full?)"); } // Conditional branch check
        out.close(); // Execute statement
        fs::rename(tmp, finalPath, ec); // Execute statement
        if (ec) { remove_quiet(tmp); return fail(Status::IoError, "cannot finalize the output file: " + ec.message()); } // Conditional branch check
#ifdef P3_PROFILE // Guard if macro is defined
        g_prof.finalizeUs = prof_now_us() - _tFin0; // Compute and assign value
        g_prof.outBytes = written; // Compute and assign value
        g_prof.wallTotalUs = prof_now_us() - _tAll0; // Compute and assign value
#endif // End conditional compilation guard
        Result r; r.inSize = n; r.outSize = written; r.outPath = finalPath; r.storedName = h.name; // Compute and assign value
        if (progress) progress(1.0, written); // Conditional branch check
        return r; // Return result to caller
    } catch (const std::bad_alloc&) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::OutOfMemory, "not enough memory"); // Execute statement
    } catch (const std::exception& e) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::IoError, e.what()); } // Execute statement
} // Close scope / block

namespace detail { // Open namespace scope
inline Result restore_jpeg(const fs::path& src, u64 fileSize, const Header& h, const Options& opt, const ProgressFn& progress) { // Define helper function or constant
    fs::path tmp; // Execute statement
    try { // Begin exception-guarded block
        std::vector<u8> file; // Allocate / manage dynamic buffer
        if (!jpeg_restore(h.jpegPayload, file) || file.size() != h.total || crc32(file.data(), file.size()) != h.fileCrc) // Conditional branch check
            return fail(Status::Corrupt, "file is damaged: the JPEG could not be rebuilt exactly"); // Return result to caller
        std::string name = safe_name(h.name); // Compute and assign value
        fs::path dir = opt.outDir.empty() ? src.parent_path() : opt.outDir; if (dir.empty()) dir = "."; // Compute and assign value
        { std::error_code e2; auto sp = fs::space(dir, e2); if (!e2 && sp.available < h.total) return fail(Status::DiskFull, "not enough free disk space for the restored file"); } // Compute and assign value
        fs::path finalPath = unique_path(dir, name); tmp = finalPath; tmp += ".tmp"; // Compute and assign value
        File out; if (!out.open(tmp, "wb")) return fail(Status::IoError, "cannot create the output file (folder not writable?)"); // Execute statement
        if (std::fwrite(file.data(), 1, file.size(), out.f) != file.size() || std::fflush(out.f) != 0) { out.close(); remove_quiet(tmp); return fail(Status::DiskFull, "write failed (disk full?)"); } // Conditional branch check
        out.close(); // Execute statement
        std::error_code ec; fs::rename(tmp, finalPath, ec); // Execute statement
        if (ec) { remove_quiet(tmp); return fail(Status::IoError, "cannot finalize the output file: " + ec.message()); } // Conditional branch check
        if (progress) progress(1.0, h.total); // Conditional branch check
        Result r; r.inSize = fileSize; r.outSize = h.total; r.outPath = finalPath; r.storedName = h.name; return r; // Compute and assign value
    } catch (const std::exception& e) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::IoError, e.what()); } // Execute statement
} // Close scope / block
inline Result restore_generated(const fs::path& src, u64 fileSize, const Header& h, const Options& opt, const ProgressFn& progress) { // Define helper function or constant
    fs::path tmp; // Execute statement
    try { // Begin exception-guarded block
        std::string name = safe_name(h.name); // Compute and assign value
        fs::path dir = opt.outDir.empty() ? src.parent_path() : opt.outDir; if (dir.empty()) dir = "."; // Compute and assign value
        { std::error_code e2; auto sp = fs::space(dir, e2); if (!e2 && sp.available < h.total) return fail(Status::DiskFull, "not enough free disk space for the restored file"); } // Compute and assign value
        fs::path finalPath = unique_path(dir, name); tmp = finalPath; tmp += ".tmp"; // Compute and assign value
        File out; if (!out.open(tmp, "wb")) return fail(Status::IoError, "cannot create the output file (folder not writable?)"); // Execute statement
        u32 w[624]; for (int i = 0; i < 624; i++) w[i] = rd32le(&h.genPrefix[(size_t)i * 4]); // Compute and assign value
        Mt19937Stream mt(w); u32 crc = 0; u64 words = (h.total - h.genTail.size()) / 4, done = 0; // Compute and assign value
        std::vector<u8> buf((size_t)4 << 20); // Allocate / manage dynamic buffer
        auto emit = [&](const u8* p, size_t n) { if (std::fwrite(p, 1, n, out.f) != n) throw Failure{Status::DiskFull, "write failed (disk full?)"}; crc = crc32_update(crc, p, n); }; // Buffered file I/O operation
        emit(h.genPrefix.data(), h.genPrefix.size()); done = 624; // Compute and assign value
        while (done < words) { // Loop while condition holds
            size_t cnt = (size_t)std::min<u64>(buf.size() / 4, words - done); // Compute and assign value
            for (size_t i = 0; i < cnt; i++) { u32 v = mt.next(); buf[i * 4] = (u8)v; buf[i * 4 + 1] = (u8)(v >> 8); buf[i * 4 + 2] = (u8)(v >> 16); buf[i * 4 + 3] = (u8)(v >> 24); } // Loop iteration
            emit(buf.data(), cnt * 4); done += cnt; // Compute and assign value
            if (progress && !progress((double)done / (double)words, done * 4)) { out.close(); remove_quiet(tmp); return fail(Status::Cancelled, "cancelled"); } // Conditional branch check
        } // Close scope / block
        if (!h.genTail.empty()) emit(h.genTail.data(), h.genTail.size()); // Conditional branch check
        if (std::fflush(out.f) != 0) throw Failure{Status::DiskFull, "write failed (disk full?)"}; // Conditional branch check
        out.close(); // Execute statement
        if (crc != h.fileCrc) { remove_quiet(tmp); return fail(Status::Corrupt, "file is damaged: checksum does not match"); } // Conditional branch check
        std::error_code ec; fs::rename(tmp, finalPath, ec); // Execute statement
        if (ec) { remove_quiet(tmp); return fail(Status::IoError, "cannot finalize the output file: " + ec.message()); } // Conditional branch check
        Result r; r.inSize = fileSize; r.outSize = h.total; r.outPath = finalPath; r.storedName = h.name; return r; // Compute and assign value
    } catch (const Failure& f) { if (!tmp.empty()) remove_quiet(tmp); return fail(f.status, f.msg); // Execute statement
    } catch (const std::exception& e) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::IoError, e.what()); } // Execute statement
} // Close scope / block
}  // namespace detail

// Restore a .3p file under its original name and extension (read from the metadata).
inline Result restore_file(const fs::path& src, const Options& opt, const ProgressFn& progress) { // Define helper function or constant
    using namespace detail; // Type alias or namespace import
    fs::path tmp; // Execute statement
    try { // Begin exception-guarded block
        std::error_code ec; // Execute statement
        u64 fileSize = (u64)fs::file_size(src, ec); if (ec) return fail(Status::IoError, "cannot read file size: " + ec.message()); // Compute and assign value
        Header h; // Execute statement
        { File f; if (!f.open(src, "rb")) return fail(Status::IoError, "cannot open the file"); if (!read_header(f.f, h, fileSize)) return fail(Status::BadFormat, "not a valid .3p file"); } // Execute statement
        if (h.gen) return restore_generated(src, fileSize, h, opt, progress); // Conditional branch check
        if (h.jpeg) return restore_jpeg(src, fileSize, h, opt, progress); // Conditional branch check
        const u64 streamLen = h.dedup ? h.litTotal : h.defl ? h.xTotal : h.total; // Compute and assign value
        const size_t segSize = seg_size_from_code(h.segCode); // Segment sizing configuration
        u64 nseg = (streamLen + segSize - 1) / segSize; // Segment sizing configuration
        if ((!h.dedup && streamLen > fileSize * 4096 + kSegment) || nseg * 5 > fileSize) return fail(Status::BadFormat, "not a valid .3p file (damaged header)"); // Conditional branch check
        std::vector<zl::DefRec> drecs; // Allocate / manage dynamic buffer
        if (h.defl) { // Conditional branch check
            std::vector<u8> tbl; // Allocate / manage dynamic buffer
            if (h.deflFlag) { tbl.resize((size_t)h.deflRaw); try { decode_segment(h.deflTbl.data(), h.deflTbl.size(), tbl.data(), tbl.size(), 20, Hint(), nullptr); } catch (const Failure&) { return fail(Status::Corrupt, "file is damaged (stream table)"); } } // Conditional branch check
            else tbl = h.deflTbl; // Fallback branch
            if (tbl.size() != h.deflRaw || !zl::parse_recs(tbl, h.deflCount, drecs)) return fail(Status::Corrupt, "file is damaged (stream table)"); // Conditional branch check
        } // Close scope / block

        struct Frame { u64 off = 0, plen = 0; u32 crc = 0; bool modeled = false; int code = 0; }; // Declare data structure / class
        std::vector<Frame> frames((size_t)nseg); // Allocate / manage dynamic buffer
        { File f; if (!f.open(src, "rb") || seek64(f.f, h.bytes) != 0) return fail(Status::IoError, "cannot open the file"); // Compute and assign value
          u64 pos = h.bytes; // Compute and assign value
          for (u64 i = 0; i < nseg; i++) { // Loop iteration
              u64 v = 0, used = 0; u8 c4[4]; // Compute and assign value
              if (!get_varint(f.f, v, used) || std::fread(c4, 1, 4, f.f) != 4) return fail(Status::Corrupt, "file is truncated or damaged"); // Conditional branch check
              Frame& fr = frames[(size_t)i]; fr.plen = v >> 3; fr.modeled = (v & 1) != 0; fr.code = (int)((v >> 1) & 3); // Compute and assign value
              fr.crc = (u32)c4[0] | ((u32)c4[1] << 8) | ((u32)c4[2] << 16) | ((u32)c4[3] << 24); // Compute and assign value
              fr.off = pos + used + 4; // Compute and assign value
              u64 L = std::min<u64>(segSize, streamLen - i * segSize); // Segment sizing configuration
              if (fr.plen > L || (!fr.modeled && fr.plen != L) || fr.off + fr.plen > fileSize) return fail(Status::Corrupt, "file is truncated or damaged"); // Conditional branch check
              pos = fr.off + fr.plen; if (seek64(f.f, pos) != 0) return fail(Status::Corrupt, "file is truncated or damaged"); // Compute and assign value
          } } // Execute statement

        std::string name = safe_name(h.name); // Compute and assign value
        fs::path dir = opt.outDir.empty() ? src.parent_path() : opt.outDir; if (dir.empty()) dir = "."; // Compute and assign value
        { std::error_code e2; auto sp = fs::space(dir, e2); if (!e2 && sp.available < h.total + (h.defl ? h.xTotal : 0)) return fail(Status::DiskFull, "not enough free disk space for the restored file"); } // Compute and assign value
        fs::path finalPath = unique_path(dir, name); // Compute and assign value
        tmp = finalPath; tmp += ".tmp"; // Compute and assign value
        File out; if (!out.open(tmp, "wb")) return fail(Status::IoError, "cannot create the output file (folder not writable?)"); // Execute statement

        unsigned threads = pick_threads(opt, nseg, h.tbMax); // Compute and assign value
        std::map<u64, std::vector<u8>> pending; // Allocate / manage dynamic buffer
        bool draining = false; // Compute and assign value
        const size_t maxPending = std::max<size_t>((size_t)threads * 2u, 4u); // Compute and assign value
        Shared sh; // Execute statement
        OutWriter writer(out.f, tmp, h.dedup ? &h.matches : nullptr, sh.cn); // Execute statement
        auto worker = [&]() { // Compute and assign value
            File f; if (!f.open(src, "rb")) throw Failure{Status::IoError, "cannot open the file"}; // Execute statement
            std::vector<u8> payload, seg; // Allocate / manage dynamic buffer
            std::unique_ptr<Model> threadModel; // Execute statement
            if (h.tbMax != 1) threadModel.reset(new Model()); // Conditional branch check
            for (;;) { // Loop iteration
                u64 idx = sh.nextSeg.fetch_add(1); // Compute and assign value
                if (idx >= nseg || sh.cn.cancel.load()) return; // Conditional branch check
                const Frame& fr = frames[(size_t)idx]; // Compute and assign value
                size_t L = (size_t)std::min<u64>(segSize, streamLen - idx * segSize); // Segment sizing configuration
                payload.resize((size_t)fr.plen); seg.resize(L); // Execute statement
                if (seek64(f.f, fr.off) != 0 || std::fread(payload.data(), 1, payload.size(), f.f) != payload.size()) throw Failure{Status::Corrupt, "file is truncated or damaged"}; // Conditional branch check
                Hint hint = make_hint(h, idx * segSize, L, fr.modeled ? fr.code : 0); // Segment sizing configuration
                if (fr.modeled) { // Conditional branch check
                    decode_segment(payload.data(), payload.size(), seg.data(), L, h.tbMax, hint, &sh.cn, threadModel.get()); // Execute statement
                    { size_t gts = 0, gte = 0; if (grid_region(h.grid, idx * segSize, L, gts, gte)) GridCodec(h.grid).inverse(seg.data() + gts, gte - gts); } // Segment sizing configuration
                    if (h.exec) exec_filter(seg.data(), L, idx * segSize, h.exec, false); // Conditional branch check
                    if (h.bshuf) { thread_local std::vector<u8> bsTmp; byte_plane_shuffle_delta(seg.data(), L, h.bshuf, false, &bsTmp); } // Reusable thread-local scratch
                    if (hint.ate > hint.ats) AudioCodec(h.audio.channels, h.audio.width, fr.code - 1).inverse(seg.data() + hint.ats, (size_t)((hint.ate - hint.ats) / (u64)h.audio.frame())); // Conditional branch check
                } // Close scope / block
                else { std::memcpy(seg.data(), payload.data(), L); sh.cn.done.fetch_add(L); sh.cn.out.fetch_add(L); } // Fallback branch
                if (crc32(seg.data(), L) != fr.crc) throw Failure{Status::Corrupt, "file is damaged: checksum does not match"}; // Conditional branch check

                bool doDrain = false; // Compute and assign value
                { // Open scope / block
                    sys::Lock l(sh.m); // Execute statement
                    while (idx != sh.nextWrite && pending.size() >= maxPending && !sh.cn.cancel.load()) sh.cv.wait(sh.m); // Loop while condition holds
                    if (sh.cn.cancel.load()) return; // Conditional branch check
                    pending.emplace(idx, std::move(seg)); // Execute statement
                    if (!draining && pending.count(sh.nextWrite)) { draining = true; doDrain = true; } // Conditional branch check
                } // Close scope / block
                while (doDrain) { // Loop while condition holds
                    std::vector<u8> cur; // Allocate / manage dynamic buffer
                    { // Open scope / block
                        sys::Lock l(sh.m); // Execute statement
                        auto it = pending.find(sh.nextWrite); // Compute and assign value
                        if (it == pending.end() || sh.cn.cancel.load()) { draining = false; sh.cv.notify_all(); break; } // Conditional branch check
                        cur = std::move(it->second); // Compute and assign value
                        pending.erase(it); // Execute statement
                    } // Close scope / block
                    try { writer.append(cur.data(), cur.size()); } // Begin exception-guarded block
                    catch (...) { sys::Lock l(sh.m); draining = false; sh.cv.notify_all(); throw; } // Handle caught exception
                    { // Open scope / block
                        sys::Lock l(sh.m); // Execute statement
                        sh.nextWrite++; // Execute statement
                        sh.cv.notify_all(); // Execute statement
                    } // Close scope / block
                } // Close scope / block
            } // Close scope / block
        }; // Close scope / block
        run_workers(sh, threads, streamLen, worker, progress); // Execute statement
        if (sh.failed) { out.close(); remove_quiet(tmp); return fail(sh.st, sh.msg); } // Conditional branch check
        if (sh.userCancel || sh.nextWrite != nseg) { out.close(); remove_quiet(tmp); return fail(Status::Cancelled, "cancelled"); } // Conditional branch check
        try { writer.finish(); } catch (const Failure& f) { out.close(); remove_quiet(tmp); return fail(f.status, f.msg); } // Begin exception-guarded block
        if (writer.pos() != (h.defl ? h.xTotal : h.total) || (h.dedup && writer.crc() != h.fileCrc)) { out.close(); remove_quiet(tmp); return fail(Status::Corrupt, "file is damaged: checksum does not match"); } // Conditional branch check
        if (std::fflush(out.f) != 0) { out.close(); remove_quiet(tmp); return fail(Status::DiskFull, "write failed (disk full?)"); } // Conditional branch check
        out.close(); // Execute statement
        if (h.defl) {                                            // second pass: put the deflate streams back
            fs::path tmp2 = finalPath; tmp2 += ".rb.tmp"; // Compute and assign value
            { File xin, fo; if (!xin.open(tmp, "rb") || !fo.open(tmp2, "wb")) { remove_quiet(tmp); remove_quiet(tmp2); return fail(Status::IoError, "cannot create the output file (folder not writable?)"); } // Execute statement
              u32 crc = 0; u64 cnt = 0; std::string err; Status es = Status::Ok; // Compute and assign value
              try { zl::rebuild(xin.f, h.xTotal, drecs, [&](const u8* q, size_t k) { if (std::fwrite(q, 1, k, fo.f) != k) throw Failure{Status::DiskFull, "write failed (disk full?)"}; crc = crc32_update(crc, q, k); cnt += k; }); } // Begin exception-guarded block
              catch (const Failure& fl) { err = fl.msg; es = fl.status; } // Handle caught exception
              if (es == Status::Ok && std::fflush(fo.f) != 0) { err = "write failed (disk full?)"; es = Status::DiskFull; } // Conditional branch check
              if (es == Status::Ok && (cnt != h.total || crc != h.fileCrc)) { err = "file is damaged: checksum does not match"; es = Status::Corrupt; } // Conditional branch check
              fo.close(); xin.close(); // Execute statement
              if (es != Status::Ok) { remove_quiet(tmp); remove_quiet(tmp2); return fail(es, err); } } // Conditional branch check
            remove_quiet(tmp); tmp = tmp2; // Compute and assign value
        } // Close scope / block
        fs::rename(tmp, finalPath, ec); // Execute statement
        if (ec) { remove_quiet(tmp); return fail(Status::IoError, "cannot finalize the output file: " + ec.message()); } // Conditional branch check
        Result r; r.inSize = fileSize; r.outSize = h.total; r.outPath = finalPath; r.storedName = h.name; // Compute and assign value
        if (progress) progress(1.0, h.total); // Conditional branch check
        return r; // Return result to caller
    } catch (const std::bad_alloc&) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::OutOfMemory, "not enough memory"); // Execute statement
    } catch (const std::exception& e) { if (!tmp.empty()) remove_quiet(tmp); return fail(Status::IoError, e.what()); } // Execute statement
} // Close scope / block

// Reads only the metadata: stored name and original size (instant, any file size).
inline Result inspect_file(const fs::path& src) { // Define helper function or constant
    using namespace detail; // Type alias or namespace import
    try { // Begin exception-guarded block
        std::error_code ec; u64 fileSize = (u64)fs::file_size(src, ec); if (ec) return fail(Status::IoError, ec.message()); // Compute and assign value
        File f; if (!f.open(src, "rb")) return fail(Status::IoError, "cannot open the file"); // Execute statement
        Header h; if (!read_header(f.f, h, fileSize)) return fail(Status::BadFormat, "not a valid .3p file"); // Execute statement
        Result r; r.inSize = fileSize; r.outSize = h.total; r.storedName = h.name; return r; // Compute and assign value
    } catch (const std::exception& e) { return fail(Status::IoError, e.what()); } // Execute statement
} // Close scope / block

}  // namespace p3
