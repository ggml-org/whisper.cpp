#include "mtl_tokenizer.h"
#include "tts-cpp/tts-cpp.h"

using namespace tts_cpp::chatterbox::detail;

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct WavInfo {
    uint16_t channels        = 0;
    uint32_t sample_rate     = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_bytes      = 0;
    std::vector<int16_t> samples_s16;

    double duration_s() const {
        if (sample_rate == 0 || channels == 0 || bits_per_sample == 0) return 0.0;
        return double(data_bytes) / double(sample_rate * channels * (bits_per_sample / 8));
    }
};

struct AudioStats {
    double rms             = 0.0;
    double peak            = 0.0;
    int    clipped_samples = 0;
    double window_std_mean = 0.0;
    int    n_windows       = 0;
};

bool read_riff_header(std::ifstream & f) {
    char riff[4]; f.read(riff, 4);
    uint32_t riff_size = 0; f.read(reinterpret_cast<char*>(&riff_size), 4);
    char wave[4]; f.read(wave, 4);
    return f && std::memcmp(riff, "RIFF", 4) == 0 && std::memcmp(wave, "WAVE", 4) == 0;
}

bool read_fmt_chunk(std::ifstream & f, uint32_t size, WavInfo & out) {
    uint16_t fmt_tag = 0;
    f.read(reinterpret_cast<char*>(&fmt_tag), 2);
    f.read(reinterpret_cast<char*>(&out.channels), 2);
    f.read(reinterpret_cast<char*>(&out.sample_rate), 4);
    uint32_t byte_rate = 0; uint16_t block_align = 0;
    f.read(reinterpret_cast<char*>(&byte_rate), 4);
    f.read(reinterpret_cast<char*>(&block_align), 2);
    f.read(reinterpret_cast<char*>(&out.bits_per_sample), 2);
    if (size > 16) f.seekg(size - 16, std::ios::cur);
    return fmt_tag == 1;
}

void read_data_chunk(std::ifstream & f, uint32_t size, WavInfo & out) {
    out.data_bytes = size;
    if (out.bits_per_sample == 16 && out.channels >= 1) {
        const size_t n = size / 2;
        out.samples_s16.resize(n);
        f.read(reinterpret_cast<char*>(out.samples_s16.data()), size);
    } else {
        f.seekg(size, std::ios::cur);
    }
}

bool parse_wav_chunks(std::ifstream & f, WavInfo & out, std::string & err) {
    bool have_fmt = false, have_data = false;
    while (f && (!have_fmt || !have_data)) {
        char id[4]; f.read(id, 4);
        uint32_t size = 0; f.read(reinterpret_cast<char*>(&size), 4);
        if (!f) break;

        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (!read_fmt_chunk(f, size, out)) { err = "fmt chunk is not PCM"; return false; }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            read_data_chunk(f, size, out);
            have_data = true;
        } else {
            f.seekg(size, std::ios::cur);
        }
        if (size & 1u) f.seekg(1, std::ios::cur);
    }

    if (!have_fmt)  { err = "missing fmt chunk";  return false; }
    if (!have_data) { err = "missing data chunk"; return false; }
    return true;
}

bool read_wav(const std::string & path, WavInfo & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open"; return false; }
    if (!read_riff_header(f)) { err = "not a RIFF/WAVE file"; return false; }
    return parse_wav_chunks(f, out, err);
}

double compute_rms(const std::vector<int16_t> & samples) {
    double sum_sq = 0.0;
    for (int16_t v : samples) {
        const double f = double(v) / 32768.0;
        sum_sq += f * f;
    }
    return std::sqrt(sum_sq / double(samples.size()));
}

double compute_peak(const std::vector<int16_t> & samples) {
    int peak_abs = 0;
    for (int16_t v : samples) {
        const int abs_v = std::abs(int(v));
        if (abs_v > peak_abs) peak_abs = abs_v;
    }
    return double(peak_abs) / 32768.0;
}

int count_clipped(const std::vector<int16_t> & samples) {
    int count = 0;
    for (int16_t v : samples) {
        if (std::abs(int(v)) >= 32760) count++;
    }
    return count;
}

double compute_single_window_std(const std::vector<int16_t> & samples, size_t offset, size_t win) {
    double mean = 0.0;
    for (size_t j = 0; j < win; ++j) mean += double(samples[offset + j]) / 32768.0;
    mean /= double(win);

    double sq = 0.0;
    for (size_t j = 0; j < win; ++j) {
        const double f = double(samples[offset + j]) / 32768.0;
        sq += (f - mean) * (f - mean);
    }
    return std::sqrt(sq / double(win));
}

double compute_window_std_mean(const std::vector<int16_t> & samples, int & n_windows) {
    const size_t win = 24000 / 4;
    double std_acc = 0.0;
    n_windows = 0;

    for (size_t i = 0; i + win <= samples.size(); i += win) {
        std_acc += compute_single_window_std(samples, i, win);
        n_windows++;
    }
    return n_windows > 0 ? std_acc / double(n_windows) : 0.0;
}

AudioStats compute_stats(const std::vector<int16_t> & samples) {
    AudioStats st;
    if (samples.empty()) return st;

    st.rms             = compute_rms(samples);
    st.peak            = compute_peak(samples);
    st.clipped_samples = count_clipped(samples);
    st.window_std_mean = compute_window_std_mean(samples, st.n_windows);
    return st;
}

void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --t3 T3.gguf --s3gen S3GEN.gguf --lang CODE --text TEXT --out OUT.wav\n"
        "                            [--seed N] [--n-gpu-layers N]\n",
        prog);
}

struct Args {
    std::string t3_path, s3gen_path, lang, text, out_path, tokens_dump_path;
    std::string mecab_dict, cangjie_tsv;
    int seed = 42;
    int n_gpu_layers = 99;
    bool verbose = false;
};

bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if      (a == "--t3")           { auto v = next("--t3");           if (!v) return false; args.t3_path = v; }
        else if (a == "--s3gen")        { auto v = next("--s3gen");        if (!v) return false; args.s3gen_path = v; }
        else if (a == "--lang")         { auto v = next("--lang");         if (!v) return false; args.lang = v; }
        else if (a == "--text")         { auto v = next("--text");         if (!v) return false; args.text = v; }
        else if (a == "--out")          { auto v = next("--out");          if (!v) return false; args.out_path = v; }
        else if (a == "--seed")         { auto v = next("--seed");         if (!v) return false; args.seed = std::atoi(v); }
        else if (a == "--n-gpu-layers") { auto v = next("--n-gpu-layers"); if (!v) return false; args.n_gpu_layers = std::atoi(v); }
        else if (a == "--verbose" || a == "-v") { args.verbose = true; }
        else if (a == "--dump-tokens")  { auto v = next("--dump-tokens"); if (!v) return false; args.tokens_dump_path = v; }
        else if (a == "--mecab-dict")  { auto v = next("--mecab-dict");  if (!v) return false; args.mecab_dict = v; }
        else if (a == "--cangjie-tsv") { auto v = next("--cangjie-tsv"); if (!v) return false; args.cangjie_tsv = v; }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

void resolve_env_fallbacks(Args & args) {
    if (args.t3_path.empty()) {
        const char * env = std::getenv("CHATTERBOX_T3_MTL");
        if (env && *env) args.t3_path = env;
    }
    if (args.s3gen_path.empty()) {
        const char * env = std::getenv("CHATTERBOX_S3GEN");
        if (env && *env) args.s3gen_path = env;
    }
    if (args.mecab_dict.empty()) {
        const char * env = std::getenv("CHATTERBOX_MECAB_DICT");
        if (env && *env) args.mecab_dict = env;
    }
    if (args.cangjie_tsv.empty()) {
        const char * env = std::getenv("CHATTERBOX_CANGJIE_TSV");
        if (env && *env) args.cangjie_tsv = env;
    }
}

int check_language_registry(const std::string & lang) {
    const auto & all_known = mtl_tokenizer::all_known_languages();
    const auto & supported = mtl_tokenizer::supported_languages();
    const bool in_known = std::find(all_known.begin(), all_known.end(), lang) != all_known.end();
    const bool in_tier1 = std::find(supported.begin(), supported.end(), lang) != supported.end();

    fprintf(stderr, "L0: lang=%s known=%d tier1=%d\n", lang.c_str(), in_known, in_tier1);
    if (!in_known) {
        fprintf(stderr, "L0 FAIL: '%s' is not in mtl_tokenizer::all_known_languages().\n", lang.c_str());
        return 1;
    }
    return 0;
}

int run_synthesis(const Args & args) {
    std::vector<std::string> cli_args = {
        "test-multilingual-synth",
        "--model",        args.t3_path,
        "--s3gen-gguf",   args.s3gen_path,
        "--language",     args.lang,
        "--text",         args.text,
        "--out",          args.out_path,
        "--seed",         std::to_string(args.seed),
        "--n-gpu-layers", std::to_string(args.n_gpu_layers),
    };
    if (args.verbose) cli_args.push_back("--verbose");
    if (!args.mecab_dict.empty()) {
        cli_args.push_back("--mecab-dict");
        cli_args.push_back(args.mecab_dict);
    }
    if (!args.cangjie_tsv.empty()) {
        cli_args.push_back("--cangjie-tsv");
        cli_args.push_back(args.cangjie_tsv);
    }
    if (!args.tokens_dump_path.empty()) {
        cli_args.push_back("--output");
        cli_args.push_back(args.tokens_dump_path);
    }

    std::vector<char *> argv_c;
    argv_c.reserve(cli_args.size());
    for (auto & s : cli_args) argv_c.push_back(s.data());

    return tts_cpp_cli_main(int(argv_c.size()), argv_c.data());
}

int validate_wav_structure(const std::string & path, WavInfo & info) {
    std::string err;
    if (!read_wav(path, info, err)) {
        fprintf(stderr, "L1 FAIL: %s (%s)\n", err.c_str(), path.c_str());
        return 3;
    }

    fprintf(stderr, "L1: channels=%u rate=%u bits=%u dur=%.3fs samples_s16=%zu\n",
            info.channels, info.sample_rate, info.bits_per_sample,
            info.duration_s(), info.samples_s16.size());

    if (info.channels != 1)         { fprintf(stderr, "L1 FAIL: expected mono\n"); return 3; }
    if (info.sample_rate != 24000)  { fprintf(stderr, "L1 FAIL: expected 24 kHz\n"); return 3; }
    if (info.bits_per_sample != 16) { fprintf(stderr, "L1 FAIL: expected 16-bit PCM\n"); return 3; }
    if (info.duration_s() < 0.5)    { fprintf(stderr, "L1 FAIL: too short (<0.5 s)\n"); return 3; }
    return 0;
}

int validate_audio_sanity(const WavInfo & info) {
    const auto st = compute_stats(info.samples_s16);
    fprintf(stderr, "L2: rms=%.4f peak=%.4f clipped=%d window_std_mean=%.4f n_win=%d\n",
            st.rms, st.peak, st.clipped_samples, st.window_std_mean, st.n_windows);

    constexpr double kMinRms    = 0.005;
    constexpr double kMinWinStd = 0.002;
    constexpr int    kMaxClippedPpm = 50000;
    const int max_clipped = int(double(info.samples_s16.size()) * kMaxClippedPpm / 1'000'000.0);

    if (st.rms < kMinRms) {
        fprintf(stderr, "L2 FAIL: rms %.4f < %.4f\n", st.rms, kMinRms);
        return 4;
    }
    if (st.window_std_mean < kMinWinStd) {
        fprintf(stderr, "L2 FAIL: window_std_mean %.4f < %.4f\n", st.window_std_mean, kMinWinStd);
        return 4;
    }
    if (st.clipped_samples > max_clipped) {
        fprintf(stderr, "L2 FAIL: clipped %d samples > %d\n", st.clipped_samples, max_clipped);
        return 4;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 64;
    resolve_env_fallbacks(args);

    if (args.t3_path.empty() || args.s3gen_path.empty() ||
        args.lang.empty() || args.text.empty() || args.out_path.empty()) {
        usage(argv[0]); return 64;
    }

    fprintf(stderr, "test-multilingual-synth: lang=%s text=\"%s\" out=%s\n",
            args.lang.c_str(), args.text.c_str(), args.out_path.c_str());

    int rc = check_language_registry(args.lang);
    if (rc != 0) return rc;

    rc = run_synthesis(args);
    if (rc != 0) {
        fprintf(stderr, "synthesis FAILED: tts_cpp_cli_main returned %d\n", rc);
        return 2;
    }

    WavInfo info;
    rc = validate_wav_structure(args.out_path, info);
    if (rc != 0) return rc;

    rc = validate_audio_sanity(info);
    if (rc != 0) return rc;

    fprintf(stderr, "RESULT: PASS (%s, %.2f s)\n",
            args.lang.c_str(), info.duration_s());
    return 0;
}
