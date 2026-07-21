#include "text_preprocess.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using tts_cpp::chatterbox::text_preprocess::decode_utf8;

namespace {

std::vector<uint32_t> utf8_to_codepoints(const std::string & s) {
    return decode_utf8(s);
}

bool is_punct_cp(uint32_t cp) {
    static const std::array<uint32_t, 41> kDrop = {{
        '.', ',', ';', ':', '!', '?', '"', '\'', '`',
        '(', ')', '[', ']', '{', '}', '<', '>',
        '-', '_', '/', '\\', '|', '*', '#', '@', '&',
        0x2018, 0x2019, 0x201C, 0x201D, 0x2013, 0x2014, 0x2026,
        0x3001, 0x3002, 0xFF0C, 0xFF0E, 0xFF1F, 0xFF01, 0x300C,
    }};
    for (uint32_t p : kDrop) if (p == cp) return true;
    return false;
}

bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x00A0 || cp == 0x3000;
}

uint32_t to_lower_ascii(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp - 'A' + 'a';
    return cp;
}

void append_normalized_cp(uint32_t cp, bool & prev_space, std::vector<uint32_t> & out) {
    cp = to_lower_ascii(cp);
    if (is_punct_cp(cp)) return;
    if (is_space_cp(cp)) {
        if (!prev_space) { out.push_back(' '); prev_space = true; }
        return;
    }
    out.push_back(cp);
    prev_space = false;
}

std::vector<uint32_t> normalize(const std::string & s) {
    auto cps = utf8_to_codepoints(s);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    bool prev_space = true;
    for (uint32_t cp : cps) {
        append_normalized_cp(cp, prev_space, out);
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

size_t levenshtein(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    if (a.empty()) return b.size();
    if (b.empty()) return a.size();
    const auto & x = a.size() < b.size() ? a : b;
    const auto & y = a.size() < b.size() ? b : a;
    std::vector<size_t> prev(x.size() + 1), cur(x.size() + 1);

    for (size_t i = 0; i <= x.size(); ++i) prev[i] = i;
    for (size_t j = 1; j <= y.size(); ++j) {
        cur[0] = j;
        for (size_t i = 1; i <= x.size(); ++i) {
            const size_t cost = (x[i - 1] == y[j - 1]) ? 0u : 1u;
            cur[i] = std::min({ prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost });
        }
        std::swap(prev, cur);
    }
    return prev[x.size()];
}

// POSIX single-quote escaping for the std::system() shell command below.
// cmd.exe uses different quoting rules, so build_whisper_command() / run_asr()
// are POSIX-only. These optional L3 tests are only built when whisper-cli +
// a model are present (see CMakeLists.txt) and are run on POSIX CI hosts;
// revisit this quoting if they ever need to run on Windows.
std::string sh_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --whisper-cli PATH --model MODEL.bin --lang CODE\n"
        "             --wav IN.wav --expected TEXT --txt-prefix OUT_PREFIX\n"
        "             [--max-cer 0.40]\n",
        prog);
}

struct Args {
    std::string whisper_cli, model, lang, wav, expected, txt_prefix;
    double max_cer = 0.40;
};

bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if      (a == "--whisper-cli") { auto v = next("--whisper-cli"); if (!v) return false; args.whisper_cli = v; }
        else if (a == "--model")       { auto v = next("--model");       if (!v) return false; args.model       = v; }
        else if (a == "--lang")        { auto v = next("--lang");        if (!v) return false; args.lang        = v; }
        else if (a == "--wav")         { auto v = next("--wav");         if (!v) return false; args.wav         = v; }
        else if (a == "--expected")    { auto v = next("--expected");    if (!v) return false; args.expected    = v; }
        else if (a == "--txt-prefix")  { auto v = next("--txt-prefix");  if (!v) return false; args.txt_prefix  = v; }
        else if (a == "--max-cer")     { auto v = next("--max-cer");     if (!v) return false; args.max_cer     = std::atof(v); }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

std::string build_whisper_command(const Args & args) {
    std::string cmd;
    cmd  = sh_quote(args.whisper_cli);
    cmd += " -m "  + sh_quote(args.model);
    cmd += " -l "  + sh_quote(args.lang);
    cmd += " -f "  + sh_quote(args.wav);
    cmd += " -otxt";
    cmd += " -of " + sh_quote(args.txt_prefix);
    cmd += " -nt -np";
    cmd += " 2>&1";
    return cmd;
}

int run_whisper(const Args & args) {
    std::string cmd = build_whisper_command(args);
    fprintf(stderr, "$ %s\n", cmd.c_str());
    return std::system(cmd.c_str());
}

int evaluate_transcript(const Args & args) {
    const std::string txt_path = args.txt_prefix + ".txt";
    std::string transcript = read_file(txt_path);
    if (transcript.empty()) {
        fprintf(stderr, "ASR FAIL: transcript file empty / missing: %s\n", txt_path.c_str());
        return 2;
    }

    const auto exp_n = normalize(args.expected);
    const auto got_n = normalize(transcript);
    if (got_n.empty()) {
        fprintf(stderr, "ASR FAIL: transcript normalized to empty (raw: \"%s\")\n", transcript.c_str());
        return 2;
    }

    // Standard CER normalizes the edit distance by the reference length
    // (not max(ref, hyp)), so values stay comparable with published
    // benchmarks. A hypothesis much longer than the reference can push CER
    // above 1.0, which is the expected behaviour of the metric.
    const size_t dist  = levenshtein(exp_n, got_n);
    const size_t denom = exp_n.size();
    const double cer   = denom == 0 ? 1.0 : double(dist) / double(denom);

    fprintf(stderr, "expected: \"%s\"\n", args.expected.c_str());
    fprintf(stderr, "got     : \"%s\"\n", transcript.c_str());
    fprintf(stderr, "CER=%.3f (dist=%zu, ref_len=%zu, hyp_len=%zu, threshold=%.3f)\n",
            cer, dist, exp_n.size(), got_n.size(), args.max_cer);

    if (cer > args.max_cer) {
        fprintf(stderr, "RESULT: FAIL (CER %.3f > %.3f)\n", cer, args.max_cer);
        return 3;
    }
    fprintf(stderr, "RESULT: PASS (%s, CER %.3f)\n", args.lang.c_str(), cer);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 64;

    if (args.whisper_cli.empty() || args.model.empty() || args.lang.empty() ||
        args.wav.empty() || args.expected.empty() || args.txt_prefix.empty()) {
        usage(argv[0]); return 64;
    }

    fprintf(stderr, "test-multilingual-asr: lang=%s wav=%s\n", args.lang.c_str(), args.wav.c_str());

    int rc = run_whisper(args);
    if (rc != 0) {
        fprintf(stderr, "whisper-cli exited with %d\n", rc);
        return 1;
    }

    return evaluate_transcript(args);
}
