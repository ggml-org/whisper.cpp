// End-to-end round-trip regression test: synthesize -> ASR -> compare.
//
// This is the end-to-end guard for the end-of-speech fix.  For a fixed set of
// English phrases it drives `tts-cli` to synthesize a wav, transcribes it with
// `whisper-cli`, and checks that the transcription:
//   * is close to the input text  (CER <= --max-cer)  -> catches CLIPPING
//     (missing trailing words inflate the edit distance), and
//   * is not much longer than the input (hyp/ref char ratio <= --max-ramble)
//     -> catches RAMBLING (the trailing-token bug re-appearing dumps lots of
//     extra transcribed words).
//
// Pure orchestration via subprocess (POSIX), mirroring test_multilingual_asr;
// no model is linked.  CMake registers it disabled unless the Chatterbox MTL
// GGUFs + whisper-cli + a Whisper model are all present, so a fixtureless
// checkout still gets a green ctest.
//
// whisper.cpp reads the 24 kHz synthesized wav directly (miniaudio resamples
// to 16 kHz internally), so no explicit resample step is needed here.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string sh_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; }
    out += "'";
    return out;
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Lowercase, drop ASCII punctuation, collapse whitespace -> char vector.
std::vector<char> normalize(const std::string & s) {
    std::vector<char> out;
    bool prev_space = true;
    for (unsigned char uc : s) {
        char c = (char) uc;
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        const bool is_alnum = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        if (is_alnum) { out.push_back(c); prev_space = false; }
        else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!prev_space) { out.push_back(' '); prev_space = true; }
        }
        // other punctuation dropped
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

size_t levenshtein(const std::vector<char> & a, const std::vector<char> & b) {
    if (a.empty()) return b.size();
    if (b.empty()) return a.size();
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t cost = (a[i - 1] == b[j - 1]) ? 0u : 1u;
            cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost });
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

struct Args {
    std::string tts_cli, t3, s3gen, ref_dir, whisper_cli, whisper_model;
    std::string lang = "en";
    std::string tmp = "/tmp";
    int    gpu_layers = 0;
    int    seed = 0;
    double max_cer = 0.50;     // generous: this is a clip/ramble gate, not an ASR-quality gate
    double max_ramble = 2.5;   // hyp chars / ref chars upper bound (no rambling)
};

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&](const char * n) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if      (k == "--tts-cli")       { auto v = val(k.c_str()); if (!v) return false; a.tts_cli = v; }
        else if (k == "--t3")            { auto v = val(k.c_str()); if (!v) return false; a.t3 = v; }
        else if (k == "--s3gen")         { auto v = val(k.c_str()); if (!v) return false; a.s3gen = v; }
        else if (k == "--ref-dir")       { auto v = val(k.c_str()); if (!v) return false; a.ref_dir = v; }
        else if (k == "--whisper-cli")   { auto v = val(k.c_str()); if (!v) return false; a.whisper_cli = v; }
        else if (k == "--whisper-model") { auto v = val(k.c_str()); if (!v) return false; a.whisper_model = v; }
        else if (k == "--lang")          { auto v = val(k.c_str()); if (!v) return false; a.lang = v; }
        else if (k == "--tmp")           { auto v = val(k.c_str()); if (!v) return false; a.tmp = v; }
        else if (k == "--gpu-layers")    { auto v = val(k.c_str()); if (!v) return false; a.gpu_layers = std::atoi(v); }
        else if (k == "--seed")          { auto v = val(k.c_str()); if (!v) return false; a.seed = std::atoi(v); }
        else if (k == "--max-cer")       { auto v = val(k.c_str()); if (!v) return false; a.max_cer = std::atof(v); }
        else if (k == "--max-ramble")    { auto v = val(k.c_str()); if (!v) return false; a.max_ramble = std::atof(v); }
        else { fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    return !a.tts_cli.empty() && !a.t3.empty() && !a.s3gen.empty() &&
           !a.whisper_cli.empty() && !a.whisper_model.empty();
}

// English phrases (varied length) that base.en transcribes reliably.  Avoid
// single-syllable words, which are inherently ASR-ambiguous on synthesized
// audio (the short-input no-clip invariant is covered deterministically by the
// model-free unit tests instead).
const std::array<const char *, 5> kPhrases = {{
    "hello how are you",
    "what time is it",
    "thank you very much for your help",
    "the quick brown fox jumps over the lazy dog",
    "welcome to the event we are glad you are here",
}};

int synth(const Args & a, const std::string & text, const std::string & wav) {
    std::string cmd = sh_quote(a.tts_cli)
        + " --model " + sh_quote(a.t3)
        + " --s3gen-gguf " + sh_quote(a.s3gen)
        + " --language " + sh_quote(a.lang)
        + " --text " + sh_quote(text)
        + " --out " + sh_quote(wav)
        + " --seed " + std::to_string(a.seed)
        + " --threads 16";
    if (!a.ref_dir.empty()) cmd += " --ref-dir " + sh_quote(a.ref_dir);
    if (a.gpu_layers > 0)   cmd += " --n-gpu-layers " + std::to_string(a.gpu_layers);
    cmd += " >/dev/null 2>&1";
    return std::system(cmd.c_str());
}

int transcribe(const Args & a, const std::string & wav, const std::string & prefix) {
    std::string cmd = sh_quote(a.whisper_cli)
        + " -m " + sh_quote(a.whisper_model)
        + " -l " + sh_quote(a.lang)
        + " -f " + sh_quote(wav)
        + " -otxt -of " + sh_quote(prefix)
        + " -nt -np >/dev/null 2>&1";
    return std::system(cmd.c_str());
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { fprintf(stderr, "usage: test-eos-roundtrip --tts-cli ... --t3 ... --s3gen ... --whisper-cli ... --whisper-model ...\n"); return 64; }

    int failures = 0, idx = 0;
    for (const char * phrase : kPhrases) {
        const std::string wav    = a.tmp + "/eos_rt_" + std::to_string(idx) + ".wav";
        const std::string prefix = a.tmp + "/eos_rt_" + std::to_string(idx);
        ++idx;

        if (synth(a, phrase, wav) != 0) {
            fprintf(stderr, "FAIL [%s]: synthesis failed\n", phrase);
            ++failures; continue;
        }
        if (transcribe(a, wav, prefix) != 0) {
            fprintf(stderr, "FAIL [%s]: whisper failed\n", phrase);
            ++failures; continue;
        }
        const std::string hyp = read_file(prefix + ".txt");
        const auto ref_n = normalize(phrase);
        const auto hyp_n = normalize(hyp);
        if (hyp_n.empty()) {
            fprintf(stderr, "FAIL [%s]: empty transcription\n", phrase);
            ++failures; continue;
        }
        const size_t dist = levenshtein(ref_n, hyp_n);
        const double cer  = ref_n.empty() ? 1.0 : double(dist) / double(ref_n.size());
        const double ramble = double(hyp_n.size()) / double(std::max<size_t>(ref_n.size(), 1));

        const bool ok = (cer <= a.max_cer) && (ramble <= a.max_ramble);
        fprintf(stderr, "%s [%s] CER=%.2f ramble=%.2f  in=\"%s\"  asr=\"%s\"\n",
                ok ? "PASS" : "FAIL", a.lang.c_str(), cer, ramble, phrase,
                std::string(hyp_n.begin(), hyp_n.end()).c_str());
        if (!ok) {
            ++failures;
            if (cer > a.max_cer)    fprintf(stderr, "    -> CER %.2f > %.2f (possible clipping/garble)\n", cer, a.max_cer);
            if (ramble > a.max_ramble) fprintf(stderr, "    -> ramble %.2f > %.2f (extra trailing content)\n", ramble, a.max_ramble);
        }
    }

    fprintf(stderr, "\n%s: %d/%zu phrases passed\n",
            failures == 0 ? "PASS" : "FAIL", int(kPhrases.size()) - failures, kPhrases.size());
    return failures == 0 ? 0 : 1;
}
