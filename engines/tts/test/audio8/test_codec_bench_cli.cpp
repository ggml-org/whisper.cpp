#include "audio8/codec_bench_cli.h"
#include <cstdio>
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

static bool accepts(std::initializer_list<const char *> args) {
    std::vector<std::string> strings{"audio8-codec-bench"};
    for (const auto * arg : args) strings.emplace_back(arg);
    std::vector<char *> argv;
    for (auto & arg : strings) argv.push_back(&arg[0]);
    try { audio8_codec_bench::parse(static_cast<int>(argv.size()), argv.data()); return true; }
    catch (const std::exception &) { return false; }
}

static bool decode_accepts(std::initializer_list<const char *> extra) {
    std::vector<std::string> strings{"audio8-codec-bench", "--codec-decoder", "dec", "--codes", "codes.csv",
                                     "--codec-backend", "cpu", "--frames", "3", "--out-f32", "out.f32"};
    for (const auto * arg : extra) strings.emplace_back(arg);
    std::vector<char *> argv;
    for (auto & arg : strings) argv.push_back(&arg[0]);
    try { audio8_codec_bench::parse(static_cast<int>(argv.size()), argv.data()); return true; }
    catch (const std::exception &) { return false; }
}

template <typename F> static bool rejects(F && f) {
    try { f(); return false; }
    catch (const std::exception &) { return true; }
}

struct temporary_directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audio8-codec-bench-cli-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    temporary_directory() { std::filesystem::create_directory(path); }
    ~temporary_directory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

int main() {
    int failed = 0;
    auto check = [&](bool result, const char * description) {
        if (!result) { std::fprintf(stderr, "FAIL: %s\n", description); ++failed; }
    };
    check(accepts({"--help"}), "help without models");
    check(!accepts({}), "missing required paths");
    check(!accepts({"--bogus"}), "unknown option");
    check(!accepts({"--in"}), "missing option value");
    check(!accepts({"--in", "--out", "x"}), "option consumed as value");
    check(accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--threads", "2", "--n-gpu-layers", "99"}), "valid reconstruction args");
    for (const auto * n : {"0", "-1", "2x", "1.5", "2147483648", "9999999999999999999999"})
        check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--threads", n}), "invalid threads");
    check(accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--n-gpu-layers", "0"}), "explicit CPU");
    for (const auto * n : {"-1", "99x", "2147483648"})
        check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--n-gpu-layers", n}), "invalid GPU layers");
    check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "in.wav"}), "input overwrite");
    check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "enc"}), "model overwrite");
    check(decode_accepts({}), "CSV decode with inferred codebook count");
    check(decode_accepts({"--mode", "decode", "--codebooks", "10", "--warmup", "0", "--repeat", "1", "--out", "out.wav"}),
          "explicit decode with both PCM outputs and zero warmup");
    check(accepts({"--mode", "decode", "--codec-decoder", "dec", "--codes", "codes", "--backend", "hexagon",
                   "--frames", "66", "--out", "out.wav", "--warmups", "2", "--repeats", "3", "--codes-format", "i32"}),
          "decoder option aliases and i32 format");
    check(accepts({"--codec-decoder", "dec", "--codes", "codes", "--codec-backend", "opencl",
                   "--frames", "3", "--out-f32", "out.f32"}), "OpenCL decode");
    check(!decode_accepts({"--mode", "invalid"}), "invalid mode");
    check(!decode_accepts({"--mode", "decode", "--mode", "decode"}), "duplicate mode");
    check(!decode_accepts({"--mode", "roundtrip"}), "roundtrip conflicts with codes");
    check(!decode_accepts({"--codec-encoder", "enc"}), "decode conflicts with encoder");
    check(!decode_accepts({"--in", "in.wav"}), "decode conflicts with WAV input");
    check(!decode_accepts({"--n-gpu-layers", "0"}), "decode conflicts with automatic backend control");
    check(!decode_accepts({"--backend", "hexagon"}), "duplicate backend alias");
    check(!decode_accepts({"--warmup", "1", "--warmups", "2"}), "duplicate warmup alias");
    check(!decode_accepts({"--repeat", "1", "--repeats", "2"}), "duplicate repeat alias");
    for (const auto * n : {"0", "-1", "2x", "1.5", "2147483648", "9999999999999999999999"}) {
        check(!decode_accepts({"--frames", n}), "invalid frames");
        check(!decode_accepts({"--codebooks", n}), "invalid codebooks");
        check(!decode_accepts({"--repeat", n}), "invalid repeat");
    }
    for (const auto * n : {"-1", "2x", "2147483648"})
        check(!decode_accepts({"--warmup", n}), "invalid warmup");
    check(!decode_accepts({"--codes-format", "binary"}), "invalid codes format");
    for (const auto * backend : {"auto", "vulkan", "HTP0", "CPU"})
        check(!accepts({"--codec-decoder", "dec", "--codes", "codes", "--codec-backend", backend,
                       "--frames", "3", "--out-f32", "out.f32"}), "invalid benchmark backend");
    check(!accepts({"--codec-decoder", "dec", "--codes", "codes", "--frames", "3", "--out-f32", "out.f32"}),
          "decode requires explicit backend");
    check(!accepts({"--codec-decoder", "dec", "--codes", "codes", "--codec-backend", "cpu", "--out-f32", "out.f32"}),
          "decode requires frame count");
    check(!accepts({"--codec-decoder", "dec", "--codes", "codes", "--codec-backend", "cpu", "--frames", "3"}),
          "decode requires PCM output");
    check(!decode_accepts({"--out-f32", "codes.csv"}), "codes overwrite");
    check(!decode_accepts({"--out-f32", "dec"}), "decoder overwrite");
    check(!decode_accepts({"--out", "out.f32"}), "WAV and raw PCM path conflict");

    namespace bench = audio8_codec_bench;
    check(bench::parse_code_row(" 7, 2,3\r", 3) == std::vector<int32_t>({7, 2, 3}), "CSV whitespace and CRLF");
    check(bench::parse_code_row("-1,2147483647,-2147483648", 3) ==
          std::vector<int32_t>({-1, INT32_MAX, INT32_MIN}), "CSV signed int32 boundaries");
    for (const auto * row : {"", "1,2", "1,2,3,4", "1,,3", "1,2,", "1,2,3,", "1,2,1.5",
                            "1,2,3x", "1,2,+3", "1,2,-", "1,2,2147483648", "1,2,-2147483649",
                            "1,2,999999999999999999999999"})
        check(rejects([&] { bench::parse_code_row(row, 3); }), "malformed CSV row");
    check(!rejects([] { bench::validate_codes({7, 3, 0, 6, 2, 1}, 2, 3, 8, 4); }), "valid semantic and residual ranges");
    check(rejects([] { bench::validate_codes({8, 3, 0}, 1, 3, 8, 4); }), "semantic upper bound");
    check(rejects([] { bench::validate_codes({7, 4, 0}, 1, 3, 8, 4); }), "residual upper bound");
    check(rejects([] { bench::validate_codes({7, 3, 0, 6, 4, 1}, 2, 3, 8, 4); }), "later frame residual bound");
    check(rejects([] { bench::validate_codes({7, -1, 0}, 1, 3, 8, 4); }), "negative token");
    check(rejects([] { bench::validate_codes({7, 0}, 1, 3, 8, 4); }), "code count mismatch");
    check(rejects([] { bench::validate_codes({7, 0, 0}, 1, 3, 0, 4); }), "invalid model vocabulary");
    check(rejects([] { bench::code_count(0, 3); }), "zero frames in code geometry");
    check(rejects([] { bench::code_count(3, -1); }), "negative codebook geometry");
    check(bench::decoder_rows({7, 3, 0, 6, 2, 1}, 2, 3) == std::vector<int32_t>({7, 6, 3, 2, 0, 1}),
          "frame-major to production codebook-major conversion");
    check(rejects([] { bench::decoder_rows({7, 3}, 2, 3); }), "transpose rejects partial geometry");

    temporary_directory temp;
    const auto csv = (temp.path / "codes.csv").string();
    auto write_csv = [&](const std::string & data) { std::ofstream file(csv, std::ios::binary); file << data; };
    write_csv("7,3,0\r\n6,2,1\r\n");
    check(bench::read_codes(csv, 2, 3) == std::vector<int32_t>({7, 3, 0, 6, 2, 1}), "reads CLI CSV frames");
    check(rejects([&] { bench::read_codes(csv, 3, 3); }), "CSV too few frames");
    check(rejects([&] { bench::read_codes(csv, 1, 3); }), "CSV too many frames");
    check(rejects([&] { bench::read_codes(csv, 2, 2); }), "CSV codebook mismatch");
    write_csv("7,3,0\n\n");
    check(rejects([&] { bench::read_codes(csv, 2, 3); }), "CSV blank frame rejected");
    const auto binary = (temp.path / "codes.i32").string();
    {
        std::ofstream file(binary, std::ios::binary);
        const unsigned char data[] = {7, 0, 0, 0, 3, 0, 0, 0, 255, 255, 255, 255};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    check(bench::read_codes(binary, 1, 3, "i32") == std::vector<int32_t>({7, 3, -1}), "LE int32 including negative token");
    check(rejects([&] { bench::read_codes(binary, 2, 3, "i32"); }), "binary too few tokens");
    check(rejects([&] { bench::read_codes(binary, 1, 2, "i32"); }), "binary too many tokens");
    { std::ofstream file(binary, std::ios::binary | std::ios::app); file.put('x'); }
    check(rejects([&] { bench::read_codes(binary, 1, 3, "i32"); }), "binary partial token rejected");
    check(rejects([&] { bench::read_codes((temp.path / "missing").string(), 1, 3); }), "missing code file");

    bench::options paths;
    paths.codes = csv;
    paths.output_f32 = (temp.path / "out.f32").string();
    check(!rejects([&] { bench::validate_output_paths(paths); }), "distinct output accepted");
    paths.output_f32 = (temp.path / "." / "codes.csv").string();
    check(rejects([&] { bench::validate_output_paths(paths); }), "normalized input alias rejected");
    const auto symlink = temp.path / "symlink.f32";
    std::error_code link_error;
    std::filesystem::create_symlink(csv, symlink, link_error);
    if (!link_error) {
        paths.output_f32 = symlink.string();
        check(rejects([&] { bench::validate_output_paths(paths); }), "symlink input alias rejected");
    }
    const auto hardlink = temp.path / "hardlink.f32";
    std::filesystem::create_hard_link(csv, hardlink, link_error);
    if (!link_error) {
        paths.output_f32 = hardlink.string();
        check(rejects([&] { bench::validate_output_paths(paths); }), "hard link input alias rejected");
    }
    paths.output_f32 = (temp.path / "out.f32").string();
    paths.output = (temp.path / "." / "out.f32").string();
    check(rejects([&] { bench::validate_output_paths(paths); }), "normalized output aliases rejected");
    const auto dangling = temp.path / "out.wav";
    std::filesystem::create_symlink("out.f32", dangling, link_error);
    if (!link_error) {
        paths.output = dangling.string();
        check(rejects([&] { bench::validate_output_paths(paths); }), "dangling symlink output aliases rejected");
    }
    paths.output = (temp.path / "case.wav").string();
    paths.output_f32 = (temp.path / "CASE.WAV").string();
    check(!std::filesystem::exists(paths.output) && !std::filesystem::exists(paths.output_f32),
          "case alias regression starts with nonexistent outputs");
    check(rejects([&] { bench::validate_output_paths(paths); }), "nonexistent case aliases rejected");
    paths.output.clear();
    paths.output_f32 = (temp.path / "CODES.CSV").string();
    check(rejects([&] { bench::validate_output_paths(paths); }), "case alias of input rejected");
#ifdef _WIN32
    check(bench::output_paths_case_equal(std::filesystem::path(L"\u00e9.wav"),
                                       std::filesystem::path(L"\u00c9.WAV")),
          "Windows Unicode case aliases rejected");
#endif
    if (!failed) std::puts("codec benchmark CLI tests passed");
    return failed ? 1 : 0;
}
