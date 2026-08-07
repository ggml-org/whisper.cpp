// Audio8 tokenizer parity against HF ground truth.
//
// Reads the vocabulary out of the LM GGUF, so the test covers what the engine
// actually loads rather than a side-loaded tokenizer.json. Cases come from
// scripts/dump-audio8-tokenizer-reference.py.

#include "audio8/tokenizer.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tts_cpp::audio8;

namespace {

class gguf_handle {
public:
    explicit gguf_handle(const char * path) {
        gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/nullptr};
        ctx_ = gguf_init_from_file(path, params);
    }
    ~gguf_handle() { if (ctx_) gguf_free(ctx_); }
    gguf_handle(const gguf_handle &) = delete;
    gguf_handle & operator=(const gguf_handle &) = delete;

    const gguf_context * get() const { return ctx_; }

private:
    gguf_context * ctx_ = nullptr;
};

bool read_strings(const gguf_context * ctx, const char * key, std::vector<std::string> & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::fprintf(stderr, "missing GGUF key %s\n", key);
        return false;
    }
    const int count = static_cast<int>(gguf_get_arr_n(ctx, id));
    out.resize(count);
    for (int index = 0; index < count; ++index) out[index] = gguf_get_arr_str(ctx, id, index);
    return true;
}

bool read_ids(const gguf_context * ctx, const char * key, std::vector<int32_t> & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::fprintf(stderr, "missing GGUF key %s\n", key);
        return false;
    }
    const int count = static_cast<int>(gguf_get_arr_n(ctx, id));
    const int32_t * values = static_cast<const int32_t *>(gguf_get_arr_data(ctx, id));
    out.assign(values, values + count);
    return true;
}

bool load_tokenizer_data(const char * path, TokenizerData & data) {
    gguf_handle file(path);
    if (!file.get()) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    return read_strings(file.get(), "tokenizer.ggml.tokens", data.tokens) &&
           read_strings(file.get(), "tokenizer.ggml.merges", data.merges) &&
           read_ids(file.get(), "tokenizer.ggml.added_token_ids", data.added_token_ids);
}

std::string unescape(const std::string & text) {
    std::string out;
    for (size_t index = 0; index < text.size(); ++index) {
        if (text[index] != '\\' || index + 1 >= text.size()) {
            out += text[index];
            continue;
        }
        const char code = text[++index];
        out += (code == 'n') ? '\n' : (code == 't') ? '\t' : code;
    }
    return out;
}

std::vector<int32_t> parse_ids(const std::string & field) {
    std::vector<int32_t> ids;
    std::stringstream stream(field);
    std::string item;
    while (std::getline(stream, item, ',')) ids.push_back(std::atoi(item.c_str()));
    return ids;
}

void print_ids(const char * label, const std::vector<int32_t> & ids) {
    std::fprintf(stderr, "  %s:", label);
    for (int32_t id : ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n");
}

struct tally {
    int passed = 0;
    int failed = 0;
};

void check_case(const Tokenizer & tokenizer, const std::string & note, const std::string & line,
                tally & counts) {
    const size_t tab = line.find('\t');
    const std::vector<int32_t> want = parse_ids(line.substr(0, tab));
    const std::vector<int32_t> got = tokenizer.encode(unescape(line.substr(tab + 1)));
    if (got == want) {
        ++counts.passed;
        std::printf("  %-34s ok\n", note.c_str());
        return;
    }
    ++counts.failed;
    std::printf("  %-34s FAIL\n", note.c_str());
    print_ids("want", want);
    print_ids("got ", got);
}

bool run_cases(const Tokenizer & tokenizer, const char * path, tally & counts) {
    std::ifstream fixture(path);
    if (!fixture) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    std::string line;
    std::string note;
    while (std::getline(fixture, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            note = line.substr(2);
            continue;
        }
        check_case(tokenizer, note, line, counts);
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <audio8-lm.gguf> <tokenizer_ref.txt>\n", argv[0]);
        return 1;
    }
    TokenizerData data;
    if (!load_tokenizer_data(argv[1], data)) return 1;

    tally counts;
    if (!run_cases(Tokenizer(data), argv[2], counts)) return 1;
    std::printf("\n%d passed, %d failed\n", counts.passed, counts.failed);
    return counts.failed == 0 ? 0 : 1;
}
