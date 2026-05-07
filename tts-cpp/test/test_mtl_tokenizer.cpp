// Validation harness for the MTL tokenizer C++ port.
//
// Usage:
//   ./build/test-mtl-tokenizer <tokenizer.json> <golden.json>
//
// The golden JSON is produced by scripts/compare-mtl-tokenizer.py. If it is
// missing, the test is skipped (returns 0) so CI can land the binary before
// the fixtures arrive.

#include "mtl_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

namespace {

struct tiny_json {
    const char * p;
    const char * end;

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool match(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }

    std::string parse_string() {
        skip_ws();
        if (p >= end || *p != '"') throw std::runtime_error("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) throw std::runtime_error("bad escape");
                char c = *p++;
                switch (c) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': {
                        if (p + 4 > end) throw std::runtime_error("short \\u");
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = p[i];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                        }
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= end && p[0] == '\\' && p[1] == 'u') {
                            p += 2;
                            uint32_t cp2 = 0;
                            for (int i = 0; i < 4; ++i) {
                                char h = p[i];
                                cp2 <<= 4;
                                if (h >= '0' && h <= '9') cp2 |= (h - '0');
                                else if (h >= 'a' && h <= 'f') cp2 |= (h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') cp2 |= (h - 'A' + 10);
                            }
                            p += 4;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                        }
                        if (cp < 0x80) out.push_back((char) cp);
                        else if (cp < 0x800) {
                            out.push_back((char) (0xC0 | (cp >> 6)));
                            out.push_back((char) (0x80 | (cp & 0x3F)));
                        } else if (cp < 0x10000) {
                            out.push_back((char) (0xE0 | (cp >> 12)));
                            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back((char) (0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back((char) (0xF0 | (cp >> 18)));
                            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
                            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back((char) (0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: throw std::runtime_error("bad escape");
                }
            } else {
                out.push_back(*p++);
            }
        }
        if (p >= end) throw std::runtime_error("unterminated string");
        ++p;
        return out;
    }

    int64_t parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
        return neg ? -v : v;
    }

    void skip_value() {
        skip_ws();
        if (p >= end) return;
        if (*p == '"') { parse_string(); return; }
        if (*p == '{') {
            ++p;
            while (p < end) {
                skip_ws();
                if (match('}')) return;
                parse_string();
                skip_ws();
                match(':');
                skip_value();
                skip_ws();
                match(',');
            }
            return;
        }
        if (*p == '[') {
            ++p;
            while (p < end) {
                skip_ws();
                if (match(']')) return;
                skip_value();
                skip_ws();
                match(',');
            }
            return;
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != '\n' && *p != ' ' && *p != '\t' && *p != '\r') ++p;
    }
};

struct test_case {
    std::string          lang;
    std::string          text;
    std::vector<int32_t> expected_ids;
};

static std::vector<test_case> parse_golden(const std::string & path, bool & exists) {
    std::ifstream f(path);
    exists = f.good();
    std::vector<test_case> cases;
    if (!exists) return cases;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string blob = ss.str();
    tiny_json jp{blob.data(), blob.data() + blob.size()};

    jp.skip_ws();
    if (!jp.match('{')) throw std::runtime_error("golden: expected {");
    while (jp.p < jp.end) {
        jp.skip_ws();
        if (jp.match('}')) break;
        std::string key = jp.parse_string();
        jp.skip_ws();
        jp.match(':');
        if (key == "cases") {
            jp.skip_ws();
            if (!jp.match('[')) throw std::runtime_error("golden: expected [");
            while (jp.p < jp.end) {
                jp.skip_ws();
                if (jp.match(']')) break;
                if (!jp.match('{')) throw std::runtime_error("golden case: expected {");
                test_case tc;
                while (jp.p < jp.end) {
                    jp.skip_ws();
                    if (jp.match('}')) break;
                    std::string k = jp.parse_string();
                    jp.skip_ws();
                    jp.match(':');
                    if (k == "lang") tc.lang = jp.parse_string();
                    else if (k == "text") tc.text = jp.parse_string();
                    else if (k == "ids") {
                        jp.skip_ws();
                        if (!jp.match('[')) throw std::runtime_error("golden ids: expected [");
                        while (jp.p < jp.end) {
                            jp.skip_ws();
                            if (jp.match(']')) break;
                            tc.expected_ids.push_back((int32_t) jp.parse_int());
                            jp.skip_ws();
                            jp.match(',');
                        }
                    } else {
                        jp.skip_value();
                    }
                    jp.skip_ws();
                    jp.match(',');
                }
                cases.push_back(std::move(tc));
                jp.skip_ws();
                jp.match(',');
            }
        } else {
            jp.skip_value();
        }
        jp.skip_ws();
        jp.match(',');
    }
    return cases;
}

static std::string format_ids(const std::vector<int32_t> & ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tokenizer.json> <golden.json>\n", argv[0]);
        return 1;
    }
    const std::string tok_path = argv[1];
    const std::string gold_path = argv[2];

    mtl_tokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        fprintf(stderr, "failed to load tokenizer from %s\n", tok_path.c_str());
        return 1;
    }
    fprintf(stderr, "loaded tokenizer: vocab_size=%d sot=%d eot=%d unk=%d\n",
            tok.vocab_size(), tok.sot_id(), tok.eot_id(), tok.unk_id());

    bool exists = false;
    std::vector<test_case> cases;
    try {
        cases = parse_golden(gold_path, exists);
    } catch (const std::exception & e) {
        fprintf(stderr, "failed to parse golden: %s\n", e.what());
        return 1;
    }

    if (!exists) {
        fprintf(stderr, "golden file %s not found - skipping comparison\n", gold_path.c_str());
        return 0;
    }
    if (cases.empty()) {
        fprintf(stderr, "golden file had no cases - nothing to check\n");
        return 0;
    }

    int pass = 0;
    int fail = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto & tc = cases[i];
        std::vector<int32_t> got;
        try {
            got = tok.encode(tc.text, tc.lang);
        } catch (const std::exception & e) {
            fprintf(stderr, "[%zu] lang=%s text=%s FAIL (exception: %s)\n",
                    i, tc.lang.c_str(), tc.text.c_str(), e.what());
            ++fail;
            continue;
        }
        if (got == tc.expected_ids) {
            ++pass;
            fprintf(stderr, "[%zu] lang=%s PASS (%zu ids)\n", i, tc.lang.c_str(), got.size());
        } else {
            ++fail;
            fprintf(stderr, "[%zu] lang=%s text=%s FAIL\n", i, tc.lang.c_str(), tc.text.c_str());
            fprintf(stderr, "    expected: %s\n", format_ids(tc.expected_ids).c_str());
            fprintf(stderr, "    got:      %s\n", format_ids(got).c_str());
        }
    }
    fprintf(stderr, "summary: %d pass, %d fail, %zu total\n", pass, fail, cases.size());
    return fail == 0 ? 0 : 1;
}
