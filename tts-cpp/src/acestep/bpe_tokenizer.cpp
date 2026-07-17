#include "bpe_tokenizer.h"

#include "gguf.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>

// Port of acestep.cpp/src/bpe.h (byte-level GPT-2/Qwen3 BPE) + bpe_decode from
// sampling.h. Behaviour is kept identical; only namespacing changed.

namespace tts_cpp::acestep {

// GPT-2 byte-level encoding table: byte [0..255] -> Unicode char (UTF-8 string).
static void build_byte_encoder(std::string byte2str[256]) {
    int bs[256], cs[256], n = 0, total = 0;
    for (int b = '!'; b <= '~'; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[total] = b; cs[total] = b; total++; }
    bool used[256] = {};
    for (int i = 0; i < total; i++) used[bs[i]] = true;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { bs[total] = b; cs[total] = 256 + n; n++; total++; }
    }
    assert(total == 256);
    for (int i = 0; i < 256; i++) {
        int  cp = cs[i];
        char buf[4];
        int  len;
        if (cp < 0x80) {
            buf[0] = (char) cp; len = 1;
        } else if (cp < 0x800) {
            buf[0] = (char) (0xC0 | (cp >> 6));
            buf[1] = (char) (0x80 | (cp & 0x3F));
            len    = 2;
        } else {
            buf[0] = (char) (0xE0 | (cp >> 12));
            buf[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
            buf[2] = (char) (0x80 | (cp & 0x3F));
            len    = 3;
        }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

static int utf8_codepoint(const char * s, int * advance) {
    unsigned char c = s[0];
    if (c < 0x80) { *advance = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *advance = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { *advance = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0) {
        *advance = 4;
        return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *advance = 1;
    return c;
}

static bool is_letter(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80) return false;
    if (cp >= 0xC0 && cp <= 0x024F && cp != 0xD7 && cp != 0xF7) return true;
    if (cp >= 0x0370 && cp <= 0x1FFF) return true;
    if (cp >= 0x2C00 && cp <= 0x2DFF) return true;
    if (cp >= 0x3040 && cp <= 0x9FFF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0x10000) return true;
    return false;
}

static bool is_digit(int cp) { return cp >= '0' && cp <= '9'; }

static bool is_whitespace(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C || cp == 0xA0 ||
           cp == 0x2000 || cp == 0x2001 || cp == 0x2002 || cp == 0x200B;
}

static bool is_newline(int cp) { return cp == '\n' || cp == '\r'; }

// GPT-2 pre-tokenizer (manual regex implementation).
static std::vector<std::string> gpt2_pre_tokenize(const std::string & text) {
    std::vector<std::string> chunks;
    const char *             s   = text.c_str();
    int                      len = (int) text.size();
    int                      i   = 0;

    while (i < len) {
        int adv;
        int cp = utf8_codepoint(s + i, &adv);

        if ((cp == '\'' || cp == 0x2019) && i + adv < len) {
            const char * rest      = s + i + adv;
            int          rlen      = len - i - adv;
            auto         try_match = [&](const char * suffix, int slen) -> bool {
                if (rlen >= slen) {
                    for (int k = 0; k < slen; k++) {
                        char c1 = rest[k], c2 = suffix[k];
                        if (c1 >= 'A' && c1 <= 'Z') c1 = (char) (c1 + 32);
                        if (c1 != c2) return false;
                    }
                    if (rlen > slen) {
                        int a2;
                        int cp2 = utf8_codepoint(rest + slen, &a2);
                        if (is_letter(cp2)) return false;
                    }
                    chunks.push_back(std::string(s + i, adv + slen));
                    i += adv + slen;
                    return true;
                }
                return false;
            };
            if (try_match("ll", 2)) continue;
            if (try_match("re", 2)) continue;
            if (try_match("ve", 2)) continue;
            if (try_match("s", 1)) continue;
            if (try_match("t", 1)) continue;
            if (try_match("m", 1)) continue;
            if (try_match("d", 1)) continue;
        }

        if (is_letter(cp)) {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_letter(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }
        if (!is_newline(cp) && !is_letter(cp) && !is_digit(cp) && !is_whitespace(cp)) {
            int start = i;
            int after = i + adv;
            if (after < len) {
                int a2;
                int cp2 = utf8_codepoint(s + after, &a2);
                if (is_letter(cp2)) {
                    i = after + a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
            }
        }

        if (is_digit(cp)) {
            int start = i;
            while (i < len && is_digit((unsigned char) s[i])) i++;
            for (int j = start; j < i; j++) chunks.push_back(std::string(s + j, 1));
            continue;
        }

        if (is_newline(cp)) {
            int start = i;
            while (i < len && is_newline((unsigned char) s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        if (is_whitespace(cp)) {
            int start  = i;
            int ws_end = i + adv;
            while (ws_end < len && is_whitespace((unsigned char) s[ws_end]) && !is_newline((unsigned char) s[ws_end]))
                ws_end++;
            bool followed_by_non_ws =
                (ws_end < len && !is_whitespace((unsigned char) s[ws_end]) && !is_newline((unsigned char) s[ws_end]));
            if (followed_by_non_ws && ws_end - start > 1) {
                int trailing = ws_end - 1;
                chunks.push_back(std::string(s + start, trailing - start));
                i = trailing;
                continue;
            }
            i = start + adv;
            if (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (is_letter(cp2)) {
                    i += a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (is_digit(cp2)) {
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (!is_whitespace(cp2) && !is_newline(cp2)) {
                    int pstart = start;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (is_whitespace(cp3) || is_letter(cp3) || is_digit(cp3)) break;
                        i += a3;
                    }
                    while (i < len && is_newline((unsigned char) s[i])) i++;
                    chunks.push_back(std::string(s + pstart, i - pstart));
                    continue;
                }
            }
            i = ws_end;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_whitespace(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (is_whitespace(cp2) || is_letter(cp2) || is_digit(cp2) || is_newline(cp2)) break;
                i += a2;
            }
            while (i < len && is_newline((unsigned char) s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
        }
    }
    return chunks;
}

bool bpe_load_from_gguf(BpeTokenizer & tok, const std::string & gguf_path) {
    build_byte_encoder(tok.byte2str);

    gguf_init_params gp  = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context *   ctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!ctx) {
        fprintf(stderr, "[bpe] failed to open %s\n", gguf_path.c_str());
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        fprintf(stderr, "[bpe] tokenizer KV not found in %s\n", gguf_path.c_str());
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int) gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int) gguf_get_arr_n(ctx, mrg_key);
    for (int i = 0; i < n_tokens; i++) tok.vocab[std::string(gguf_get_arr_str(ctx, tok_key, (size_t) i))] = i;
    for (int i = 0; i < n_merges; i++) tok.merges[std::string(gguf_get_arr_str(ctx, mrg_key, (size_t) i))] = i;

    gguf_free(ctx);

    tok.n_vocab = (int) tok.vocab.size();
    tok.eos_id  = 151643;
    tok.id_to_str.assign(tok.n_vocab, std::string());
    for (auto & kv : tok.vocab) {
        if (kv.second >= 0 && kv.second < tok.n_vocab) tok.id_to_str[kv.second] = kv.first;
    }
    fprintf(stderr, "[bpe] loaded from GGUF: %d vocab, %d merges\n", tok.n_vocab, n_merges);
    return true;
}

static std::string byte_level_encode(const BpeTokenizer & tok, const std::string & text) {
    std::string out;
    for (unsigned char c : text) out += tok.byte2str[c];
    return out;
}

static std::vector<std::string> bpe_merge(const std::unordered_map<std::string, int> & merge_rank,
                                          const std::vector<std::string> &             symbols) {
    if (symbols.size() <= 1) return symbols;
    std::vector<std::string> work = symbols;
    while (work.size() > 1) {
        int best_rank = INT_MAX, best_pos = -1;
        for (int i = 0; i < (int) work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto        it  = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_pos = i; }
        }
        if (best_pos < 0) break;
        work[best_pos] = work[best_pos] + work[best_pos + 1];
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

static void encode_chunk(const BpeTokenizer & tok, const std::string & chunk, std::vector<int> & ids) {
    std::string              encoded = byte_level_encode(tok, chunk);
    std::vector<std::string> symbols;
    const char *             s   = encoded.c_str();
    int                      len = (int) encoded.size();
    int                      i   = 0;
    while (i < len) {
        int adv;
        utf8_codepoint(s + i, &adv);
        symbols.push_back(std::string(s + i, adv));
        i += adv;
    }
    std::vector<std::string> merged = bpe_merge(tok.merges, symbols);
    for (const auto & piece : merged) {
        auto it = tok.vocab.find(piece);
        if (it != tok.vocab.end()) {
            ids.push_back(it->second);
        } else {
            fprintf(stderr, "[bpe] WARNING: unknown token '%s'\n", piece.c_str());
            for (unsigned char c : piece) {
                auto it2 = tok.vocab.find(std::string(1, c));
                if (it2 != tok.vocab.end()) ids.push_back(it2->second);
            }
        }
    }
}

std::vector<int> bpe_encode(const BpeTokenizer & tok, const std::string & text, bool add_eos) {
    std::vector<int>  ids;
    const std::string special = "<|endoftext|>";
    size_t            pos     = 0;
    while (pos < text.size()) {
        size_t      found   = text.find(special, pos);
        std::string segment = (found == std::string::npos) ? text.substr(pos) : text.substr(pos, found - pos);
        if (!segment.empty()) {
            auto chunks = gpt2_pre_tokenize(segment);
            for (const auto & chunk : chunks) encode_chunk(tok, chunk, ids);
        }
        if (found == std::string::npos) break;
        ids.push_back(tok.eos_id);
        pos = found + special.size();
    }
    if (add_eos) ids.push_back(tok.eos_id);
    return ids;
}

std::string bpe_decode(const BpeTokenizer & tok, const std::vector<int> & ids) {
    // byte decoder: GPT-2 codepoint -> raw byte
    std::unordered_map<int, uint8_t> byte_dec;
    for (int b = 0; b < 256; b++) {
        int adv;
        int cp       = utf8_codepoint(tok.byte2str[b].c_str(), &adv);
        byte_dec[cp] = (uint8_t) b;
    }

    std::string result;
    for (int id : ids) {
        if (id == TOKEN_THINK) { result += "<think>"; continue; }
        if (id == TOKEN_THINK_END) { result += "</think>"; continue; }
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) continue;
        if (id >= AUDIO_CODE_BASE) continue;
        if (id < 0 || id >= (int) tok.id_to_str.size()) continue;
        const std::string & str = tok.id_to_str[id];
        if (str.empty()) continue;
        const char * p = str.c_str();
        while (*p) {
            int  adv;
            int  cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) result += (char) it->second;
            p += adv;
        }
    }
    return result;
}

int bpe_utf8_codepoint(const char * s, int * advance) { return utf8_codepoint(s, advance); }

} // namespace tts_cpp::acestep
