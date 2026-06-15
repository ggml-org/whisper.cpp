#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::chatterbox::text_preprocess {

std::vector<uint32_t> decode_utf8(const std::string & text);
std::string           encode_codepoint(uint32_t cp);

std::string decompose_korean_to_jamo(const std::string & text);
std::string convert_katakana_to_hiragana(const std::string & text);

// Split `text` into TTS-friendly segments of at most `max_chars` bytes:
// sentence-split on . ? !, soft-break over-long sentences at , : ;, then
// greedily merge short fragments.  Byte-oriented (max_chars counts bytes,
// not codepoints).  Returns {text} unchanged when text is empty or
// max_chars <= 0.  Bounds T3 sequence length (prosody) and per-chunk
// streaming cost; shared by the CLI and the Engine.
std::vector<std::string> split_text_for_tts(const std::string & text, int max_chars);

// Append `src` PCM onto `dst`, crossfading the trailing/leading `fade_ms`
// via a raised-cosine ramp to remove clicks at segment seams.  Falls back to
// a plain append when `dst` is empty or `fade_ms <= 0`.
void append_pcm_crossfade(std::vector<float> & dst, const std::vector<float> & src,
                          int sr, int fade_ms);

class CangjieTable {
public:
    CangjieTable() = default;
    void load(const std::filesystem::path & tsv_path);
    bool empty() const { return m_table.empty(); }
    size_t size()  const { return m_table.size(); }
    std::string convert(const std::string & text) const;

private:
    struct Entry { std::string code; std::string suffix; };
    std::unordered_map<uint32_t, Entry> m_table;

    void read_tsv_entries(std::istream & f,
                          std::unordered_map<std::string, std::vector<uint32_t>> & code_to_cps);
    void assign_disambiguation_suffixes(
            const std::unordered_map<std::string, std::vector<uint32_t>> & code_to_cps);
    static void emit_cangjie_tokens(const Entry & entry, std::string & out);
};

#ifdef TTS_CPP_HAS_MECAB
class MeCabTagger {
public:
    MeCabTagger();
    ~MeCabTagger();
    MeCabTagger(const MeCabTagger &)            = delete;
    MeCabTagger & operator=(const MeCabTagger &) = delete;
    MeCabTagger(MeCabTagger &&) noexcept;
    MeCabTagger & operator=(MeCabTagger &&) noexcept;

    void load(const std::filesystem::path & dic_dir);
    bool loaded() const;
    std::string convert(const std::string & text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
#endif

} // namespace tts_cpp::chatterbox::text_preprocess
