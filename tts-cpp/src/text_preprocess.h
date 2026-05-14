#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::chatterbox::text_preprocess {

std::vector<uint32_t> decode_utf8(const std::string & text);
std::string           encode_codepoint(uint32_t cp);

std::string decompose_korean_to_jamo(const std::string & text);
std::string convert_katakana_to_hiragana(const std::string & text);

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

    void read_tsv_entries(std::ifstream & f,
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
