#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::chatterbox::detail {

class mtl_tokenizer {
public:
    bool load_from_json(const std::string & json_blob);
    bool load_from_file(const std::string & path);

    // Encode text with optional language prefix. Returns token IDs.
    // Throws std::runtime_error on unsupported language.
    std::vector<int32_t> encode(const std::string & text,
                                const std::string & language_id = "") const;

    // Decode IDs back to text (best-effort, for debugging).
    std::string decode(const std::vector<int32_t> & ids) const;

    bool is_language_supported(const std::string & lang) const;

    int32_t sot_id() const;
    int32_t eot_id() const;
    int32_t unk_id() const;
    int32_t vocab_size() const;

    // Tier-1 language codes handled natively by this build.
    static const std::vector<std::string> & supported_languages();

    // Full list of language codes the Python reference tokenizer accepts.
    static const std::vector<std::string> & all_known_languages();

private:
    struct added_token {
        std::string content;
        int32_t     id;
    };

    std::unordered_map<std::string, int32_t> m_vocab;
    std::vector<std::string>                 m_id_to_token;
    std::unordered_map<std::string, int32_t> m_bpe_ranks;

    std::vector<added_token> m_added_tokens;

    std::string m_unk_token = "[UNK]";

    int32_t m_sot_id = -1;
    int32_t m_eot_id = -1;
    int32_t m_unk_id = -1;
    int32_t m_space_id = -1;

    void index_vocab();
    void bpe_word(const std::string & word, std::vector<int32_t> & out) const;
};

} // namespace tts_cpp::chatterbox::detail
