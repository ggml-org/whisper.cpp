#pragma once
// ACE-Step metadata FSM (constrained decoding for LM Phase 1) — QVAC-21921.
//
// Port of acestep.cpp/src/metadata-fsm.h, adapted to tts-cpp's BpeTokenizer.
// PrefixTree gives token-level constraints; MetadataFSM enforces the CoT YAML
// structure (bpm -> caption -> duration -> keyscale -> language -> timesignature
// -> </think>) so a bare caption expands into valid, in-distribution metadata.

#include "bpe_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::acestep {

// Prefix tree: maps a token-sequence prefix to the set of valid next tokens.
struct PrefixTree {
    std::map<std::vector<int>, std::vector<int>> nodes;

    void add(const std::vector<int> & seq) {
        for (size_t i = 0; i < seq.size(); i++) {
            std::vector<int> prefix(seq.begin(), seq.begin() + i);
            int              next = seq[i];
            auto &           vec  = nodes[prefix];
            if (std::find(vec.begin(), vec.end(), next) == vec.end()) {
                vec.push_back(next);
            }
        }
    }

    const std::vector<int> * get(const std::vector<int> & prefix) const {
        auto it = nodes.find(prefix);
        return it != nodes.end() ? &it->second : nullptr;
    }
};

struct MetadataFSM {
    enum State {
        BPM_NAME,
        BPM_VALUE,
        CAPTION_NAME,
        CAPTION_VALUE,
        DURATION_NAME,
        DURATION_VALUE,
        KEYSCALE_NAME,
        KEYSCALE_VALUE,
        LANGUAGE_NAME,
        LANGUAGE_VALUE,
        TIMESIG_NAME,
        TIMESIG_VALUE,
        THINK_END,
        CODES,
        DISABLED
    };

    State            state    = DISABLED;
    int              name_pos = 0;
    std::vector<int> value_acc;
    bool             enabled                 = false;
    bool             caption_pending_newline = false;
    bool             skip_caption            = false;

    std::vector<int> bpm_name, caption_name, duration_name;
    std::vector<int> keyscale_name, language_name, timesig_name;
    PrefixTree       bpm_tree, duration_tree, keyscale_tree, language_tree, timesig_tree;
    int              newline_tok   = -1;
    int              think_end_tok = TOKEN_THINK_END;
    int              vocab_size    = 0;

    const BpeTokenizer *             bpe_ptr = nullptr;
    std::unordered_map<int, uint8_t> byte_dec;

    bool        caption_ending = false;
    std::string pending_field_name;

    std::vector<int> inject_queue;

    std::string forced_bpm, forced_duration, forced_keyscale, forced_language, forced_timesig;

    static std::vector<int> tokenize_strip(const BpeTokenizer & bpe, const std::string & full,
                                           const std::string & prefix) {
        std::vector<int> full_tok = bpe_encode(bpe, full, false);
        std::vector<int> pre_tok  = bpe_encode(bpe, prefix, false);
        if (full_tok.size() >= pre_tok.size() && std::equal(pre_tok.begin(), pre_tok.end(), full_tok.begin())) {
            return std::vector<int>(full_tok.begin() + pre_tok.size(), full_tok.end());
        }
        return bpe_encode(bpe, full.substr(prefix.size()), false);
    }

    void build_value_tree(const BpeTokenizer & bpe, PrefixTree & tree, const std::string & field_prefix,
                          const std::vector<std::string> & values) {
        for (auto & val : values) {
            std::string      full = field_prefix + val + "\n";
            std::vector<int> vtok = tokenize_strip(bpe, full, field_prefix);
            tree.add(vtok);
        }
    }

    std::string decode_token(int id) const {
        if (!bpe_ptr || id < 0 || id >= (int) bpe_ptr->id_to_str.size()) {
            return "";
        }
        const std::string & s = bpe_ptr->id_to_str[id];
        std::string         result;
        const char *        p = s.c_str();
        while (*p) {
            int  adv;
            int  cp = bpe_utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) {
                result += (char) it->second;
            }
            p += adv;
        }
        return result;
    }

    void init(const BpeTokenizer & bpe, int vsize) {
        vocab_size = vsize;
        bpe_ptr    = &bpe;

        for (int b = 0; b < 256; b++) {
            int adv;
            int cp       = bpe_utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t) b;
        }

        auto nl     = bpe_encode(bpe, "\n", false);
        newline_tok = nl.empty() ? -1 : nl[0];

        bpm_name      = bpe_encode(bpe, "bpm:", false);
        caption_name  = bpe_encode(bpe, "caption:", false);
        duration_name = bpe_encode(bpe, "duration:", false);
        keyscale_name = bpe_encode(bpe, "keyscale:", false);
        language_name = bpe_encode(bpe, "language:", false);
        timesig_name  = bpe_encode(bpe, "timesignature:", false);

        {
            std::vector<std::string> vals;
            for (int v = 30; v <= 300; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, bpm_tree, "bpm:", vals);
        }
        {
            std::vector<std::string> vals;
            for (int v = 10; v <= 600; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, duration_tree, "duration:", vals);
        }
        {
            const char * notes[] = { "A", "B", "C", "D", "E", "F", "G" };
            const char * accs[]  = { "", "#", "b", "\xe2\x99\xaf", "\xe2\x99\xad" };
            const char * modes[] = { "major", "minor" };
            std::vector<std::string> vals;
            for (auto n : notes)
                for (auto a : accs)
                    for (auto md : modes) vals.push_back(std::string(n) + a + " " + md);
            build_value_tree(bpe, keyscale_tree, "keyscale:", vals);
        }
        {
            std::vector<std::string> vals = {
                "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",  "es", "fa",      "fi",
                "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja",  "ko", "la",      "lt",
                "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa",  "sk", "sr",      "sv",
                "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
            };
            build_value_tree(bpe, language_tree, "language:", vals);
        }
        {
            std::vector<std::string> vals = { "2", "3", "4", "6" };
            build_value_tree(bpe, timesig_tree, "timesignature:", vals);
        }

        fprintf(stderr, "[FSM] Prefix trees: bpm=%zu, dur=%zu, key=%zu, lang=%zu, tsig=%zu nodes\n",
                bpm_tree.nodes.size(), duration_tree.nodes.size(), keyscale_tree.nodes.size(),
                language_tree.nodes.size(), timesig_tree.nodes.size());
        enabled  = true;
        state    = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    void reset() {
        state                   = BPM_NAME;
        name_pos                = 0;
        caption_pending_newline = false;
        caption_ending          = false;
        pending_field_name.clear();
        value_acc.clear();
        inject_queue.clear();
    }

    void force_field(const BpeTokenizer & bpe, State value_state, const std::string & val) {
        switch (value_state) {
            case BPM_VALUE:
                forced_bpm = val;
                bpm_tree   = PrefixTree();
                build_value_tree(bpe, bpm_tree, "bpm:", { val });
                break;
            case DURATION_VALUE:
                forced_duration = val;
                duration_tree   = PrefixTree();
                build_value_tree(bpe, duration_tree, "duration:", { val });
                break;
            case KEYSCALE_VALUE:
                forced_keyscale = val;
                keyscale_tree   = PrefixTree();
                build_value_tree(bpe, keyscale_tree, "keyscale:", { val });
                break;
            case LANGUAGE_VALUE:
                forced_language = val;
                language_tree   = PrefixTree();
                build_value_tree(bpe, language_tree, "language:", { val });
                break;
            case TIMESIG_VALUE:
                forced_timesig = val;
                timesig_tree   = PrefixTree();
                build_value_tree(bpe, timesig_tree, "timesignature:", { val });
                break;
            default:
                break;
        }
    }

    const std::vector<int> * current_name_tokens() const {
        switch (state) {
            case BPM_NAME:      return &bpm_name;
            case CAPTION_NAME:  return &caption_name;
            case DURATION_NAME: return &duration_name;
            case KEYSCALE_NAME: return &keyscale_name;
            case LANGUAGE_NAME: return &language_name;
            case TIMESIG_NAME:  return &timesig_name;
            default:            return nullptr;
        }
    }

    const PrefixTree * current_value_tree() const {
        switch (state) {
            case BPM_VALUE:      return &bpm_tree;
            case DURATION_VALUE: return &duration_tree;
            case KEYSCALE_VALUE: return &keyscale_tree;
            case LANGUAGE_VALUE: return &language_tree;
            case TIMESIG_VALUE:  return &timesig_tree;
            default:             return nullptr;
        }
    }

    State next_name_state() const {
        switch (state) {
            case BPM_NAME:
            case BPM_VALUE:      return skip_caption ? DURATION_NAME : CAPTION_NAME;
            case CAPTION_NAME:
            case CAPTION_VALUE:  return DURATION_NAME;
            case DURATION_NAME:
            case DURATION_VALUE: return KEYSCALE_NAME;
            case KEYSCALE_NAME:
            case KEYSCALE_VALUE: return LANGUAGE_NAME;
            case LANGUAGE_NAME:
            case LANGUAGE_VALUE: return TIMESIG_NAME;
            case TIMESIG_NAME:
            case TIMESIG_VALUE:  return THINK_END;
            default:             return CODES;
        }
    }

    void apply_mask(float * logits) {
        if (!enabled || state == CODES || state == DISABLED) {
            return;
        }

        if (!inject_queue.empty()) {
            int forced = inject_queue[0];
            for (int v = 0; v < vocab_size; v++)
                if (v != forced) logits[v] = -1e9f;
            return;
        }

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int) name->size()) {
            int forced = (*name)[name_pos];
            for (int v = 0; v < vocab_size; v++)
                if (v != forced) logits[v] = -1e9f;
            return;
        }

        const char *        prefix = nullptr;
        const std::string * forced = nullptr;
        switch (state) {
            case BPM_VALUE:      prefix = "bpm:";           forced = &forced_bpm;      break;
            case DURATION_VALUE: prefix = "duration:";      forced = &forced_duration; break;
            case KEYSCALE_VALUE: prefix = "keyscale:";      forced = &forced_keyscale; break;
            case LANGUAGE_VALUE: prefix = "language:";      forced = &forced_language; break;
            case TIMESIG_VALUE:  prefix = "timesignature:"; forced = &forced_timesig;  break;
            default:             break;
        }
        if (prefix && forced && !forced->empty() && bpe_ptr) {
            std::string      text  = std::string(prefix) + *forced + "\n";
            std::vector<int> vtoks = tokenize_strip(*bpe_ptr, text, prefix);
            if (!vtoks.empty()) {
                inject_queue = vtoks;
                int ftok     = inject_queue[0];
                for (int v = 0; v < vocab_size; v++)
                    if (v != ftok) logits[v] = -1e9f;
                return;
            }
        }

        const PrefixTree * tree = current_value_tree();
        if (tree) {
            const std::vector<int> * allowed = tree->get(value_acc);
            if (allowed && !allowed->empty()) {
                std::vector<float> saved(allowed->size());
                for (size_t i = 0; i < allowed->size(); i++) saved[i] = logits[(*allowed)[i]];
                for (int v = 0; v < vocab_size; v++) logits[v] = -1e9f;
                for (size_t i = 0; i < allowed->size(); i++) logits[(*allowed)[i]] = saved[i];
            } else if (newline_tok >= 0) {
                for (int v = 0; v < vocab_size; v++)
                    if (v != newline_tok) logits[v] = -1e9f;
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + AUDIO_CODE_COUNT; v++)
                if (v < vocab_size) logits[v] = -1e9f;
            return;
        }

        if (state == THINK_END) {
            for (int v = 0; v < vocab_size; v++)
                if (v != think_end_tok) logits[v] = -1e9f;
            return;
        }
    }

    void update(int token) {
        if (!enabled || state == CODES || state == DISABLED) {
            return;
        }

        if (!inject_queue.empty()) {
            inject_queue.erase(inject_queue.begin());
            if (inject_queue.empty()) {
                state    = next_name_state();
                name_pos = 0;
                value_acc.clear();
            }
            return;
        }

        if (caption_pending_newline) {
            caption_pending_newline = false;
            std::string tok_text    = decode_token(token);
            if (!tok_text.empty() && tok_text[0] != ' ' && tok_text[0] != '\t') {
                caption_ending     = true;
                pending_field_name = tok_text;
                if (tok_text.find(':') != std::string::npos) {
                    std::string field = pending_field_name.substr(0, pending_field_name.find(':'));
                    while (!field.empty() && field.back() == ' ') field.pop_back();
                    State target = field_name_to_value_state(field);
                    if (target != DISABLED) {
                        state    = target;
                        name_pos = 0;
                        value_acc.clear();
                        caption_ending = false;
                        pending_field_name.clear();
                    }
                }
                return;
            }
            return;
        }

        if (caption_ending) {
            std::string tok_text = decode_token(token);
            pending_field_name += tok_text;
            if (tok_text.find(':') != std::string::npos || pending_field_name.find(':') != std::string::npos) {
                std::string field = pending_field_name.substr(0, pending_field_name.find(':'));
                while (!field.empty() && field.back() == ' ') field.pop_back();
                State target = field_name_to_value_state(field);
                if (target != DISABLED) {
                    state    = target;
                    name_pos = 0;
                    value_acc.clear();
                    caption_ending = false;
                    pending_field_name.clear();
                }
            }
            return;
        }

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int) name->size()) {
            name_pos++;
            if (name_pos >= (int) name->size()) {
                switch (state) {
                    case BPM_NAME:      state = BPM_VALUE;      break;
                    case CAPTION_NAME:  state = CAPTION_VALUE;  break;
                    case DURATION_NAME: state = DURATION_VALUE; break;
                    case KEYSCALE_NAME: state = KEYSCALE_VALUE; break;
                    case LANGUAGE_NAME: state = LANGUAGE_VALUE; break;
                    case TIMESIG_NAME:  state = TIMESIG_VALUE;  break;
                    default:            break;
                }
                value_acc.clear();
            }
            return;
        }

        if (current_value_tree()) {
            if (token == newline_tok) {
                state    = next_name_state();
                name_pos = 0;
                value_acc.clear();
            } else {
                value_acc.push_back(token);
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            if (token == newline_tok) caption_pending_newline = true;
            return;
        }

        if (state == THINK_END) {
            state = CODES;
            return;
        }
    }

    State field_name_to_value_state(const std::string & field) const {
        if (field == "duration")      return DURATION_VALUE;
        if (field == "keyscale")      return KEYSCALE_VALUE;
        if (field == "language")      return LANGUAGE_VALUE;
        if (field == "timesignature") return TIMESIG_VALUE;
        return DISABLED;
    }
};

} // namespace tts_cpp::acestep
