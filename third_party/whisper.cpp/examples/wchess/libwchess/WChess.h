#pragma once
#include "whisper.h"
#include <string>
#include <vector>
#include <memory>

class Chessboard;

class WChess {
public:
    using CheckRunningCb = bool (*)();
    using GetAudioCb = bool (*)(std::vector<float> &);
    using SetMovesCb = void (*)(const std::string &, float);
    using SetGrammarCb = void (*)(const std::string &);
    using ClearAudioCb = void (*)();

    struct callbacks {
        GetAudioCb get_audio = nullptr;
        SetMovesCb set_move = nullptr;
        SetGrammarCb set_grammar = nullptr;
    };

    struct settings {
        int32_t vad_ms     = 2000;
        int32_t prompt_ms  = 5000;
        int32_t command_ms = 4000;
        float vad_thold    = 0.2f;
        float freq_thold   = 100.0f;
        bool print_energy  = false;
    };

    WChess(
        whisper_context * ctx,
        const whisper_full_params & wparams,
        callbacks cb,
        settings s
    );
    ~WChess();

    void run();

    std::string stringify_board() const;

    std::string get_grammar() const;

private:
    bool get_audio(std::vector<float>& pcmf32) const;
    void set_move(const std::string& moves, float prob) const;
    void set_grammar(const std::string& grammar) const;

    std::string transcribe(
                    const std::vector<float> & pcmf32,
                    float & logprob_min,
                    float & logprob_sum,
                    int & n_tokens,
                    int64_t & t_ms);

    whisper_context * m_ctx;
    whisper_full_params m_wparams;
    const callbacks m_cb;
    const settings m_settings;
    std::unique_ptr<Chessboard> m_board;
};
