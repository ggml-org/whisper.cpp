#include "WChess.h"
#include "Chessboard.h"
#include "grammar-parser.h"
#include "common.h"
#include <chrono>

WChess::WChess(whisper_context * ctx,
        const whisper_full_params & wparams,
        callbacks cb,
        settings s)
        : m_ctx(ctx)
        , m_wparams(wparams)
        , m_cb(cb)
        , m_settings(s)
        , m_board(new Chessboard())
{}

WChess::~WChess() = default;

void WChess::set_move(const std::string& moves, float prob) const {
    if (m_cb.set_move) (*m_cb.set_move)(moves, prob);
}

void WChess::set_grammar(const std::string& grammar) const {
    if (m_cb.set_grammar) (*m_cb.set_grammar)(grammar);
}

bool WChess::get_audio(std::vector<float>& pcmf32) const {
    if (m_cb.get_audio) return (*m_cb.get_audio)(pcmf32);
    return false;
}

std::string WChess::stringify_board() const {
    return m_board->stringifyBoard();
}

std::string WChess::get_grammar() const {
    return m_board->grammar();
}

void WChess::run() {
    bool have_prompt  = true;
    bool ask_prompt   = !have_prompt;

    float logprob_min  = 0.0f;

    float logprob_sum  = 0.0f;

    int n_tokens  = 0;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    const std::string k_prompt = have_prompt ? "" : "rook to d4, f3";
    int64_t t_ms = 0;

    if (ask_prompt) {
        fprintf(stdout, "\n");
        fprintf(stdout, "%s: Say the following phrase: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
        fprintf(stdout, "\n");

        ask_prompt = false;
    }

    while (get_audio(pcmf32_cur)) {
        if (!pcmf32_cur.empty()) {
            // fprintf(stdout, "%s: Processing ...\n", __func__);

            if (!have_prompt) {
                const auto txt = ::trim(transcribe(pcmf32_cur, logprob_min, logprob_sum, n_tokens, t_ms));

                fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int) t_ms);

                const float sim = similarity(txt, k_prompt);

                if (txt.length() < 0.8*k_prompt.length() || txt.length() > 1.2*k_prompt.length() || sim < 0.8f) {
                    fprintf(stdout, "%s: WARNING: prompt not recognized, try again\n", __func__);
                    ask_prompt = true;
                } else {
                    fprintf(stdout, "\n");
                    fprintf(stdout, "%s: The prompt has been recognized!\n", __func__);
                    fprintf(stdout, "%s: Waiting for voice commands ...\n", __func__);
                    fprintf(stdout, "\n");

                    // save the audio for the prompt
                    pcmf32_prompt = pcmf32_cur;
                    have_prompt = true;
                    m_board->setPrompt(k_prompt);
                }
            } else {
                if (!pcmf32_prompt.empty()) pcmf32_cur.insert(pcmf32_cur.begin(), pcmf32_prompt.begin(), pcmf32_prompt.end());
                constexpr size_t MIN_SIZE = 1.2 * WHISPER_SAMPLE_RATE;
                if (MIN_SIZE > pcmf32_cur.size()) pcmf32_cur.insert(pcmf32_cur.begin(), MIN_SIZE - pcmf32_cur.size(), 0.0f);

                // fprintf(stdout, "%s: grammar rules:\n'%s'\n", __func__, m_board->grammar().c_str());

                auto grammar_parsed = grammar_parser::parse(m_board->grammar().c_str());
                auto grammar_rules  = grammar_parsed.c_rules();

                m_wparams.grammar_rules   = grammar_rules.data();
                m_wparams.n_grammar_rules = grammar_rules.size();

                m_wparams.i_start_rule    = grammar_parsed.symbol_ids.at("move");
                auto txt = ::trim(transcribe(pcmf32_cur, logprob_min, logprob_sum, n_tokens, t_ms));

                const float p = 100.0f * std::exp(logprob_min);

                fprintf(stdout, "%s: heard '%s'\n", __func__, txt.c_str());

                // find the prompt in the text
                float best_sim = 0.0f;
                size_t best_len = 0;
                for (int n = 0.8*k_prompt.size(); n <= 1.2*k_prompt.size(); ++n) {
                    const auto prompt = txt.substr(0, n);

                    const float sim = similarity(prompt, k_prompt);

                    //fprintf(stderr, "%s: prompt = '%s', sim = %f\n", __func__, prompt.c_str(), sim);

                    if (sim > best_sim) {
                        best_sim = sim;
                        best_len = n;
                    }
                }

                fprintf(stdout, "%s:   DEBUG: txt = '%s', prob = %.2f%%\n", __func__, txt.c_str(), p);
                std::string command = ::trim(txt.substr(best_len));

                fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", command.c_str(), "\033[0m", (int) t_ms);
                fprintf(stdout, "\n");

                if (!command.empty()) {
                    set_move(m_board->process(command), p);
                    set_grammar(m_board->grammar());
                }
                if (m_board->grammar().empty()) {
                    fprintf(stdout, "%s: No more moves possible\n", __func__);
                    break;
                }
            }
        }

        if (ask_prompt) {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s: Say the following phrase: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
            fprintf(stdout, "\n");

            ask_prompt = false;
        }
    }
}

std::string WChess::transcribe(
                const std::vector<float> & pcmf32,
                float & logprob_min,
                float & logprob_sum,
                int & n_tokens,
                int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    logprob_min = 0.0f;
    logprob_sum = 0.0f;
    n_tokens    = 0;
    t_ms = 0;

    if (whisper_full(m_ctx, m_wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return {};
    }

    std::string result;

    const int n_segments = whisper_full_n_segments(m_ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(m_ctx, i);

        result += text;

        const int n = whisper_full_n_tokens(m_ctx, i);
        for (int j = 0; j < n; ++j) {
            const auto token = whisper_full_get_token_data(m_ctx, i, j);

            if(token.plog > 0.0f) return {};
            logprob_min = std::min(logprob_min, token.plog);
            logprob_sum += token.plog;
            ++n_tokens;
        }
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}
