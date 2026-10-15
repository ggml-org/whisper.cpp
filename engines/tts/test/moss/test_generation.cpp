#include "moss/frontend.h"
#include "moss/generation.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

DelayConfig test_config() {
    DelayConfig config;
    config.n_vq = 3;
    config.text_vocab = 200;
    config.audio_vocab = 11;
    config.audio_pad_code = 10;
    config.audio_start_token_id = 100;
    config.audio_end_token_id = 101;
    config.audio_user_slot_token_id = 102;
    config.audio_assistant_gen_slot_token_id = 103;
    config.audio_assistant_delay_slot_token_id = 104;
    return config;
}

constexpr int32_t PAD_TOKEN = 0;
constexpr int32_t IM_END_TOKEN = 1;

DelayLogits uniform_logits(const DelayConfig & config) {
    DelayLogits logits;
    logits.text.assign((size_t) config.text_vocab, 0.0f);
    logits.audio.assign((size_t) config.n_vq,
            std::vector<float>((size_t) config.audio_vocab, 0.0f));
    return logits;
}

DelayLogits logits_preferring(const DelayConfig & config, int32_t text_token, int32_t audio_code) {
    DelayLogits logits = uniform_logits(config);
    logits.text[(size_t) text_token] = 50.0f;
    for (auto & channel : logits.audio) {
        channel[(size_t) audio_code] = 50.0f;
    }
    return logits;
}

SamplingConfig greedy_sampling() {
    SamplingConfig sampling;
    sampling.text_temperature = 0.0f;
    sampling.audio_temperature = 0.0f;
    return sampling;
}

std::vector<DelayRow> seeded_prompt(const DelayConfig & config) {
    std::vector<DelayRow> rows(2);
    rows[0].text = 42;
    rows[0].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    rows[1].text = config.audio_start_token_id;
    rows[1].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    return rows;
}

void test_delay_round_trip() {
    const int n_vq = 3;
    const int pad = 10;
    const std::vector<int32_t> codes = {0, 1, 2, 3, 4, 5};
    const std::vector<int32_t> delayed = apply_delay_pattern(codes, 2, n_vq, pad);
    check(delayed.size() == 4 * 3, "delayed frame count");
    const std::vector<int32_t> expected = {
        0, pad, pad,
        3, 1, pad,
        pad, 4, 2,
        pad, pad, 5,
    };
    check(delayed == expected, "delay pattern layout");
    check(apply_de_delay_pattern(delayed, 4, n_vq, pad) == codes, "de-delay round trip");
    check(apply_de_delay_pattern({pad, pad, pad}, 1, n_vq, pad).empty(), "de-delay short input");
}

void test_repetition_penalty() {
    std::vector<float> logits = {2.0f, -2.0f, 1.0f};
    const std::vector<int32_t> history = {0, 1, 1};
    apply_repetition_penalty(logits, history, 2.0f);
    check(logits[0] == 1.0f, "positive logit divided");
    check(logits[1] == -4.0f, "negative logit multiplied");
    check(logits[2] == 1.0f, "unseen logit untouched");
}

void test_sampling_determinism() {
    std::vector<float> logits = {0.1f, 5.0f, 0.2f, 4.9f, 0.3f};
    std::mt19937 rng_a(123);
    std::mt19937 rng_b(123);
    const int32_t a = sample_row(logits, 0.8f, 2, true, rng_a);
    const int32_t b = sample_row(logits, 0.8f, 2, true, rng_b);
    check(a == b, "top-k sampling is deterministic under one seed");
    check(a == 1 || a == 3, "top-k keeps only the two best tokens");
    std::mt19937 rng_c(7);
    check(sample_row({0.0f, 100.0f, 0.0f}, 1.0f, 0, true, rng_c) == 1,
            "dominant logit wins the multinomial");
}

void test_argmax_path() {
    std::mt19937 rng(1);
    check(sample_row({0.5f, 3.0f, 1.0f}, 1.0f, 50, false, rng) == 1, "argmax when not sampling");
}

void test_top_p_zero_is_greedy() {
    std::mt19937 rng_a(1);
    std::mt19937 rng_b(999);
    const std::vector<float> logits = {0.1f, 5.0f, 0.2f, 4.9f, 0.3f};
    check(sample_row(logits, 0.0f, 3, true, rng_a) == 1, "top_p 0 keeps only the best token");
    check(sample_row(logits, 0.0f, 3, true, rng_b) == 1, "top_p 0 is seed independent");
}

std::vector<DelayRow> penalty_prompt(const DelayConfig & config) {
    std::vector<DelayRow> rows(4);
    rows[0].text = 42;
    rows[0].audio = {5, 7, 7};
    rows[1].text = config.audio_start_token_id;
    rows[1].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    rows[2].text = config.audio_assistant_gen_slot_token_id;
    rows[2].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    rows[3].text = config.audio_assistant_gen_slot_token_id;
    rows[3].audio.assign((size_t) config.n_vq, config.audio_pad_code);
    return rows;
}

DelayLogits penalty_logits(const DelayConfig & config) {
    DelayLogits logits = uniform_logits(config);
    logits.text[(size_t) config.audio_assistant_gen_slot_token_id] = 50.0f;
    for (auto & channel : logits.audio) {
        channel[5] = 10.0f;
        channel[7] = 9.9f;
    }
    return logits;
}

void test_repetition_penalty_scoping() {
    const DelayConfig config = test_config();
    SamplingConfig sampling = greedy_sampling();
    sampling.audio_repetition_penalty = 2.0f;
    std::mt19937 rng(1);
    DelayState state(config, penalty_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    const DelayRow row = state.step(penalty_logits(config), sampling, rng);
    check(row.audio[0] == 7, "channel 0 is penalized only by its own history");
    check(row.audio[1] == 5, "rest channels are not penalized by channel 0 codes");
    check(row.audio[2] == 5, "rest channels share the rest-scoped history");
}

void test_state_machine_drain() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(1234);
    DelayState state(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);

    DelayRow row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 4),
            sampling, rng);
    check(row.text == config.audio_assistant_gen_slot_token_id, "step 0 keeps generating");
    check(row.audio[0] == 4, "channel 0 opens on step 0");
    check(row.audio[1] == config.audio_pad_code, "channel 1 still closed on step 0");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 5),
            sampling, rng);
    check(row.audio[0] == 5 && row.audio[1] == 5, "channels ramp in one per step");
    check(row.audio[2] == config.audio_pad_code, "channel 2 closed until step 2");

    row = state.step(logits_preferring(config, config.audio_assistant_delay_slot_token_id, 6),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "model arms the drain");
    check(row.audio[0] == 6 && row.audio[1] == 6 && row.audio[2] == 6, "all channels open");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 7),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "drain forces delay slots");
    check(row.audio[0] == config.audio_pad_code, "channel 0 drains first");
    check(row.audio[1] == 7 && row.audio[2] == 7, "later channels still open while draining");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 8),
            sampling, rng);
    check(row.text == config.audio_assistant_delay_slot_token_id, "second drain step");
    check(row.audio[1] == config.audio_pad_code && row.audio[2] == 8, "drain advances by channel");

    row = state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, 9),
            sampling, rng);
    check(row.text == config.audio_end_token_id, "audio end is forced after n_vq drain steps");

    row = state.step(logits_preferring(config, IM_END_TOKEN, 9), sampling, rng);
    check(row.text == IM_END_TOKEN, "im_end is reachable after the drain");
    check(state.stopping(), "im_end stops the generation");

    const AudioSegments segments = state.generated_audio(2);
    check(segments.size() == 1, "one utterance yields one segment");
    check(!segments.empty() && segments[0].size() % config.n_vq == 0,
            "generated audio is frame aligned");
    check(!segments.empty() && std::none_of(segments[0].begin(), segments[0].end(),
            [&](int32_t code) { return code == config.audio_pad_code; }),
            "no pad codes leak into decoded segments");
}

constexpr uint32_t STATE_SEED = 1234;
constexpr uint32_t LOGIT_SOURCE_SEED = 77;
constexpr uint32_t SAMPLER_SEED = 5;
constexpr float RANDOM_LOGIT_SPREAD = 3.0f;
constexpr int OPEN_CODE = 4;
constexpr int ARMING_CODE = 5;
constexpr int DRAIN_CODE = 6;
constexpr int SLOT_SAMPLING_STEPS = 24;

void step_through_drain(DelayState & state, const DelayConfig & config, const SamplingConfig & sampling,
                        std::mt19937 & rng) {
    for (int drain = 0; drain < config.n_vq; ++drain) {
        check(state.text_logits() == TextLogits::None, "the drain forces its text tokens");
        state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, DRAIN_CODE), sampling, rng);
    }
}

void test_text_logits_follow_the_state() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(STATE_SEED);
    DelayState state(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    check(state.text_logits() == TextLogits::Slots, "an audio prompt samples only the slot tokens");

    state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id, OPEN_CODE), sampling, rng);
    check(state.text_logits() == TextLogits::Slots, "generation keeps sampling the slot tokens");

    state.step(logits_preferring(config, config.audio_assistant_delay_slot_token_id, ARMING_CODE), sampling, rng);
    step_through_drain(state, config, sampling, rng);
    check(state.text_logits() == TextLogits::Full, "after audio_end the whole vocabulary is sampled");

    state.step(logits_preferring(config, IM_END_TOKEN, DRAIN_CODE), sampling, rng);
    check(state.stopping() && state.text_logits() == TextLogits::None, "a stopped state reads no logits");
}

void fill_random(std::vector<float> & values, std::mt19937 & rng) {
    std::normal_distribution<float> logit(0.0f, RANDOM_LOGIT_SPREAD);
    for (float & value : values) value = logit(rng);
}

DelayLogits random_logits(const DelayConfig & config, std::mt19937 & rng) {
    DelayLogits logits = uniform_logits(config);
    fill_random(logits.text, rng);
    for (auto & channel : logits.audio) {
        fill_random(channel, rng);
    }
    return logits;
}

DelayLogits slots_only(const DelayConfig & config, DelayLogits logits) {
    std::vector<float> text(logits.text.size(), -std::numeric_limits<float>::infinity());
    for (int32_t id : {config.audio_assistant_gen_slot_token_id, config.audio_assistant_delay_slot_token_id}) {
        text[(size_t) id] = logits.text[(size_t) id];
    }
    logits.text = std::move(text);
    return logits;
}

void test_slot_logits_sample_like_the_whole_head() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling;
    std::mt19937 source(LOGIT_SOURCE_SEED), rng_full(SAMPLER_SEED), rng_slots(SAMPLER_SEED);
    DelayState full(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    DelayState slots(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    for (int i = 0; i < SLOT_SAMPLING_STEPS && !full.stopping(); ++i) {
        const DelayLogits logits = random_logits(config, source);
        const bool slot_step = full.text_logits() == TextLogits::Slots;
        const DelayRow expected = full.step(logits, sampling, rng_full);
        const DelayRow actual = slots.step(slot_step ? slots_only(config, logits) : logits, sampling, rng_slots);
        check(expected.text == actual.text && expected.audio == actual.audio,
                "slot-only logits sample the same rows as the whole head");
    }
}

void test_early_stop_masks() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(1);
    DelayState state(config, seeded_prompt(config), PAD_TOKEN, IM_END_TOKEN);
    const DelayRow row = state.step(logits_preferring(config, IM_END_TOKEN, 2), sampling, rng);
    check(row.text != IM_END_TOKEN, "im_end is masked during the ramp-in");
}

void test_incremental_de_delay() {
    const DelayConfig config = test_config();
    const SamplingConfig sampling = greedy_sampling();
    std::mt19937 rng(7);
    const std::vector<DelayRow> prompt = seeded_prompt(config);
    DelayState state(config, prompt, PAD_TOKEN, IM_END_TOKEN);
    check(state.available_frames((int) prompt.size()) == 0, "no frames before any step");
    for (int step = 0; step < 6; ++step) {
        state.step(logits_preferring(config, config.audio_assistant_gen_slot_token_id,
                step % 7), sampling, rng);
        const int expected = std::max(0, step + 1 - (config.n_vq - 1));
        check(state.available_frames((int) prompt.size()) == expected,
                "frame becomes available n_vq - 1 steps after its row");
    }
    const size_t begin = prompt.size() * (size_t) config.n_vq;
    const std::vector<int32_t> delayed(state.audio_history().begin() + begin,
            state.audio_history().end());
    const std::vector<int32_t> expected = apply_de_delay_pattern(delayed,
            (int) (delayed.size() / (size_t) config.n_vq), config.n_vq, config.audio_pad_code);
    const int frames = state.available_frames((int) prompt.size());
    check((size_t) frames * config.n_vq == expected.size(), "incremental frame count matches de-delay");
    for (int frame = 0; frame < frames; ++frame) {
        const std::vector<int32_t> codes = state.frame_codes((int) prompt.size(), frame);
        for (int channel = 0; channel < config.n_vq; ++channel) {
            check(codes[(size_t) channel] == expected[(size_t) frame * config.n_vq + channel],
                    "incremental frame matches the batch de-delay");
        }
    }
}

void test_segment_extraction() {
    const int n_vq = 2;
    const int pad = 9;
    const std::vector<int32_t> codes = {
        1, 2,
        pad, pad,
        3, 4,
        5, 6,
        pad, pad,
    };
    const AudioSegments segments = extract_audio_segments(codes, 5, n_vq, pad);
    check(segments.size() == 2, "pad frames separate independent segments");
    check(segments.size() == 2 && segments[0] == std::vector<int32_t>({1, 2}),
            "first segment keeps its own frames");
    check(segments.size() == 2 && segments[1] == std::vector<int32_t>({3, 4, 5, 6}),
            "second segment starts after the separator");
    check(extract_audio_segments({pad, pad}, 1, n_vq, pad).empty(), "all-pad input yields nothing");
    check(extract_audio_segments({1, 2}, 1, n_vq, pad).size() == 1,
            "a segment running to the end is kept");
}

void test_top_k_tie_break() {
    std::mt19937 rng(3);
    check(sample_row({1.0f, 5.0f, 5.0f, 0.0f}, 1.0f, 1, true, rng) == 1,
            "equal logits resolve to the lowest index");
    std::mt19937 rng_b(3);
    check(sample_row({5.0f, 1.0f, 5.0f, 5.0f}, 1.0f, 1, true, rng_b) == 0,
            "the tie break does not depend on position in the vocabulary scan");
}

void test_top_k_larger_than_vocab() {
    std::mt19937 rng(5);
    const int32_t token = sample_row({0.0f, 0.0f, 50.0f}, 1.0f, 100, true, rng);
    check(token == 2, "top_k beyond the vocabulary keeps every candidate");
}

std::vector<int32_t> fake_encode(const std::string & span) {
    std::vector<int32_t> ids;
    for (char c : span) {
        ids.push_back((int32_t) (unsigned char) c % 90);
    }
    return ids;
}

void test_duration_tokens_field() {
    const DelayConfig config = test_config();
    PromptTokens tokens;
    tokens.pad = PAD_TOKEN;
    tokens.im_start = 2;
    tokens.im_end = 3;
    std::string captured;
    const TextEncoder recorder = [&captured](const std::string & span) {
        captured += span;
        return std::vector<int32_t>{(int32_t) (span.size() % 90)};
    };
    captured.clear();
    build_prompt_rows(config, tokens, recorder, "hola", "es", 0, {});
    check(captured.find("- Tokens:\nNone\n") != std::string::npos,
            "free-length prompts carry Tokens: None");
    captured.clear();
    build_prompt_rows(config, tokens, recorder, "hola", "es", 38, {});
    check(captured.find("- Tokens:\n38\n") != std::string::npos,
            "duration_tokens lands as the Tokens field");
    check(captured.find("[pause") == std::string::npos,
            "the template adds no pause markers of its own");
    captured.clear();
    build_prompt_rows(config, tokens, recorder, "hola [pause 2.0s] mundo", "es", 0, {});
    check(captured.find("- Text:\nhola [pause 2.0s] mundo\n") != std::string::npos,
            "inline pause markers pass through the text field untouched");
}

void test_prompt_rows() {
    const DelayConfig config = test_config();
    PromptTokens tokens;
    tokens.pad = PAD_TOKEN;
    tokens.im_start = 2;
    tokens.im_end = 3;

    const std::vector<DelayRow> plain = build_prompt_rows(config, tokens, fake_encode,
            "hola", "es", 0, {});
    check(plain.size() > 4, "plain prompt has rows");
    check(plain.front().text == tokens.im_start, "prompt opens with im_start");
    check(plain.back().text == config.audio_start_token_id, "prompt ends with the audio seed row");
    for (const DelayRow & row : plain) {
        check((int) row.audio.size() == config.n_vq, "every row carries n_vq channels");
    }
    for (size_t i = 0; i + 1 < plain.size(); ++i) {
        for (int32_t code : plain[i].audio) {
            check(code == config.audio_pad_code, "plain prompt rows carry only pad codes");
        }
    }

    const int reference_frames = 4;
    std::vector<int32_t> reference_codes((size_t) reference_frames * config.n_vq);
    for (size_t i = 0; i < reference_codes.size(); ++i) {
        reference_codes[i] = (int32_t) (i % 7);
    }
    PromptAudio cloned_audio;
    cloned_audio.speaker_codes.push_back(reference_codes);
    const std::vector<DelayRow> cloned = build_prompt_rows(config, tokens, fake_encode,
            "hola", "es", 0, cloned_audio);
    int slot_rows = 0;
    int rows_with_codes = 0;
    for (const DelayRow & row : cloned) {
        if (row.text == config.audio_user_slot_token_id) {
            slot_rows++;
        }
        for (int32_t code : row.audio) {
            if (code != config.audio_pad_code) {
                rows_with_codes++;
                break;
            }
        }
    }
    check(slot_rows == reference_frames + config.n_vq - 1,
            "placeholder expands to frames + n_vq - 1 slots");
    check(rows_with_codes == reference_frames + config.n_vq - 1,
            "delayed reference codes land on the slot rows");
}

} // namespace

int main() {
    test_delay_round_trip();
    test_repetition_penalty();
    test_sampling_determinism();
    test_argmax_path();
    test_top_p_zero_is_greedy();
    test_repetition_penalty_scoping();
    test_state_machine_drain();
    test_text_logits_follow_the_state();
    test_slot_logits_sample_like_the_whole_head();
    test_early_stop_masks();
    test_incremental_de_delay();
    test_segment_extraction();
    test_top_k_tie_break();
    test_top_k_larger_than_vocab();
    test_duration_tokens_field();
    test_prompt_rows();
    if (failures == 0) {
        std::printf("moss generation conformance: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss generation conformance: %d failures\n", failures);
    return 1;
}
