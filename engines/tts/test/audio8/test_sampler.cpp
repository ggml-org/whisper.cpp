// Audio8 sampler parity: the candidate filter the reference runs before every
// draw, checked against the score vectors it dumped.
//
// The draw itself cannot be compared across runtimes -- torch's Gumbel noise is
// its own generator -- but everything feeding it can, and the filter is where
// the reference departs from the usual order: top-k and top-p score the raw
// logits, and the temperature only scales what survives. Getting that backwards
// still produces speech, just from a different nucleus, so the fixture is the
// only thing that catches it.
//
// The fixture also states the settings it was dumped with, so the test reads
// those rather than restating them.

#include "audio8/sampling.h"
#include "json.hpp"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

// Both sides are one multiply away from the same logit, so anything above a
// float rounding step is a real disagreement.
constexpr double SCORE_TOLERANCE = 1e-4;

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/" + name + ".npy");
    }

    nlohmann::json meta() const {
        std::ifstream file(dir + "/meta.json");
        if (!file) throw std::runtime_error("cannot open " + dir + "/meta.json");
        return nlohmann::json::parse(file);
    }
};

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

std::vector<float> row_of(const npy_array & array, size_t row, size_t width) {
    const float * source = as_f32(array) + row * width;
    return std::vector<float>(source, source + width);
}

sampling_params settings_of(const fixture & data) {
    const nlohmann::json sampling = data.meta().at("sampling");
    sampling_params params;
    params.temperature = sampling.at("temperature").get<float>();
    params.top_k = sampling.at("top_k").get<int>();
    params.top_p = sampling.at("top_p").get<float>();
    return params;
}

// A rejected candidate is -inf on both sides; a kept one has to agree
// numerically. Counting the two ways they can disagree names the bug, since a
// too-wide nucleus and a too-narrow one come from opposite mistakes.
struct disagreement {
    size_t only_engine = 0;
    size_t only_reference = 0;
    double worst = 0.0;

    bool clean() const {
        return only_engine == 0 && only_reference == 0 && worst <= SCORE_TOLERANCE;
    }
};

void compare_row(const std::vector<float> & got, const float * want,
                 disagreement & found) {
    for (size_t index = 0; index < got.size(); ++index) {
        const bool engine_rejected = !std::isfinite(got[index]);
        const bool reference_rejected = !std::isfinite(want[index]);
        if (engine_rejected && reference_rejected) continue;
        if (engine_rejected) {
            ++found.only_engine;
            continue;
        }
        if (reference_rejected) {
            ++found.only_reference;
            continue;
        }
        const double delta = std::fabs(static_cast<double>(got[index]) - want[index]);
        if (delta > found.worst) found.worst = delta;
    }
}

disagreement compare_all(const fixture & data, const sampling_params & params) {
    const npy_array logits = data.load("sem_logits");
    const npy_array filtered = data.load("filtered");
    if (logits.shape != filtered.shape) {
        throw std::runtime_error("the logits and the filtered scores differ in shape");
    }
    const size_t steps = static_cast<size_t>(logits.shape[0]);
    const size_t width = static_cast<size_t>(logits.shape[1]);

    disagreement found;
    for (size_t step = 0; step < steps; ++step) {
        compare_row(filter_scores(row_of(logits, step, width), params),
                    as_f32(filtered) + step * width, found);
    }
    std::fprintf(stderr,
                 "  [filter] steps=%zu width=%zu  max|delta|=%.3e  rejected only "
                 "here=%zu  only upstream=%zu\n",
                 steps, width, found.worst, found.only_engine, found.only_reference);
    return found;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <ref-dir>\n", argv[0]);
        return 1;
    }
    const fixture data{argv[1]};
    try {
        const disagreement found = compare_all(data, settings_of(data));
        std::printf("\n%s\n", found.clean() ? "PASS" : "FAIL");
        return found.clean() ? 0 : 1;
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "sampler: %s\n", failure.what());
        return 1;
    }
}
