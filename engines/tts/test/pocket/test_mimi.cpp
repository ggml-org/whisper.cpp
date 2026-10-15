#include "pocket/mimi.h"
#include "pocket/backend.h"
#include "npy.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using tts_cpp::pocket::detail::Mimi;
using Clock = std::chrono::steady_clock;
static std::vector<float> load(const std::string & path) {
    auto a = npy_load(path);
    if (a.dtype != "<f4") throw std::runtime_error("expected float32 fixture");
    std::vector<float> result(a.n_elements());
    std::memcpy(result.data(), a.data.data(), result.size()*sizeof(float));
    return result;
}
static void compare(const char * name, const std::vector<float> & got, const std::vector<float> & ref, double tolerance) {
    if (got.size() != ref.size()) throw std::runtime_error(std::string(name)+": size mismatch");
    double worst = 0, error = 0, energy = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) throw std::runtime_error("non-finite audio");
        double d = double(got[i])-ref[i]; worst = std::max(worst, std::abs(d));
        error += d*d; energy += double(ref[i])*ref[i];
    }
    std::printf("%s max_abs=%.8g snr_db=%.3f\n", name, worst, 10*std::log10(energy/std::max(error, 1e-30)));
    if (worst > tolerance) throw std::runtime_error(std::string(name)+": parity failure");
}
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) { std::fprintf(stderr, "usage: test-pocket-mimi model.gguf reference-directory [cpu|opencl]\n"); return 2; }
    try {
        auto backend = tts_cpp::pocket::detail::select_backend(false, argc == 4 ? argv[3] : "cpu");
        Mimi model(argv[1], 1, ggml_backend_get_device(backend.get()));
        std::printf("Mimi backend: %s\n", model.backend_name());
        const std::string dir = argv[2];
        const auto latents = load(dir+"/latents.npy"), pcm = load(dir+"/pcm.npy");
        std::vector<float> singles;
        const auto start = Clock::now();
        for (size_t i = 0; i < latents.size(); i += model.latent_dim()) {
            const auto out = model.decode({latents.begin()+i, latents.begin()+i+model.latent_dim()});
            singles.insert(singles.end(), out.begin(), out.end());
        }
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        std::printf("native codec seconds=%.6f audio_seconds=%.3f RTF=%.6f\n", seconds,
                    double(singles.size())/model.sample_rate(), seconds*model.sample_rate()/singles.size());
        compare("native decoder vs PyTorch", singles, pcm, 0.001);
        model.reset_decoder();
        std::vector<float> chunks;
        size_t begin = 0, iteration = 0;
        // Cross the 250-position attention window and every convolution seam.
        const size_t counts[] = {1, 7, 16, 2, 3};
        while (begin < latents.size()) {
            const auto end = std::min(latents.size(), begin+counts[iteration++%5]*model.latent_dim());
            const auto out = model.decode({latents.begin()+begin, latents.begin()+end});
            chunks.insert(chunks.end(), out.begin(), out.end()); begin = end;
        }
        compare("variable chunks vs single frames", chunks, singles, 0.001);
        const auto expected_encoded = load(dir+"/encoded.npy");
        if (std::any_of(expected_encoded.begin(), expected_encoded.end(), [](float x) { return x != 0; })) {
            auto encoded = model.encode(pcm);
            compare("native encoder vs PyTorch", encoded, expected_encoded, 0.003);
            auto again = model.encode(pcm);
            compare("encoder reset", again, encoded, 0.000001);
        } else {
            bool rejected = false;
            try { model.encode(pcm); } catch (const std::exception & e) {
                rejected = std::string(e.what()).find("disabled voice encoder") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("disabled voice encoder was accepted");
            std::puts("Disabled voice encoder rejected (non-zero encoder parity needs the random fixture)");
        }
        bool rejected = false;
        try { model.decode(std::vector<float>(32, std::numeric_limits<float>::quiet_NaN())); }
        catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("accepted NaN latents");
        model.reset_decoder();
        compare("decoder reset", model.decode({latents.begin(), latents.begin()+32}),
                {singles.begin(), singles.begin()+1920}, 0.000001);
        // Input validation must preserve a live stream; execution failure
        // must reset it. Neither an encoder call nor its rejection may alter it.
        auto reject_decode = [&](const std::vector<float> & bad) {
            bool failed = false;
            try { model.decode(bad); } catch (const std::exception &) { failed = true; }
            if (!failed) throw std::runtime_error("accepted invalid decoder input");
        };
        reject_decode({});
        reject_decode(std::vector<float>(31));
        reject_decode(std::vector<float>(32*17));
        reject_decode(std::vector<float>(32, std::numeric_limits<float>::quiet_NaN()));
        try { model.encode(std::vector<float>(1921, 0.1f)); } catch (const std::exception & e) {
            if (std::string(e.what()).find("disabled voice encoder") == std::string::npos) throw;
        }
        compare("encoder and rejection preserve decoder", model.decode({latents.begin()+32, latents.begin()+64}),
                {singles.begin()+1920, singles.begin()+3840}, 0.000001);
        reject_decode(std::vector<float>(32, std::numeric_limits<float>::max()));
        compare("execution failure resets decoder", model.decode({latents.begin(), latents.begin()+32}),
                {singles.begin(), singles.begin()+1920}, 0.000001);
        drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, 24000, 32};
        drwav wav;
        if (!drwav_init_file_write(&wav, (dir+"/pocket-native-decoder.wav").c_str(), &format, nullptr))
            throw std::runtime_error("cannot open waveform output");
        auto written = drwav_write_pcm_frames(&wav, singles.size(), singles.data()); drwav_uninit(&wav);
        if (written != singles.size()) throw std::runtime_error("incomplete waveform write");
        std::puts("Pocket Mimi parity, streaming and reset checks passed");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
