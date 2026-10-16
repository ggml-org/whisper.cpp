#pragma once

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

struct cosyvoice_coreml_input {
    const char * name;
    const float * data;
    std::vector<int64_t> shape;
    bool finite_already_checked = false;
};

struct cosyvoice_coreml_output {
    const char * name;
    float * data;
    std::vector<int64_t> shape;
};

class cosyvoice_coreml {
public:
    static bool has_sidecar(const std::string & gguf, const std::string & suffix = "");
    static std::shared_ptr<cosyvoice_coreml> load(const std::string & gguf, const std::string & suffix = "");
    std::vector<int64_t> dimensions(const char * name) const;
    bool matches(const char * name, const std::vector<int64_t> & shape) const;
    bool stateful() const;
    void reset();
    void clear_predictions() { predictions_.store(0); }
    void predict(std::initializer_list<cosyvoice_coreml_input> inputs,
                 std::initializer_list<cosyvoice_coreml_output> outputs);
    const std::string & label() const { return label_; }
    uint64_t predictions() const { return predictions_.load(); }

private:
    std::shared_ptr<void> model_;
    std::string label_;
    std::atomic<uint64_t> predictions_{0};
};

bool cosyvoice_coreml_frontend(const std::shared_ptr<cosyvoice_coreml> & model,
                               const std::vector<int32_t> & tokens, const std::vector<float> & speaker,
                               std::vector<float> & mu, std::vector<float> & spks);
bool cosyvoice_coreml_euler(const std::shared_ptr<cosyvoice_coreml> & model, int head_dim,
                           const std::vector<float> & mu, const std::vector<float> & cond,
                           const std::vector<float> & spks, int frames, std::vector<float> & x);
bool cosyvoice_coreml_f0(const std::shared_ptr<cosyvoice_coreml> & model,
                        const std::vector<float> & mel, int frames, std::vector<float> & f0);
bool cosyvoice_coreml_spectrum(const std::shared_ptr<cosyvoice_coreml> & model,
                              const std::vector<float> & mel, int frames,
                              const std::vector<float> & source, int source_frames,
                              std::vector<float> & spectrum);
