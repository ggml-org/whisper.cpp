#pragma once

#include "coreml/parakeet-diarizer.h"
#include "parakeet_coreml_path.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace parakeet {

constexpr const char * kCoremlDiarizerUnitsEnv = "PARAKEET_COREML_COMPUTE_UNITS";
constexpr const char * kCoremlDiarizerDisableEnv = "PARAKEET_COREML_DIARIZER_DISABLE";
constexpr const char * kCoremlDisableEnv = "PARAKEET_COREML_DISABLE";

class CoremlDiarizers {
public:
    CoremlDiarizers(std::string gguf_path, std::vector<int> capacities);
    ~CoremlDiarizers();
    CoremlDiarizers(const CoremlDiarizers &) = delete;
    CoremlDiarizers & operator=(const CoremlDiarizers &) = delete;

    bool covers(int frames);
    bool prepare(int frames);
    int run(int valid_frames,
            const parakeet_coreml_rows * inputs, int n_inputs,
            const parakeet_coreml_rows_out * outputs, int n_outputs);
    std::string label();
    std::vector<int> capacities() const;

private:
    struct Slot {
        int capacity = 0;
        parakeet_coreml_diarizer * context = nullptr;
        bool failed = false;
    };

    Slot * slot_for(int frames);
    Slot * loaded_slot_for(int frames);
    bool load(Slot & slot);
    void quarantine(Slot & slot);

    std::string gguf_path_;
    std::vector<Slot> slots_;
    std::string label_;
    std::mutex mutex_;
};

bool coreml_diarizers_disabled();

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path);

std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path);

}
