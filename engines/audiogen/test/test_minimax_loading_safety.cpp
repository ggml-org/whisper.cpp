#include "minimax/mm3-lm-graph.h"
#include "audiogen-cpp/minimax/engine.h"
#include "minimax_tiny_model.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                                      \
    do {                                                                                      \
        ++checks;                                                                             \
        if (!(condition)) {                                                                   \
            ++failures;                                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition);          \
        }                                                                                     \
    } while (0)

bool contains_error(const std::vector<std::string> & errors, const std::string & expected) {
    for (const std::string & error : errors) {
        if (error.find(expected) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void test_typed_metadata_getters() {
    GGUFModel model = {};
    model.gguf = gguf_init_empty();

    const uint32_t array_value = 9;
    gguf_set_val_u32(model.gguf, "correct.u32", 42);
    gguf_set_val_f32(model.gguf, "correct.f32", 1.25f);
    gguf_set_val_str(model.gguf, "correct.string", "value");
    gguf_set_val_bool(model.gguf, "correct.bool", true);

    gguf_set_val_i32(model.gguf, "wrong-scalar.u32", 42);
    gguf_set_val_u32(model.gguf, "wrong-scalar.f32", 1);
    gguf_set_val_bool(model.gguf, "wrong-scalar.string", true);
    gguf_set_val_str(model.gguf, "wrong-scalar.bool", "true");

    gguf_set_arr_data(model.gguf, "wrong-array.u32", GGUF_TYPE_UINT32, &array_value, 1);
    gguf_set_arr_data(model.gguf, "wrong-array.f32", GGUF_TYPE_UINT32, &array_value, 1);
    gguf_set_arr_data(model.gguf, "wrong-array.string", GGUF_TYPE_UINT32, &array_value, 1);
    gguf_set_arr_data(model.gguf, "wrong-array.bool", GGUF_TYPE_UINT32, &array_value, 1);

    CHECK(gf_get_u32(model, "correct.u32") == 42);
    CHECK(gf_get_f32(model, "correct.f32") == 1.25f);
    CHECK(std::string(gf_get_str(model, "correct.string")) == "value");
    CHECK(gf_get_bool(model, "correct.bool"));

    CHECK(gf_get_u32(model, "missing") == 0);
    CHECK(gf_get_f32(model, "missing") == 0.0f);
    CHECK(std::string(gf_get_str(model, "missing")).empty());
    CHECK(!gf_get_bool(model, "missing"));

    CHECK(gf_get_u32(model, "wrong-scalar.u32") == 0);
    CHECK(gf_get_f32(model, "wrong-scalar.f32") == 0.0f);
    CHECK(std::string(gf_get_str(model, "wrong-scalar.string")).empty());
    CHECK(!gf_get_bool(model, "wrong-scalar.bool"));

    CHECK(gf_get_u32(model, "wrong-array.u32") == 0);
    CHECK(gf_get_f32(model, "wrong-array.f32") == 0.0f);
    CHECK(std::string(gf_get_str(model, "wrong-array.string")).empty());
    CHECK(!gf_get_bool(model, "wrong-array.bool"));

    gguf_free(model.gguf);
}

void test_tensor_range_validation() {
    const size_t maximum = std::numeric_limits<size_t>::max();

    CHECK(gf_tensor_range_fits(100, 20, 30, 50));
    CHECK(gf_tensor_range_fits(100, 100, 0, 0));
    CHECK(!gf_tensor_range_fits(100, 101, 0, 0));
    CHECK(!gf_tensor_range_fits(100, 20, 81, 0));
    CHECK(!gf_tensor_range_fits(100, 20, 80, 1));
    CHECK(!gf_tensor_range_fits(maximum, maximum - 4, 10, 0));
    CHECK(!gf_tensor_range_fits(maximum, maximum - 4, 4, 2));
    CHECK(gf_tensor_range_fits(maximum, maximum - 4, 4, 0));
    CHECK(gf_tensor_range_fits(maximum, 0, maximum, 0));
}

std::filesystem::path tiny_gguf_path() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("minimax-loading-safety-" + std::to_string(unique) + ".gguf");
}

bool write_tiny_gguf(const std::filesystem::path & path) {
    constexpr size_t kContextBytes = 4096;
    constexpr int64_t kTensorElements = 4;

    ggml_init_params params = { kContextBytes, nullptr, false };
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    ggml_tensor * tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, kTensorElements);
    ggml_set_name(tensor, "tiny.weight");
    float * data = static_cast<float *>(tensor->data);
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    gguf_add_tensor(gguf, tensor);
    const bool written = gguf_write_to_file(gguf, path.string().c_str(), false);
    gguf_free(gguf);
    ggml_free(context);
    return written;
}

void test_truncated_tiny_gguf() {
    const std::filesystem::path path = tiny_gguf_path();
    std::filesystem::create_directories(path.parent_path());
    CHECK(write_tiny_gguf(path));

    GGUFModel complete = {};
    CHECK(gf_load(&complete, path.string().c_str()));
    size_t tensor_end = 0;
    if (complete.gguf) {
        const int64_t tensor_index = gguf_find_tensor(complete.gguf, "tiny.weight");
        CHECK(tensor_index >= 0);
        if (tensor_index >= 0) {
            const size_t tensor_offset = gguf_get_tensor_offset(complete.gguf, tensor_index);
            const size_t tensor_size = gguf_get_tensor_size(complete.gguf, tensor_index);
            const bool fits =
                gf_tensor_range_fits(complete.file_size, complete.data_offset, tensor_offset, tensor_size);
            CHECK(fits);
            if (fits) {
                const size_t data_size = complete.file_size - complete.data_offset;
                const size_t trailing_size = data_size - tensor_offset - tensor_size;
                tensor_end = complete.file_size - trailing_size;
            }
        }
    }
    gf_close(&complete);

    CHECK(tensor_end > 0);
    std::filesystem::resize_file(path, tensor_end - 1);

    GGUFModel truncated = {};
    CHECK(!gf_load(&truncated, path.string().c_str()));
    std::filesystem::remove(path);
}

constexpr int64_t kStackWidth = 4;

ggml_tensor * add_stack_part(ggml_context * context, gguf_context * gguf, const char * name, ggml_type type,
                             int64_t rows, float base) {
    ggml_tensor * tensor = ggml_new_tensor_2d(context, type, kStackWidth, rows);
    ggml_set_name(tensor, name);
    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        const float value = base + (float) i;
        if (type == GGML_TYPE_F16) {
            static_cast<ggml_fp16_t *>(tensor->data)[i] = ggml_fp32_to_fp16(value);
        } else {
            static_cast<float *>(tensor->data)[i] = value;
        }
    }
    gguf_add_tensor(gguf, tensor);
    return tensor;
}

// Two F32 parts that stack and an F16 part that does not stack with them.
bool write_stacking_gguf(const std::filesystem::path & path) {
    constexpr size_t kContextBytes = 16384;
    ggml_init_params params = { kContextBytes, nullptr, false };
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    add_stack_part(context, gguf, "part.a", GGML_TYPE_F32, 2, 0.0f);
    add_stack_part(context, gguf, "part.b", GGML_TYPE_F32, 3, 100.0f);
    add_stack_part(context, gguf, "part.half", GGML_TYPE_F16, 2, 200.0f);
    const bool written = gguf_write_to_file(gguf, path.string().c_str(), false);
    gguf_free(gguf);
    ggml_free(context);
    return written;
}

std::vector<float> expected_stack() {
    std::vector<float> values;
    for (int64_t i = 0; i < 2 * kStackWidth; i++) {
        values.push_back((float) i);
    }
    for (int64_t i = 0; i < 3 * kStackWidth; i++) {
        values.push_back(100.0f + (float) i);
    }
    return values;
}

// Stacking must concatenate the parts row for row, and quietly decline (no
// load error) whenever the parts cannot share one tensor, so the caller falls
// back to loading them one by one.
void test_stacked_projection_loading() {
    const std::filesystem::path path = tiny_gguf_path();
    CHECK(write_stacking_gguf(path));
    GGUFModel gf = {};
    CHECK(gf_load(&gf, path.string().c_str()));
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);

    WeightCtx wctx = {};
    wctx_init(&wctx, 4);
    std::map<std::string, ggml_tensor *> tmap;
    std::vector<std::string>             errors;
    MM3Loader                            loader{ &wctx, &gf, &tmap, &errors };

    ggml_tensor * stacked = loader.try_stacked("part.ab", { "part.a", "part.b" }, kStackWidth, { 2, 3 });
    CHECK(stacked != nullptr);
    CHECK(loader.try_stacked("part.mixed", { "part.a", "part.half" }, kStackWidth, { 2, 2 }) == nullptr);
    CHECK(loader.try_stacked("part.missing", { "part.a", "part.none" }, kStackWidth, { 2, 2 }) == nullptr);
    CHECK(loader.try_stacked("part.shape", { "part.a", "part.b" }, kStackWidth, { 2, 2 }) == nullptr);
    CHECK(errors.empty());
    CHECK(tmap.size() == 1 && tmap.count("part.ab") == 1);

    if (stacked && cpu && wctx_alloc(&wctx, cpu)) {
        CHECK(stacked->type == GGML_TYPE_F32 && stacked->ne[0] == kStackWidth && stacked->ne[1] == 5);
        std::vector<float> values((size_t) ggml_nelements(stacked));
        ggml_backend_tensor_get(stacked, values.data(), 0, ggml_nbytes(stacked));
        CHECK(values == expected_stack());
    }

    wctx_free(&wctx);
    if (cpu) {
        ggml_backend_free(cpu);
    }
    gf_close(&gf);
    std::filesystem::remove(path);
}

MM3LmConfig valid_lm_config() {
    MM3LmConfig config;
    config.block_count = 1;
    config.context_length = 10240;
    config.embedding_length = 4;
    config.head_count = 1;
    config.head_count_kv = 1;
    config.key_length = 4;
    config.value_length = 4;
    config.vocab_size = 100;
    config.semantic_vocab_offset = 10;
    config.semantic_vocab_size = 20;
    config.num_codebooks = 2;
    config.eos_audio = 1;
    config.frame_rate = 25;
    config.max_audio_frames = 9000;
    config.max_prompt_tokens = 5000;
    return config;
}

void test_context_metadata_validation() {
    std::vector<std::string> errors;
    MM3LmConfig config = valid_lm_config();
    mm3_validate_lm_config(config, &errors);
    CHECK(errors.empty());

    errors.clear();
    config.context_length = 0;
    mm3_validate_lm_config(config, &errors);
    CHECK(contains_error(errors, "qwen3.context_length is 0 or missing"));

    errors.clear();
    config = valid_lm_config();
    config.context_length = 8999;
    mm3_validate_lm_config(config, &errors);
    CHECK(contains_error(errors, "max_audio_frames exceeds qwen3.context_length"));

    errors.clear();
    config = valid_lm_config();
    config.max_prompt_tokens = 10241;
    mm3_validate_lm_config(config, &errors);
    CHECK(contains_error(errors, "max_prompt_tokens exceeds qwen3.context_length"));
}

void test_context_position_validation() {
    const int64_t maximum = std::numeric_limits<int64_t>::max();

    CHECK(mm3_lm_positions_fit(14000, 5000, 9000));
    CHECK(!mm3_lm_positions_fit(13999, 5000, 9000));
    CHECK(!mm3_lm_positions_fit(14000, 5000, 9001));
    CHECK(!mm3_lm_positions_fit(14000, 14001, 0));
    CHECK(!mm3_lm_positions_fit(0, 0, 0));
    CHECK(!mm3_lm_positions_fit(14000, -1, 1));
    CHECK(mm3_lm_positions_fit(maximum, maximum, 0));
    CHECK(!mm3_lm_positions_fit(maximum, maximum, 1));
}

void test_context_bucket_validation() {
    int64_t bucket = 0;
    CHECK(mm3_lm_bucket_for_context(512, 1000, &bucket));
    CHECK(bucket == 512);
    CHECK(mm3_lm_bucket_for_context(513, 1000, &bucket));
    CHECK(bucket == 768);
    CHECK(mm3_lm_bucket_for_context(900, 1000, &bucket));
    CHECK(bucket == 1000);
    CHECK(mm3_lm_bucket_for_context(1000, 1000, &bucket));
    CHECK(bucket == 1000);
    CHECK(!mm3_lm_bucket_for_context(1001, 1000, &bucket));
    CHECK(!mm3_lm_bucket_for_context(1, 0, &bucket));
    CHECK(!mm3_lm_bucket_for_context(1, 1, nullptr));
    CHECK(mm3_lm_bucket_for_context(std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(), &bucket));
    CHECK(bucket == std::numeric_limits<int64_t>::max());
}

void test_backend_release_after_post_probe_failure() {
    const std::filesystem::path missing =
        std::filesystem::temp_directory_path() / "minimax-loading-safety-missing.gguf";
    std::filesystem::remove(missing);

    MM3Model model;
    model.lm_file.found = true;
    model.lm_file.probe_ok = true;
    model.lm_file.path = missing.string();
    model.synth_file.found = true;
    model.synth_file.probe_ok = true;
    model.synth_file.path = missing.string();

    CHECK(g_backend_refs == 0);
    std::string error;
    CHECK(!mm3_load(&model, &error));
    CHECK(!error.empty());
    CHECK(!model.loaded);
    CHECK(!model.backend_ref);
    CHECK(model.backend == nullptr);
    CHECK(model.cpu_backend == nullptr);
    CHECK(g_backend_refs == 0);
    CHECK(g_backend_cache.backend == nullptr);
    CHECK(g_backend_cache.cpu_backend == nullptr);

    mm3_unload(&model);
    mm3_unload(&model);
    CHECK(!model.backend_ref);
    CHECK(model.backend == nullptr);
    CHECK(model.cpu_backend == nullptr);
    CHECK(g_backend_refs == 0);
}

void test_coreml_failure_precedes_native_allocation() {
    MM3Model model;
    model.lm_file.found = model.lm_file.probe_ok = true;
    model.synth_file.found = model.synth_file.probe_ok = true;
    model.coreml_conditioner_path = "missing-minimax-conditioner.mlmodelc";
    model.synth_cfg.cond.hidden_dim = 3;
    model.synth_cfg.cond.out_dim = 2;
    model.synth_cfg.cond.kernel_size = 3;
    model.synth_cfg.cond.padding = 1;
    std::string error;
    CHECK(!mm3_load(&model, &error));
    CHECK(!error.empty());
    CHECK(model.coreml_conditioner == nullptr);
    CHECK(!model.loaded && !model.backend_ref);
    CHECK(!model.wctx_lm.ctx && !model.wctx_synth.ctx);
    CHECK(g_backend_refs == 0);
    model.synth_cfg.cond.kernel_size = 1;
    CHECK(!mm3_load(&model, &error));
    CHECK(error == "Core ML conditioner requires kernel=3 and padding=1");
    mm3_unload(&model);
    CHECK(g_backend_refs == 0);
}

void test_public_engine_forwards_coreml_path() {
    minimax_tiny::Pair pair("coreml-options");
    CHECK(pair.written);
    tts_cpp::minimax::EngineOptions options;
    options.lm_model_path = pair.lm.string();
    options.synth_model_path = pair.synth.string();
    options.device = "cpu";
    options.coreml_conditioner_path = "missing-minimax-conditioner.mlmodelc";
    bool rejected = false;
    try {
        const auto engine = tts_cpp::minimax::Engine::create(options);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("Core ML conditioner") != std::string::npos;
    }
    CHECK(rejected);
    options.coreml_conditioner_path.clear();
    CHECK(tts_cpp::minimax::Engine::create(options) != nullptr);
}

}

int main() {
    test_typed_metadata_getters();
    test_tensor_range_validation();
    test_truncated_tiny_gguf();
    test_stacked_projection_loading();
    test_context_metadata_validation();
    test_context_position_validation();
    test_context_bucket_validation();
    test_backend_release_after_post_probe_failure();
    test_coreml_failure_precedes_native_allocation();
    test_public_engine_forwards_coreml_path();
    std::fprintf(stderr, "[test-minimax-loading-safety] %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
