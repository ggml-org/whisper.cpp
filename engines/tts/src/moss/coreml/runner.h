#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct moss_coreml_model;

struct moss_coreml_model * moss_coreml_load(const char * path_mlmodelc, const char * compute_units);

void moss_coreml_free(struct moss_coreml_model * model);

int moss_coreml_feature_dims(const struct moss_coreml_model * model, const char * name, int64_t * dims,
                             int max_dims);

int moss_coreml_has_state(const struct moss_coreml_model * model);

int moss_coreml_reset_state(struct moss_coreml_model * model);

int moss_coreml_predict(struct moss_coreml_model * model,
                        const char * const * input_names, const float * const * inputs, int n_inputs,
                        const char * const * output_names, float * const * outputs, int n_outputs);

int moss_coreml_predict_batch(struct moss_coreml_model * model,
                              const char * const * input_names, const float * const * inputs, int n_inputs,
                              const char * const * output_names, float * const * outputs, int n_outputs,
                              int batch_size);

const char * moss_coreml_label(const struct moss_coreml_model * model);

#if defined(__cplusplus)
}
#endif
