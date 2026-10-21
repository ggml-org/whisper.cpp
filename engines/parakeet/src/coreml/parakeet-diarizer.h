#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_diarizer;

struct parakeet_coreml_rows {
    const char  * name;
    int64_t       row_offset;
    int64_t       rows;
    int64_t       columns;
    const float * data;
};

struct parakeet_coreml_rows_out {
    const char * name;
    int64_t      row_offset;
    int64_t      rows;
    int64_t      columns;
    float      * data;
};

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init(const char * path_mlmodelc,
                                                                const char * units_env);

void parakeet_coreml_diarizer_free(struct parakeet_coreml_diarizer * ctx);

int64_t parakeet_coreml_diarizer_capacity(const struct parakeet_coreml_diarizer * ctx);

int parakeet_coreml_diarizer_run(struct parakeet_coreml_diarizer * ctx,
                                 int64_t valid_frames,
                                 const struct parakeet_coreml_rows * inputs,
                                 int n_inputs,
                                 const struct parakeet_coreml_rows_out * outputs,
                                 int n_outputs);

const char * parakeet_coreml_diarizer_label(const struct parakeet_coreml_diarizer * ctx);

#if defined(__cplusplus)
}
#endif
