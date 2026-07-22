package io.github.ggerganov.whispercpp.ggml;

public enum GgmlType {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    REMOVED_GGML_TYPE_Q4_2,  // support has been removed
    REMOVED_GGML_TYPE_Q4_3, // support has been removed
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
}
