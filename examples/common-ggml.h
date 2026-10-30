#pragma once

#include "ggml.h"

#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

// Structure for per-tensor quantization specification
struct tensor_quant_spec {
    std::string pattern;  // regex pattern to match tensor names
    ggml_type quant_type; // quantization type for matched tensors
};

enum ggml_ftype ggml_parse_ftype(const char * str);

// Parse a quantization type string (e.g., "q4_0", "q8_0")
ggml_type ggml_parse_qtype(const char * str);

void ggml_print_ftypes(FILE * fp = stderr);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);

// Extended quantization function with per-tensor quantization support
bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip,
        const std::vector<tensor_quant_spec> & tensor_quant_specs);
