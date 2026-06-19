#include "supertonic_voice_json.h"

// nlohmann/json is vendored as a single header (src/json.hpp) rather than
// pulled via vcpkg: tts-cpp builds standalone (its CMake only
// find_package()s ggml / OpenMP / BLAS — there is no vcpkg manifest in
// this tree), and the parser must stay robust against untrusted input, so
// a battle-tested parser beats a hand-rolled one.  Mirrors the existing
// single-header vendoring convention in this tree (e.g. npy.h).
#include "json.hpp"

#include <fstream>
#include <sstream>

namespace tts_cpp::supertonic::detail {

using nlohmann::json;

namespace {

// A voice field is either the canonical `{ "data": [...] }` object or a
// bare `[...]` array. Returns the data array, or nullptr when neither.
const json * find_data_array(const json & field) {
    if (field.is_object()) {
        const auto dit = field.find("data");
        return dit != field.end() ? &(*dit) : nullptr;
    }
    if (field.is_array()) {
        return &field;
    }
    return nullptr;
}

bool to_f32_vector(const json & arr, std::vector<float> & out) {
    out.clear();
    out.reserve(arr.size());
    for (const auto & v : arr) {
        if (!v.is_number()) {
            return false;
        }
        out.push_back(v.get<float>());
    }
    return true;
}

// Product of a shape array.  Sets ok=false if any dimension is not a
// positive integer — a zero / negative / non-integer dim is an invalid
// shape, not a tensor we should silently accept.
long long shape_product(const json & shape, bool & ok) {
    long long prod = 1;
    ok = true;
    for (const auto & dim : shape) {
        if (!dim.is_number_integer() || dim.get<long long>() <= 0) {
            ok = false;
            return 0;
        }
        prod *= dim.get<long long>();
    }
    return prod;
}

// `shape` is optional; when present it must be a non-empty array of
// positive integers whose product equals the flattened data length.
bool shape_matches_data(const json & field, size_t data_len, std::string * mismatch) {
    if (!field.is_object()) {
        return true;
    }
    const auto sit = field.find("shape");
    if (sit == field.end()) {
        return true;
    }
    if (!sit->is_array() || sit->empty()) {
        if (mismatch) {
            *mismatch = "shape must be a non-empty array of positive integers";
        }
        return false;
    }
    bool ok = false;
    const long long prod = shape_product(*sit, ok);
    if (!ok) {
        if (mismatch) {
            *mismatch = "shape must contain only positive integer dimensions";
        }
        return false;
    }
    if (static_cast<size_t>(prod) != data_len) {
        if (mismatch) {
            *mismatch = "shape product " + std::to_string(prod) +
                        " != data length " + std::to_string(data_len);
        }
        return false;
    }
    return true;
}

bool read_f32_field(const json & node, const char * key,
                    std::vector<float> & out, std::string * err) {
    const auto it = node.find(key);
    if (it == node.end()) {
        if (err) *err = std::string("voice json: missing '") + key + "'";
        return false;
    }

    const json & field = *it;
    const json * data  = find_data_array(field);
    if (!data || !data->is_array()) {
        if (err) *err = std::string("voice json: '") + key +
                        "' must be an object with a 'data' array or a bare array";
        return false;
    }

    if (!to_f32_vector(*data, out)) {
        if (err) *err = std::string("voice json: '") + key +
                        "'.data contains a non-numeric element";
        return false;
    }

    std::string mismatch;
    if (!shape_matches_data(field, out.size(), &mismatch)) {
        if (err) *err = std::string("voice json: '") + key + "' " + mismatch;
        return false;
    }

    return true;
}

void read_optional_name(const json & j, external_voice & out) {
    const auto nit = j.find("name");
    if (nit != j.end() && nit->is_string()) {
        out.name = nit->get<std::string>();
        return;
    }
    const auto mit = j.find("metadata");
    if (mit != j.end() && mit->is_object()) {
        const auto mnit = mit->find("name");
        if (mnit != mit->end() && mnit->is_string()) {
            out.name = mnit->get<std::string>();
        }
    }
}

}  // namespace

bool parse_supertonic_voice_json(const std::string & json_text,
                                 external_voice & out,
                                 std::string * err) {
    json j;
    try {
        j = json::parse(json_text);
    } catch (const std::exception & e) {
        if (err) *err = std::string("voice json: parse error: ") + e.what();
        return false;
    }

    if (!j.is_object()) {
        if (err) *err = "voice json: top-level value must be an object";
        return false;
    }

    if (!read_f32_field(j, "style_ttl", out.ttl, err)) return false;
    if (!read_f32_field(j, "style_dp", out.dp, err)) return false;

    if (out.ttl.empty() || out.dp.empty()) {
        if (err) *err = "voice json: style_ttl and style_dp must be non-empty";
        return false;
    }

    read_optional_name(j, out);
    return true;
}

bool load_supertonic_voice_json_file(const std::string & path,
                                     external_voice & out,
                                     std::string * err) {
    // Generous upper bound on a voice JSON.  A full voice is ~13k floats
    // (style_ttl [1,50,256] + style_dp [1,8,16]); even at full text
    // precision that's well under a megabyte.  16 MiB leaves headroom for
    // metadata while rejecting a pathological / malicious file before we
    // read it whole and build a DOM.
    constexpr std::streamoff MAX_VOICE_JSON_BYTES = 16 * 1024 * 1024;

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err) *err = "voice json: cannot open file: " + path;
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size > MAX_VOICE_JSON_BYTES) {
        if (err) {
            *err = "voice json: file too large (" + std::to_string(size) +
                   " bytes > " + std::to_string(MAX_VOICE_JSON_BYTES) + " byte limit)";
        }
        return false;
    }
    f.seekg(0, std::ios::beg);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_supertonic_voice_json(ss.str(), out, err);
}

}  // namespace tts_cpp::supertonic::detail
