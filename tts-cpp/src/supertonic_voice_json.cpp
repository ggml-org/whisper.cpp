#include "supertonic_voice_json.h"

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

long long shape_product(const json & shape, bool & ok) {
    long long prod = 1;
    ok = true;
    for (const auto & dim : shape) {
        if (!dim.is_number_integer()) {
            ok = false;
            return 0;
        }
        prod *= dim.get<long long>();
    }
    return prod;
}

bool shape_matches_data(const json & field, size_t data_len, std::string * mismatch) {
    if (!field.is_object()) {
        return true;
    }
    const auto sit = field.find("shape");
    if (sit == field.end() || !sit->is_array()) {
        return true;
    }
    bool ok = false;
    const long long prod = shape_product(*sit, ok);
    if (ok && prod > 0 && static_cast<size_t>(prod) != data_len) {
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
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = "voice json: cannot open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_supertonic_voice_json(ss.str(), out, err);
}

}  // namespace tts_cpp::supertonic::detail
