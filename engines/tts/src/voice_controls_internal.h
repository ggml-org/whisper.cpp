#pragma once

// Value canonicalization shared by the engines' own field validators, so
// there is one case-folding and one error format across the library.

#include <initializer_list>
#include <string>
#include <vector>

namespace tts_cpp {
namespace controls {
namespace detail {

// ASCII-only on purpose: locale-independent (std::tolower would misbehave
// under e.g. a Turkish LC_CTYPE), and all valid field values are ASCII.
std::string to_lower(const std::string & value);

// Returns the canonical lowercase form, or throws std::invalid_argument
// formatted as: <context>: invalid <field> "<value>" (valid: a, b, c)
std::string canon_in(const char * context, const char * field,
                     const std::string & value,
                     std::initializer_list<const char *> valid);

std::string canon_in(const char * context, const char * field,
                     const std::string & value,
                     const std::vector<std::string> & valid);

} // namespace detail
} // namespace controls
} // namespace tts_cpp
