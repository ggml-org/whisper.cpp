#pragma once
// Portable setenv/unsetenv for the test harnesses. MSVC's CRT has no
// POSIX setenv/unsetenv (only _putenv_s), so provide same-named shims
// there; every other platform gets the real functions from <cstdlib>.
#include <cstdlib>
#include <initializer_list>
#include <string>
#ifdef _MSC_VER
inline int setenv(const char * name, const char * value, int overwrite) {
    // POSIX: with overwrite=0 an existing variable must be left untouched
    // (and 0 returned) — don't let the shim diverge from that silently.
    if (!overwrite && std::getenv(name) != nullptr) {
        return 0;
    }
    return _putenv_s(name, value);
}
inline int unsetenv(const char * name) {
    return _putenv_s(name, ""); // empty value removes the variable
}
#endif

// Writable scratch directory for tests that emit temp fixtures. POSIX
// runners set TMPDIR; Windows sets TEMP/TMP and has no /tmp, so fall
// through the whole chain before defaulting.
inline std::string test_tmpdir() {
    for (const char * var : { "TMPDIR", "TEMP", "TMP" }) {
        if (const char * v = std::getenv(var); v && *v) return v;
    }
    return "/tmp";
}
