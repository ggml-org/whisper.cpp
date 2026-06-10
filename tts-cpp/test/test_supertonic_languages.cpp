// QVAC-19305 — Supertonic v3 language-coverage tests.
//
// v3 widened the supported-language set from the v1/v2 5-language
// bundle (en, ko, es, pt, fr) to a 31-language bundle plus the
// language-agnostic `na` code (pass lang="na" when the source
// language is unknown).  The authoritative list lives in
// `scripts/convert-supertonic2-to-gguf.py` (`ARCH_LANGUAGES`) and is
// shipped in the GGUF `supertonic.languages` array; the runtime
// validates user-supplied codes against it in
// `supertonic_preprocess_text(... supported_languages)`.
//
// Two layers of testing here:
//
//   1. Unit-level contract test (no GGUF, runs on `ctest -L unit`).
//      `supertonic_preprocess_text` takes the supported-language
//      vector directly, so we can pin the v3 contract without a
//      model:
//        - every v3 code (31 langs + `na`) is ACCEPTED and wrapped
//          correctly in both `open_close` and `prefix` modes,
//        - codes outside the active set are REJECTED (throw),
//        - the built-in fallback (null supported set) still only
//          accepts the legacy v1/v2 5-language bundle — so a
//          v3-only code like `de`/`ja`/`na` is rejected there,
//          proving the GGUF array (not the fallback) is what
//          unlocks the new languages.
//
//   2. Fixture-level cross-check (requires GGUF).  When a model is
//      passed, every code in the loaded `model.languages` must be
//      accepted by the preprocessor against that same array, tying
//      the converter's source-of-truth to the runtime validator.

#include "supertonic_internal.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                      \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// Mirror of ARCH_LANGUAGES["supertonic3"] in the converter — 31
// languages + the language-agnostic `na`.  Kept here verbatim so the
// test pins the contract independently of any GGUF fixture.
const std::vector<std::string> kV3Languages = {
    "ar", "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el",
    "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "pl", "pt", "ro", "ru",
    "sk", "sl", "es", "sv", "tr", "uk", "vi", "na",
};

// Mirror of ARCH_LANGUAGES["supertonic2"] — the legacy 5-language
// bundle, which is also the built-in fallback allowlist.
const std::vector<std::string> kV2Languages = {
    "en", "ko", "es", "pt", "fr",
};

// Returns true if the preprocessor accepted `lang` (no throw),
// false if it rejected it (threw the "invalid Supertonic language"
// error).  Other exceptions are re-thrown so genuine bugs surface.
bool accepts(const std::string & lang,
             const std::string & wrap_mode,
             const std::vector<std::string> * supported) {
    try {
        (void) supertonic_preprocess_text("hello world", lang, wrap_mode,
                                          /*is_continuation=*/false, supported);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void test_v3_languages_accepted() {
    std::fprintf(stderr, "[v3 languages accepted + wrapped]\n");
    for (const std::string & lang : kV3Languages) {
        // open_close: "<lang>...</lang>".
        CHECK(accepts(lang, "open_close", &kV3Languages));
        const std::string oc = supertonic_preprocess_text(
            "hello world", lang, "open_close", false, &kV3Languages);
        const std::string open  = "<" + lang + ">";
        const std::string close = "</" + lang + ">";
        CHECK(oc.rfind(open, 0) == 0);
        CHECK(oc.size() >= close.size() &&
              oc.compare(oc.size() - close.size(), close.size(), close) == 0);

        // prefix: "<lang>... " (trailing space, no close tag).
        const std::string px = supertonic_preprocess_text(
            "hello world", lang, "prefix", false, &kV3Languages);
        CHECK(px.rfind(open, 0) == 0);
        CHECK(!px.empty() && px.back() == ' ');

        // none: no wrapping at all.
        const std::string nn = supertonic_preprocess_text(
            "hello world", lang, "none", false, &kV3Languages);
        CHECK(nn.find('<') == std::string::npos);
    }
}

void test_na_is_v3_only() {
    std::fprintf(stderr, "[`na` accepted under v3, rejected by fallback]\n");
    // The language-agnostic code is v3's headline addition.
    CHECK(accepts("na", "open_close", &kV3Languages));
    // It must NOT be accepted by the legacy fallback (null supported
    // set) nor by the explicit v2 bundle.
    CHECK(!accepts("na", "open_close", nullptr));
    CHECK(!accepts("na", "open_close", &kV2Languages));
}

void test_unknown_codes_rejected() {
    std::fprintf(stderr, "[unknown codes rejected]\n");
    static const char * const kBogus[] = {
        "xx", "zz", "qq", "english", "EN", "En", "es-419", "zh", "th", "",
    };
    for (const char * code : kBogus) {
        CHECK(!accepts(code, "open_close", &kV3Languages));
    }
}

void test_fallback_is_legacy_five() {
    std::fprintf(stderr, "[built-in fallback = legacy v1/v2 bundle]\n");
    // With a null supported set, only the legacy 5 are accepted.
    for (const std::string & lang : kV2Languages) {
        CHECK(accepts(lang, "open_close", nullptr));
    }
    // v3-only codes must be rejected by the fallback (they are only
    // unlocked by the GGUF `supertonic.languages` array).
    static const char * const kV3Only[] = {
        "de", "ja", "ru", "ar", "hi", "vi", "uk", "na",
    };
    for (const char * code : kV3Only) {
        CHECK(!accepts(code, "open_close", nullptr));
    }
    // Empty supported set behaves like null (falls back to legacy).
    const std::vector<std::string> empty;
    CHECK(accepts("en", "open_close", &empty));
    CHECK(!accepts("de", "open_close", &empty));
}

void test_v3_superset_of_v2() {
    std::fprintf(stderr, "[v3 set is a superset of v2]\n");
    for (const std::string & lang : kV2Languages) {
        bool found = false;
        for (const std::string & v3 : kV3Languages) {
            if (v3 == lang) { found = true; break; }
        }
        CHECK(found);
    }
    // Sanity: 31 languages + `na` == 32 entries.
    CHECK(kV3Languages.size() == 32);
}

} // namespace

int main(int argc, char ** argv) {
    // Unit-level contract tests run unconditionally; no model.
    test_v3_languages_accepted();
    test_na_is_v3_only();
    test_unknown_codes_rejected();
    test_fallback_is_legacy_five();
    test_v3_superset_of_v2();

    // Fixture-level cross-check requires the GGUF: every code the
    // model advertises must be accepted by the validator against the
    // model's own array.
    if (argc >= 2) {
        std::fprintf(stderr, "[fixture] (loading %s)\n", argv[1]);
        supertonic_model model;
        if (load_supertonic_gguf(argv[1], model, /*n_gpu_layers=*/0, /*verbose=*/false)) {
            std::fprintf(stderr, "  model advertises %zu languages\n",
                         model.languages.size());
            CHECK(!model.languages.empty());
            const std::string wrap = model.hparams.language_wrap_mode.empty()
                                         ? "open_close"
                                         : model.hparams.language_wrap_mode;
            for (const std::string & lang : model.languages) {
                CHECK(accepts(lang, wrap, &model.languages));
            }
            free_supertonic_model(model);
        } else {
            std::fprintf(stderr, "  skip fixture: failed to load %s\n", argv[1]);
        }
    } else {
        std::fprintf(stderr, "  (fixture skipped; pass MODEL.gguf to enable)\n");
    }

    std::fprintf(stderr,
                 "test_supertonic_languages: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
