#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include <cstdio>

static int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

int main() {
    const float model_rate = 0.7f;

    CHECK(resolve_s3gen_cfg_rate(-1.0f, model_rate)   == model_rate, "sentinel -1 -> model rate");
    CHECK(resolve_s3gen_cfg_rate(-0.001f, model_rate) == model_rate, "any negative -> model rate");
    CHECK(resolve_s3gen_cfg_rate(-1.0f, 0.0f)         == 0.0f,       "sentinel -> model rate of 0");
    CHECK(resolve_s3gen_cfg_rate(-1.0f, 1.3f)         == 1.3f,       "sentinel -> model rate 1.3");

    CHECK(resolve_s3gen_cfg_rate(0.0f, model_rate)  == 0.0f, "caller 0 overrides -> cfg off");
    CHECK(resolve_s3gen_cfg_rate(-0.0f, model_rate) == 0.0f, "caller -0.0 overrides -> cfg off");

    CHECK(resolve_s3gen_cfg_rate(0.5f, model_rate) == 0.5f,       "caller 0.5 overrides model rate");
    CHECK(resolve_s3gen_cfg_rate(model_rate, 0.0f) == model_rate, "caller 0.7 overrides model rate 0");
    CHECK(resolve_s3gen_cfg_rate(2.0f, model_rate) == 2.0f,       "caller 2.0 overrides model rate");

    s3gen_synthesize_opts opts;
    CHECK(opts.cfg_rate < 0.0f, "default opts cfg_rate is the sentinel");
    CHECK(resolve_s3gen_cfg_rate(opts.cfg_rate, model_rate) == model_rate,
          "untouched opts resolves to the model rate");

    CHECK(apply_cfg_rate_env_override(model_rate, nullptr) == model_rate, "null env -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "")      == model_rate, "empty env -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "   ")   == model_rate, "whitespace env -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "foo")   == model_rate, "unparseable env -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "0.5x")  == model_rate, "trailing garbage -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "-1")    == model_rate, "negative sentinel env -> keep current");
    CHECK(apply_cfg_rate_env_override(model_rate, "-0.7")  == model_rate, "any negative env -> keep current");

    CHECK(apply_cfg_rate_env_override(model_rate, "0")   == 0.0f, "env 0 -> cfg off");
    CHECK(apply_cfg_rate_env_override(model_rate, "0.5") == 0.5f, "env 0.5 -> override");
    CHECK(apply_cfg_rate_env_override(model_rate, "1.3") == 1.3f, "env 1.3 -> override");

    if (g_failures) {
        fprintf(stderr, "test-s3gen-cfg-rate: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-s3gen-cfg-rate: all checks passed\n");
    return 0;
}
