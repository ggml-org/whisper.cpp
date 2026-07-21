#include "supertonic_internal.h"
#include "npy.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR\n", argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argv[2];
    const std::string text = "The quick brown fox jumps over the lazy dog.";

    supertonic_model model;
    if (!load_supertonic_gguf(model_path, model)) {
        return 1;
    }

    int rc = 0;
    try {
        std::vector<int32_t> ids;
        std::string normalized;
        std::string error;
        if (!supertonic_text_to_ids(model, text, "en", ids, &normalized, &error)) {
            fprintf(stderr, "text_to_ids failed: %s\n", error.c_str());
            rc = 1;
        } else {
            npy_array ref = npy_load(ref_dir + "/text_ids.npy");
            if (ref.dtype != "<i8" || ref.shape.size() != 2 || ref.shape[0] != 1) {
                fprintf(stderr, "unexpected text_ids.npy shape/dtype\n");
                rc = 1;
            } else if ((size_t) ref.shape[1] != ids.size()) {
                fprintf(stderr, "token length mismatch: got %zu ref %lld\n",
                        ids.size(), (long long) ref.shape[1]);
                rc = 1;
            } else {
                const int64_t * ref_ids = reinterpret_cast<const int64_t *>(ref.data.data());
                for (size_t i = 0; i < ids.size(); ++i) {
                    if ((int64_t) ids[i] != ref_ids[i]) {
                        fprintf(stderr, "token mismatch at %zu: got %d ref %lld\n",
                                i, ids[i], (long long) ref_ids[i]);
                        rc = 1;
                        break;
                    }
                }
            }
            if (rc == 0) {
                (void) require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer2.weight");
                fprintf(stderr, "supertonic preprocess/load: PASS (%zu tokens, normalized='%s')\n",
                        ids.size(), normalized.c_str());
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test failed: %s\n", e.what());
        rc = 1;
    }

    free_supertonic_model(model);
    return rc;
}
