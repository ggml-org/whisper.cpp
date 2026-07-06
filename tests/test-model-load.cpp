#include "ggml.h"
#include "whisper.h"

#include <cstdint>
#include <cstdio>
#include <fstream>

static void write_u32(std::ofstream & fout, uint32_t value) {
    fout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

static void write_i32(std::ofstream & fout, int32_t value) {
    fout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

int main() {
    const char * path = "invalid-ftype.ggml";

    {
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return 1;
        }

        write_u32(fout, GGML_FILE_MAGIC);
        write_i32(fout, 1);    // n_vocab
        write_i32(fout, 1500); // n_audio_ctx
        write_i32(fout, 384);  // n_audio_state
        write_i32(fout, 6);    // n_audio_head
        write_i32(fout, 4);    // n_audio_layer
        write_i32(fout, 448);  // n_text_ctx
        write_i32(fout, 384);  // n_text_state
        write_i32(fout, 6);    // n_text_head
        write_i32(fout, 4);    // n_text_layer
        write_i32(fout, 80);   // n_mels
        write_i32(fout, 5);    // invalid ftype

        if (!fout) {
            return 2;
        }
    }

    whisper_context_params params = whisper_context_default_params();
    whisper_context * ctx = whisper_init_from_file_with_params(path, params);
    if (ctx != nullptr) {
        whisper_free(ctx);
        std::remove(path);
        return 3;
    }

    std::remove(path);
    return 0;
}
