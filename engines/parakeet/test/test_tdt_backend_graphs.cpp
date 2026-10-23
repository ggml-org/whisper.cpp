// The TDT decoder graphs execute directly on their backend, without the
// scheduler's per-op CPU fallback, so the backend must accept every node with
// its operands where they were allocated: a node it refuses still runs and
// decodes garbage (the stacked [w_ih | w_hh] LSTM weight once sat outside
// Hexagon's repack buffer, and TranscribePsy decoded "myell W"). Loads the
// model on the requested backend, prepares the decoder with its default graph
// refinements, checks every node of the three fixed graphs, and decodes one
// clip from a CPU encoder pass.
//   test-tdt-backend-graphs <tdt.gguf> <wav> [--backend NAME]
// Exit 0 on success, 3 when the backend is not available, 1 on failure.
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kBackendUnavailable = 11;
constexpr int kSkipReturnCode     = 3;

struct Opts {
    std::string model_path;
    std::string wav_path;
    std::string backend = "cpu";
};

int parse_args(int argc, char ** argv, Opts & o) {
    if (argc < 3) return 2;
    o.model_path = argv[1];
    o.wav_path   = argv[2];
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--backend") != 0) return 2;
        o.backend = argv[i + 1];
    }
    return 0;
}

int count_unsupported(const char * name, ggml_backend_t backend, ggml_cgraph * graph) {
    int refused = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (ggml_backend_supports_op(backend, node)) continue;
        std::fprintf(stderr, "[tdt-backend-graphs] %s node %d %s (%s, src0 %s) refused by %s\n", name, i,
                     ggml_op_desc(node), node->name, ggml_type_name(node->type),
                     node->src[0] ? ggml_type_name(node->src[0]->type) : "-", ggml_backend_name(backend));
        ++refused;
    }
    return refused;
}

int encode_on_cpu(const Opts & o, parakeet::EncoderOutputs & enc) {
    using namespace parakeet;
    ParakeetCtcModel cpu;
    if (load_from_gguf(o.model_path, cpu, 0, 0, false, "cpu") != 0) return 1;
    std::vector<float> samples;
    int sr = 0;
    if (load_wav_mono_f32(o.wav_path, samples, sr) != 0) return 1;
    std::vector<float> mel;
    int n_frames = 0;
    if (compute_log_mel(samples.data(), (int) samples.size(), cpu.mel_cfg, mel, n_frames) != 0) return 1;
    return run_encoder(cpu, mel.data(), n_frames, cpu.mel_cfg.n_mels, enc, -1, false) == 0 ? 0 : 1;
}

int check_decoder(const parakeet::ParakeetCtcModel & model, const parakeet::EncoderOutputs & enc) {
    using namespace parakeet;
    TdtRuntimeWeights rt;
    if (int rc = tdt_prepare_runtime(model, rt); rc != 0) {
        std::fprintf(stderr, "[tdt-backend-graphs] tdt_prepare_runtime rc=%d\n", rc);
        return 1;
    }
    std::printf("[tdt-backend-graphs] decoder on %s: graphs %d, fused cell %d, stacked input %d, argmax on device %d\n",
                ggml_backend_name(rt.backend), (int) rt.use_graphs, (int) rt.fused_lstm_cell,
                (int) rt.concat_lstm_input, (int) rt.argmax_on_gpu);
    int refused = 0;
    if (rt.use_graphs) {
        refused += count_unsupported("g_lstm", rt.backend, rt.g_lstm);
        refused += count_unsupported("g_joint", rt.backend, rt.g_joint);
        refused += count_unsupported("g_lstm_joint", rt.backend, rt.g_lstm_joint);
    }
    TdtDecodeOptions dopts;
    TdtDecodeResult  dres;
    if (int rc = tdt_greedy_decode(model, rt, enc.encoder_out.data(), enc.n_enc_frames, enc.d_model, dopts, dres);
        rc != 0) {
        std::fprintf(stderr, "[tdt-backend-graphs] tdt_greedy_decode rc=%d\n", rc);
        return 1;
    }
    std::printf("[tdt-backend-graphs] %d refused nodes, %zu tokens decoded\n", refused, dres.token_ids.size());
    return refused == 0 && !dres.token_ids.empty() ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    using namespace parakeet;
    Opts o;
    if (parse_args(argc, argv, o) != 0) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> <wav> [--backend NAME]\n", argv[0]);
        return 2;
    }
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(o.model_path, model, 0, 0, false, o.backend); rc != 0) {
        if (rc == kBackendUnavailable) {
            std::printf("[tdt-backend-graphs] SKIP: backend %s unavailable\n", o.backend.c_str());
            return kSkipReturnCode;
        }
        std::fprintf(stderr, "[tdt-backend-graphs] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    EncoderOutputs enc;
    if (encode_on_cpu(o, enc) != 0) {
        std::fprintf(stderr, "[tdt-backend-graphs] CPU encoder pass failed\n");
        return 1;
    }
    const int rc = check_decoder(model, enc);
    std::printf("[tdt-backend-graphs] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
