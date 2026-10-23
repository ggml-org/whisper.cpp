#pragma once

#if defined(MM3_ANE_VALIDATION) || defined(AUDIOGEN_USE_MINIMAX_COREML)
#define MM3_COREML_RUNTIME_ENABLED
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct MM3Model;
struct MM3DepthFrame;
struct MM3DitBranch;

namespace mm3_ane_validation {
struct Runtime;
bool available();
std::shared_ptr<Runtime> load(const MM3Model & model, const std::string & directory);
void reset(Runtime & runtime);
void set_cancel_handler(Runtime & runtime, std::function<bool()> handler);
int64_t context_positions(const Runtime & runtime);
void lm_embedding(Runtime & runtime, int32_t conditional, int32_t unconditional, std::vector<float> & out);
void lm_feedback(Runtime & runtime, int32_t semantic, const int32_t * acoustic, float * out);
void lm_step(Runtime & runtime, const float * input, int64_t position, float * hidden, float * logits);
void depth(Runtime & runtime, const float * conditional, const float * unconditional, int32_t semantic,
           const int32_t * forced, MM3DepthFrame * out, void * rng, int top_k);
void condition(Runtime & runtime, const float * hidden, int64_t frames, int64_t latents, std::vector<float> & out);
void dit(Runtime & runtime, const float * latent, const float * condition, const std::vector<MM3DitBranch> & branches,
         float time, int64_t length);
void vocoder(Runtime & runtime, const std::vector<float> & latent, int64_t length, std::vector<float> & out);
}
#endif
