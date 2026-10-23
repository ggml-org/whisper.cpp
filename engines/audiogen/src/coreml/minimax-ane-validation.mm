#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "minimax/ane-validation.h"
#include "minimax/mm3-model.h"
#include "minimax/mm3-depth-graph.h"
#include "minimax/mm3-dit-graph.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <limits>
#include <random>
#include <stdexcept>

namespace mm3_ane_validation {

bool available() {
    if (@available(macOS 15.0, *)) return true;
    return false;
}

using Shape = std::vector<int64_t>;
using TensorSpecs = std::map<std::string, Shape>;

struct ModelSpec {
    TensorSpecs inputs;
    TensorSpecs outputs;
};

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string error_text(NSError * error) {
    return error.localizedDescription.UTF8String ?: "Core ML error";
}

static MLMultiArray * make_array(NSArray<NSNumber *> * shape) {
    NSError * error = nil;
    auto * array = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:&error];
    require(array != nil, error_text(error));
    return array;
}

struct Array {
    MLMultiArray * value;
    float * data;
    int64_t shape[4];
    int64_t stride[4];

    explicit Array(MLMultiArray * array): value(array), data(static_cast<float *>(array.dataPointer)) {
        require(array != nil && array.dataType == MLMultiArrayDataTypeFloat32 && array.shape.count == 4,
                "Expected a four-dimensional F32 Core ML tensor");
        for(int index=0;index<4;++index) {
            shape[index]=array.shape[index].longLongValue;
            stride[index]=array.strides[index].longLongValue;
        }
    }
};

static float & cell(const Array & array,int64_t b,int64_t c,int64_t h,int64_t s) {
    return array.data[b*array.stride[0]+c*array.stride[1]+h*array.stride[2]+s*array.stride[3]];
}

static void zero(MLMultiArray * array) {
    const Array view(array);
    for (int64_t b=0; b<view.shape[0]; ++b)
        for (int64_t c=0; c<view.shape[1]; ++c)
            for (int64_t h=0; h<view.shape[2]; ++h)
                for (int64_t s=0; s<view.shape[3]; ++s)
                    cell(view,b,c,h,s)=0;
}

static bool matches_shape(NSArray<NSNumber *> * actual, const Shape & expected) {
    if (actual.count != expected.size()) return false;
    for (size_t index = 0; index < expected.size(); ++index) {
        if (actual[index].longLongValue != expected[index]) return false;
    }
    return true;
}

static void validate_descriptions(NSDictionary<NSString *, MLFeatureDescription *> * descriptions,
                                  const TensorSpecs & expected, const std::string & label) {
    require(descriptions.count == expected.size(), label + ": unexpected feature count");
    for (const auto & entry : expected) {
        auto * description = descriptions[@(entry.first.c_str())];
        require(description != nil && description.type == MLFeatureTypeMultiArray &&
                description.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat32 &&
                matches_shape(description.multiArrayConstraint.shape, entry.second),
                label + ": incompatible tensor " + entry.first);
    }
}

struct Model {
    MLModel * model = nil;
    id state = nil;
    NSMutableDictionary<NSString *, MLMultiArray *> * inputs = nil;
    MLPredictionOptions * options = nil;
    ModelSpec spec;
    std::string label;
    const std::function<bool()> * should_cancel;

    Model(const std::filesystem::path & path, MLComputeUnits compute_units, ModelSpec expected,
          const std::function<bool()> * cancelled)
        : spec(std::move(expected)), label(path.string()), should_cancel(cancelled) {
        if (@available(macOS 15.0, *)) {
            NSError * error = nil;
            auto * configuration = [[MLModelConfiguration alloc] init];
            configuration.computeUnits = compute_units;
            model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())]
                configuration:configuration error:&error];
            require(model != nil, label + ": " + error_text(error));
            validate_descriptions(model.modelDescription.inputDescriptionsByName, spec.inputs, label + " inputs");
            validate_descriptions(model.modelDescription.outputDescriptionsByName, spec.outputs, label + " outputs");
            inputs = [NSMutableDictionary dictionary];
            for (NSString * name in model.modelDescription.inputDescriptionsByName) {
                inputs[name] = make_array(model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint.shape);
            }
            auto * backings = [NSMutableDictionary dictionary];
            for (NSString * name in model.modelDescription.outputDescriptionsByName) {
                backings[name] = make_array(model.modelDescription.outputDescriptionsByName[name].multiArrayConstraint.shape);
            }
            options = [[MLPredictionOptions alloc] init];
            options.outputBackings = backings;
            reset();
        } else {
            throw std::runtime_error("MiniMax full Core ML requires macOS 15 or later");
        }
    }

    void check_cancelled() const {
        if (should_cancel && *should_cancel && (*should_cancel)()) {
            throw std::runtime_error("MiniMax Core ML generation cancelled");
        }
    }

    void reset() {
        check_cancelled();
        if (@available(macOS 15.0, *)) {
            if (model.modelDescription.stateDescriptionsByName.count) state = [model newState];
        }
    }

    id<MLFeatureProvider> predict(NSDictionary<NSString *, MLMultiArray *> * overrides = nil) {
        check_cancelled();
        if (@available(macOS 15.0, *)) {
            NSError * error = nil;
            NSMutableDictionary * values = [inputs mutableCopy];
            if (overrides) [values addEntriesFromDictionary:overrides];
            auto * provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
            require(provider != nil, label + ": " + error_text(error));
            auto result = state ? [model predictionFromFeatures:provider usingState:state options:options error:&error]
                                : [model predictionFromFeatures:provider options:options error:&error];
            require(result != nil, label + ": " + error_text(error));
            for (const auto & entry : spec.outputs) {
                auto * array = [result featureValueForName:@(entry.first.c_str())].multiArrayValue;
                require(array != nil && array.dataType == MLMultiArrayDataTypeFloat32 &&
                        matches_shape(array.shape, entry.second), label + ": incompatible output " + entry.first);
            }
            check_cancelled();
            return result;
        }
        throw std::runtime_error("MiniMax full Core ML requires macOS 15 or later");
    }
};

struct Runtime {
    MM3LmConfig lm;
    MM3SynthConfig synth;
    GGUFModel lm_file{};
    GGUFModel synth_file{};
    std::function<bool()> should_cancel;
    std::map<std::string, std::unique_ptr<Model>> models;
    int64_t capacity=0;
    int64_t depth_context=0;
    int64_t condition_frames=0;
    int64_t dit_latents=0;
    int64_t vocoder_latents=0;

    Runtime() { lm_file.fd=-1; synth_file.fd=-1; }

    ~Runtime() {
        gf_close(&lm_file);
        gf_close(&synth_file);
    }

    Model & get(const std::string & name) { return *models.at(name); }

    void row(const GGUFModel & file, const char * name, int64_t index, float * destination, int64_t width) {
        const auto id=gguf_find_tensor(file.gguf,name);
        require(id>=0,std::string("Missing tensor ")+name);
        auto * tensor=ggml_get_tensor(file.meta,name);
        require(tensor != nullptr && tensor->ne[0] == width && index >= 0 && index < tensor->ne[1],
                "Incompatible embedding shape or row outside the original vocabulary");
        const auto * source=gf_tensor_source(file,id)+index*tensor->nb[1];
        if(tensor->type==GGML_TYPE_F16) ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(source),destination,tensor->ne[0]);
        else if(tensor->type==GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float *>(source),tensor->ne[0],destination);
        else throw std::runtime_error("MiniMax Core ML expects original F16 or F32 embedding rows");
    }
};

static int64_t positive_integer(NSDictionary * manifest, NSString * key) {
    id value = manifest[key];
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
            std::string("MiniMax Core ML manifest requires integer ") + key.UTF8String);
    const auto number = [value longLongValue];
    require(number > 0 && [value doubleValue] == static_cast<double>(number),
            std::string("Invalid MiniMax Core ML manifest value for ") + key.UTF8String);
    return number;
}

static ModelSpec model_spec(const Runtime & runtime, const std::string & name) {
    const Shape lm_hidden = {1, runtime.lm.embedding_length, 1, 2};
    const Shape depth_hidden = {1, runtime.synth.depth.embedding_length, 1, 2};
    const Shape dit_hidden = {2, runtime.synth.dit.embedding_length, 1, runtime.dit_latents + 1};
    if (name == "lm-head") return {{{"hidden", lm_hidden}},
        {{"hidden_out", lm_hidden}, {"logits", {1, int64_t(runtime.lm.semantic_vocab_size) + 1, 1, 2}}}};
    if (name == "feedback") return {{{"embeddings", {1, runtime.lm.embedding_length, runtime.lm.num_codebooks, 1}}},
        {{"feedback", {1, runtime.lm.embedding_length, 1, 1}}}};
    if (name == "depth-input") return {{{"hidden", lm_hidden}, {"positional", depth_hidden}}, {{"hidden_out", depth_hidden}}};
    if (name.rfind("depth-head-", 0) == 0) return {{{"hidden", depth_hidden}},
        {{"hidden_out", depth_hidden}, {"logits", {1, runtime.synth.depth.audio_vocab_size, 1, 2}}}};
    if (name == "conditioner") return {
        {{"hidden", {1, runtime.synth.cond.hidden_dim, runtime.synth.cond.num_layers, runtime.condition_frames + 2}}},
        {{"condition", {1, runtime.synth.cond.out_dim, 1, runtime.condition_frames + 2}}}};
    if (name == "dit-input") return {
        {{"latent", {1, runtime.synth.dit.in_channels, 1, runtime.dit_latents}},
         {"condition", {1, runtime.synth.dit.condition_dim, 1, runtime.dit_latents}},
         {"fourier", {1, runtime.synth.dit.fourier_dim, 1, 1}}, {"gates", {2, 1, 1, 1}}}, {{"hidden_out", dit_hidden}}};
    if (name == "dit-output") return {{{"hidden", dit_hidden}},
        {{"velocity", {2, runtime.synth.dit.in_channels, 1, runtime.dit_latents}}}};
    if (name == "vocoder") return {
        {{"latent", {1, runtime.synth.voc.fold_channels, 1, runtime.vocoder_latents}},
         {"valid", {1, 1, 1, runtime.vocoder_latents}}},
        {{"waveform", {1, 1, 1, runtime.vocoder_latents * runtime.synth.voc.total_upsample}}}};
    if (name.rfind("lm-", 0) == 0) return {
        {{"hidden", lm_hidden}, {"selector", {1, runtime.capacity}}, {"mask", {1, runtime.capacity, 1, 1}},
         {"cosine", {1, 1, runtime.lm.key_length / 2, 1}}, {"sine", {1, 1, runtime.lm.key_length / 2, 1}}},
        {{"hidden_out", lm_hidden}}};
    if (name.rfind("depth-", 0) == 0) return {
        {{"hidden", depth_hidden}, {"selector", {1, runtime.depth_context}}, {"mask", {1, runtime.depth_context, 1, 1}}},
        {{"hidden_out", depth_hidden}}};
    if (name.rfind("dit-", 0) == 0) return {
        {{"hidden", dit_hidden}, {"mask", {1, runtime.dit_latents + 1, 1, 1}}}, {{"hidden_out", dit_hidden}}};
    throw std::runtime_error("Unknown MiniMax Core ML program " + name);
}

std::shared_ptr<Runtime> load(const MM3Model & model, const std::string & directory) {
    @autoreleasepool {
        require(available(), "MiniMax full Core ML requires macOS 15 or later");
        const auto start=std::chrono::steady_clock::now();
        auto result=std::make_shared<Runtime>();
        require(model.ane_validation_compute_units == "CPU_ONLY" ||
                model.ane_validation_compute_units == "CPU_AND_GPU" ||
                model.ane_validation_compute_units == "CPU_AND_NE" ||
                model.ane_validation_compute_units == "ALL", "Unsupported MiniMax Core ML compute units");
        auto compute_units = MLComputeUnitsCPUAndNeuralEngine;
        if (model.ane_validation_compute_units == "CPU_ONLY") {
            compute_units = MLComputeUnitsCPUOnly;
        } else if (model.ane_validation_compute_units == "CPU_AND_GPU") {
            compute_units = MLComputeUnitsCPUAndGPU;
        } else if (model.ane_validation_compute_units == "ALL") {
            compute_units = MLComputeUnitsAll;
        }
        fprintf(stderr,"[MM3-ANE] Allowed compute units: %s\n",model.ane_validation_compute_units.c_str());
        result->lm=model.lm_cfg;
        result->synth=model.synth_cfg;
        NSError * error=nil;
        auto * json=[NSData dataWithContentsOfFile:@((directory+"/manifest.json").c_str())];
        require(json != nil,"Missing MiniMax Core ML manifest");
        NSDictionary * manifest=[NSJSONSerialization JSONObjectWithData:json options:0 error:&error];
        require([manifest isKindOfClass:NSDictionary.class],"Invalid MiniMax Core ML manifest");
        require(positive_integer(manifest, @"format") == 1, "Unsupported MiniMax Core ML profile format");
        result->capacity = positive_integer(manifest, manifest[@"lm_profile_positions"] ? @"lm_profile_positions" : @"context");
        result->depth_context = positive_integer(manifest, @"depth_context");
        result->condition_frames = positive_integer(manifest, @"condition_frames");
        result->dit_latents = positive_integer(manifest, @"dit_latents");
        result->vocoder_latents = positive_integer(manifest, @"vocoder_latents");
        require(positive_integer(manifest, @"context") == model.lm_cfg.context_length,
                "Core ML metadata must preserve the original model context");
        require(positive_integer(manifest, @"lm_layers") == model.lm_cfg.block_count &&
                positive_integer(manifest, @"depth_layers") == model.synth_cfg.depth.block_count &&
                positive_integer(manifest, @"dit_layers") == model.synth_cfg.dit.block_count,
                "Core ML profile layer counts do not match the model");
        require(result->depth_context >= model.synth_cfg.depth.num_codebooks &&
                result->condition_frames <= std::numeric_limits<int64_t>::max() - 2 &&
                result->dit_latents < std::numeric_limits<int64_t>::max() &&
                result->vocoder_latents > 64 && model.synth_cfg.voc.total_upsample > 0 &&
                result->vocoder_latents <= std::numeric_limits<int64_t>::max() / model.synth_cfg.voc.total_upsample,
                "Invalid MiniMax Core ML profile dimensions");
        require(result->capacity>0 && result->capacity<=model.lm_cfg.context_length,"Invalid compiled MiniMax Core ML cache profile");
        require(gf_load(&result->lm_file,model.lm_file.path.c_str()),"Cannot map original LM embedding rows");
        require(gf_load(&result->synth_file,model.synth_file.path.c_str()),"Cannot map original synthesis embedding rows");
        std::vector<std::string> names={"lm-head","feedback","depth-input","conditioner","dit-input","dit-output","vocoder"};
        for (uint32_t index=0;index<model.lm_cfg.block_count;++index) names.push_back("lm-"+std::to_string(index));
        for (uint32_t index=0;index<model.synth_cfg.depth.block_count;++index) names.push_back("depth-"+std::to_string(index));
        for (uint32_t index=0;index+1<model.synth_cfg.depth.num_codebooks;++index) names.push_back("depth-head-"+std::to_string(index));
        for (uint32_t index=0;index<model.synth_cfg.dit.block_count;++index) names.push_back("dit-"+std::to_string(index));
        for (const auto & name:names) {
            fprintf(stderr,"[MM3-ANE] Loading %s\n",name.c_str());
            result->models.emplace(name,std::make_unique<Model>(std::filesystem::path(directory)/(name+".mlmodelc"), compute_units,
                model_spec(*result, name), &result->should_cancel));
        }
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        fprintf(stderr,"[MM3-ANE] All %zu programs resident, load %.3f s; generation has not started\n",names.size(),elapsed);
        return result;
    }
}

void reset(Runtime & runtime) {
    for (auto & pair : runtime.models) pair.second->reset();
}

void set_cancel_handler(Runtime & runtime, std::function<bool()> handler) {
    runtime.should_cancel = std::move(handler);
}

int64_t context_positions(const Runtime & runtime) { return runtime.capacity; }

void lm_embedding(Runtime & runtime,int32_t conditional,int32_t unconditional,std::vector<float> & out) {
    out.resize(2*runtime.lm.embedding_length);
    runtime.row(runtime.lm_file,"token_embd.weight",conditional,out.data(),runtime.lm.embedding_length);
    runtime.row(runtime.lm_file,"token_embd.weight",unconditional,out.data()+runtime.lm.embedding_length,runtime.lm.embedding_length);
}

void lm_feedback(Runtime & runtime,int32_t semantic,const int32_t * acoustic,float * out) {
    @autoreleasepool {
        auto & model=runtime.get("feedback");
        const Array input(model.inputs[@"embeddings"]);
        const auto H=runtime.lm.embedding_length;
        std::vector<float> row(H);
        for(uint32_t layer=0;layer<runtime.lm.num_codebooks;++layer) {
            if(layer==0) runtime.row(runtime.lm_file,"token_embd.weight",semantic,row.data(),H);
            else runtime.row(runtime.synth_file,"depth.audio_embd.weight",acoustic[layer-1],row.data(),H);
            for(uint32_t c=0;c<H;++c) cell(input,0,c,layer,0)=row[c];
        }
        const Array value([model.predict() featureValueForName:@"feedback"].multiArrayValue);
        for(uint32_t c=0;c<H;++c) out[c]=cell(value,0,c,0,0);
    }
}

static void attention_inputs(Model & model,int64_t position) {
    auto * selector=model.inputs[@"selector"];
    const Array mask(model.inputs[@"mask"]);
    auto * indices=static_cast<float *>(selector.dataPointer);
    for(int64_t index=0;index<selector.shape[1].longLongValue;++index) {
        indices[index*selector.strides[1].longLongValue]=index==position?1:0;
        cell(mask,0,index,0,0)=index<=position?0:-INFINITY;
    }
}

static void pack_pair(MLMultiArray * array,const float * input,int64_t hidden) {
    const Array view(array);
    for(int64_t channel=0;channel<hidden;++channel)
        for(int branch=0;branch<2;++branch) cell(view,0,channel,0,branch)=input[branch*hidden+channel];
}

static void unpack_pair(MLMultiArray * array,float * output,int64_t hidden) {
    const Array view(array);
    for(int64_t channel=0;channel<hidden;++channel)
        for(int branch=0;branch<2;++branch) output[branch*hidden+channel]=cell(view,0,channel,0,branch);
}

void lm_step(Runtime & runtime,const float * input,int64_t position,float * hidden,float * logits) {
    @autoreleasepool {
        require(position>=0 && position<runtime.capacity,"LM position outside original context");
        auto & first=runtime.get("lm-0");
        pack_pair(first.inputs[@"hidden"],input,runtime.lm.embedding_length);
        MLMultiArray * value=first.inputs[@"hidden"];
        for(uint32_t index=0;index<runtime.lm.block_count;++index) {
            auto & model=runtime.get("lm-"+std::to_string(index));
            attention_inputs(model,position);
            const Array cosine(model.inputs[@"cosine"]),sine(model.inputs[@"sine"]);
            for(uint32_t frequency=0;frequency<runtime.lm.key_length/2;++frequency) {
                const auto phase=position*std::pow(runtime.lm.rope_freq_base,-2.0*frequency/runtime.lm.key_length);
                cell(cosine,0,0,frequency,0)=std::cos(phase);
                cell(sine,0,0,frequency,0)=std::sin(phase);
            }
            value=[model.predict(@{@"hidden":value}) featureValueForName:@"hidden_out"].multiArrayValue;
        }
        auto features=runtime.get("lm-head").predict(@{@"hidden":value});
        unpack_pair([features featureValueForName:@"hidden_out"].multiArrayValue,hidden,runtime.lm.embedding_length);
        unpack_pair([features featureValueForName:@"logits"].multiArrayValue,logits,runtime.lm.semantic_vocab_size+1);
    }
}

void depth(Runtime & runtime,const float * conditional,const float * unconditional,int32_t semantic,
           const int32_t * forced,MM3DepthFrame * out,void * generator,int top_k) {
    @autoreleasepool {
        const auto start=std::chrono::steady_clock::now();
        const auto H=runtime.synth.depth.embedding_length;
        const auto V=runtime.synth.depth.audio_vocab_size;
        const auto count=runtime.synth.depth.num_codebooks-1;
        out->n_codes=count;
        out->hiddens.resize(count*H);
        out->logits_cond.resize(count*V);
        out->logits_uncond.resize(count*V);
        std::vector<float> row(2*H),position(H),logits(2*V),hidden(2*H),scratch;
        for(uint32_t token=0;token<=count;++token) {
            auto & entry=runtime.get("depth-input");
            if(token==0) {
                std::copy_n(conditional,H,row.data());
                std::copy_n(unconditional,H,row.data()+H);
            } else {
                if(token==1) runtime.row(runtime.lm_file,"token_embd.weight",semantic+runtime.lm.semantic_vocab_offset,row.data(),H);
                else runtime.row(runtime.synth_file,"depth.audio_embd.weight",out->codes[token-2]+(token-2)*V,row.data(),H);
                std::copy_n(row.data(),H,row.data()+H);
            }
            runtime.row(runtime.synth_file,"depth.pos_embd.weight",token,position.data(),H);
            pack_pair(entry.inputs[@"hidden"],row.data(),H);
            const Array positional(entry.inputs[@"positional"]);
            for(uint32_t channel=0;channel<H;++channel) for(int b=0;b<2;++b) cell(positional,0,channel,0,b)=position[channel];
            MLMultiArray * value=[entry.predict() featureValueForName:@"hidden_out"].multiArrayValue;
            for(uint32_t index=0;index<runtime.synth.depth.block_count;++index) {
                auto & block=runtime.get("depth-"+std::to_string(index));
                attention_inputs(block,token);
                value=[block.predict(@{@"hidden":value}) featureValueForName:@"hidden_out"].multiArrayValue;
            }
            if(token==0) continue;
            auto features=runtime.get("depth-head-"+std::to_string(token-1)).predict(@{@"hidden":value});
            unpack_pair([features featureValueForName:@"hidden_out"].multiArrayValue,hidden.data(),H);
            unpack_pair([features featureValueForName:@"logits"].multiArrayValue,logits.data(),V);
            std::copy_n(hidden.data(),H,out->hiddens.data()+(token-1)*H);
            std::copy_n(logits.data(),V,out->logits_cond.data()+(token-1)*V);
            std::copy_n(logits.data()+V,V,out->logits_uncond.data()+(token-1)*V);
            for(uint32_t index=0;index<V;++index) logits[index]=logits[V+index]+runtime.lm.ar_cfg_scale*(logits[index]-logits[V+index]);
            out->codes[token-1]=forced?forced[token-1]:generator?
                mm3_sample_top_k(logits.data(),V,top_k,*static_cast<std::mt19937_64 *>(generator),&scratch):
                static_cast<int32_t>(std::max_element(logits.begin(),logits.begin()+V)-logits.begin());
        }
        out->ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
}

void condition(Runtime & runtime,const float * hidden,int64_t frames,int64_t latents,std::vector<float> & out) {
    @autoreleasepool {
        auto & model=runtime.get("conditioner");
        auto * input=model.inputs[@"hidden"];
        const Array input_view(input);
        const auto H=runtime.synth.cond.hidden_dim;
        const auto layers=runtime.synth.cond.num_layers;
        const auto channels=runtime.synth.cond.out_dim;
        const auto window=runtime.condition_frames;
        std::vector<float> projected(frames*channels);
        for(int64_t first=0;first<frames;first+=window) {
            zero(input);
            for(int64_t index=0;index<window+2;++index) {
                const auto frame=first+index-1;
                if(frame<0 || frame>=frames) continue;
                for(uint32_t c=0;c<H;++c) for(uint32_t layer=0;layer<layers;++layer)
                    cell(input_view,0,c,layer,index)=hidden[(frame*layers+layer)*H+c];
            }
            const Array value([model.predict() featureValueForName:@"condition"].multiArrayValue);
            for(int64_t frame=first;frame<std::min(frames,first+window);++frame)
                for(uint32_t c=0;c<channels;++c) projected[frame*channels+c]=cell(value,0,c,0,frame-first+1);
        }
        out.resize(latents*channels);
        for(int64_t frame=0;frame<latents;++frame)
            std::copy_n(projected.data()+std::min(frame*frames/latents,frames-1)*channels,channels,out.data()+frame*channels);
    }
}

void dit(Runtime & runtime,const float * latent,const float * conditioning,const std::vector<MM3DitBranch> & branches,
         float time,int64_t length) {
    @autoreleasepool {
        require(length>0 && length<=runtime.dit_latents,"DiT validation window does not match the original window");
        auto & entry=runtime.get("dit-input");
        const auto C=runtime.synth.dit.in_channels;
        const auto H=runtime.synth.dit.condition_dim;
        zero(entry.inputs[@"latent"]);
        zero(entry.inputs[@"condition"]);
        const Array input_latent(entry.inputs[@"latent"]),input_condition(entry.inputs[@"condition"]);
        for(uint32_t channel=0;channel<C;++channel) for(int64_t index=0;index<length;++index)
            cell(input_latent,0,channel,0,index)=latent[channel*length+index];
        for(uint32_t channel=0;channel<H;++channel) for(int64_t index=0;index<length;++index)
            cell(input_condition,0,channel,0,index)=conditioning[index*H+channel];
        std::vector<float> frequency(runtime.synth.dit.fourier_dim/2);
        const auto id=gguf_find_tensor(runtime.synth_file.gguf,"dit.time_fourier.weight");
        auto * source=ggml_get_tensor(runtime.synth_file.meta,"dit.time_fourier.weight");
        require(id >= 0 && source != nullptr && source->type == GGML_TYPE_F32 &&
                ggml_nelements(source) == static_cast<int64_t>(frequency.size()), "Expected original F32 time frequencies");
        std::copy_n(reinterpret_cast<const float *>(gf_tensor_source(runtime.synth_file,id)),frequency.size(),frequency.data());
        const Array fourier(entry.inputs[@"fourier"]);
        for(size_t index=0;index<frequency.size();++index) {
            const double phase=2*3.14159265358979323846*time*frequency[index];
            cell(fourier,0,index,0,0)=std::cos(phase);
            cell(fourier,0,index+frequency.size(),0,0)=std::sin(phase);
        }
        for(size_t first=0;first<branches.size();first+=2) {
            const Array gates(entry.inputs[@"gates"]);
            for(int branch=0;branch<2;++branch) cell(gates,branch,0,0,0)=branches[std::min(first+branch,branches.size()-1)].gate;
            MLMultiArray * hidden=[entry.predict() featureValueForName:@"hidden_out"].multiArrayValue;
            for(uint32_t index=0;index<runtime.synth.dit.block_count;++index) {
                auto & block=runtime.get("dit-"+std::to_string(index));
                const Array mask(block.inputs[@"mask"]);
                for(int64_t position=0;position<=runtime.dit_latents;++position) cell(mask,0,position,0,0)=position<=length?0:-INFINITY;
                hidden=[block.predict(@{@"hidden":hidden}) featureValueForName:@"hidden_out"].multiArrayValue;
            }
            const Array value([runtime.get("dit-output").predict(@{@"hidden":hidden}) featureValueForName:@"velocity"].multiArrayValue);
            for(size_t branch=first;branch<std::min(first+2,branches.size());++branch)
                for(uint32_t c=0;c<C;++c) for(int64_t index=0;index<length;++index)
                    branches[branch].velocity[c*length+index]=cell(value,branch-first,c,0,index);
        }
    }
}

void vocoder(Runtime & runtime,const std::vector<float> & latent,int64_t length,std::vector<float> & out) {
    @autoreleasepool {
        const auto channels=runtime.synth.voc.fold_channels;
        const auto up=runtime.synth.voc.total_upsample;
        const auto chunk=runtime.vocoder_latents;
        const int64_t overlap=32;
        const int64_t core=chunk-2*overlap;
        require(length>0 && core>0,"Invalid vocoder validation dimensions");
        require(latent.size()==static_cast<size_t>(2*channels*length),"Invalid original vocoder latent shape");
        out.assign(2*length*up,0);
        auto & model=runtime.get("vocoder");
        for(int64_t first=0;first<length;first+=(length<=chunk?length:core)) {
            const auto end=length<=chunk?length:std::min(first+core,length);
            const auto start=length<=chunk?0:std::max(first-overlap,int64_t(0));
            const auto last=length<=chunk?length:std::min(end+overlap,length);
            const auto count=last-start;
            for(int branch=0;branch<2;++branch) {
                zero(model.inputs[@"latent"]);
                zero(model.inputs[@"valid"]);
                const Array input(model.inputs[@"latent"]),valid(model.inputs[@"valid"]);
                for(int64_t index=0;index<count;++index) cell(valid,0,0,0,index)=1;
                for(uint32_t channel=0;channel<channels;++channel) for(int64_t index=0;index<count;++index)
                    cell(input,0,channel,0,index)=latent[(branch*channels+channel)*length+start+index];
                const Array value([model.predict() featureValueForName:@"waveform"].multiArrayValue);
                for(int64_t index=first*up;index<end*up;++index) out[branch*length*up+index]=cell(value,0,0,0,index-start*up);
            }
        }
    }
}
}
