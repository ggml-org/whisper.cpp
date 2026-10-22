#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "packed_affine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Args { uint32_t M,N,K,bits; };
using Clock=std::chrono::steady_clock;
static NSData * bytes(NSString * path) {
    NSData * d=[NSData dataWithContentsOfFile:path];
    if(!d.length) throw std::runtime_error("missing input: "+std::string(path.UTF8String));
    return d;
}
static void check(bool b,const char * message) {if(!b)throw std::runtime_error(message);}
static double median(std::vector<double> x) {std::sort(x.begin(),x.end());return x[x.size()/2];}
static NSDictionary * compare(const float * a,const float * b,size_t count) {
    double error=0,den=0,maxerr=0;
    for(size_t i=0;i<count;++i) {
        check(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite matrix output");
        const double delta=double(a[i])-b[i];error+=delta*delta;den+=double(b[i])*b[i];maxerr=std::max(maxerr,std::abs(delta));
    }
    const double rel=std::sqrt(error/std::max(den,1e-30));
    check(rel<0.001,"relative L2 exceeds frozen 0.001 probe tolerance");
    return @{ @"relative_l2":@(rel), @"max_abs":@(maxerr), @"elements":@(count) };
}
int main(int argc,char ** argv) {
    @autoreleasepool { try {
        if(argc!=4)throw std::runtime_error("usage: bench-affine-metal CASES_JSON KERNEL_METAL OUTPUT_JSON");
        NSDictionary * manifest=[NSJSONSerialization JSONObjectWithData:bytes(@(argv[1])) options:0 error:nil];
        id<MTLDevice> metal=MTLCreateSystemDefaultDevice();check(metal!=nil,"Metal required");
        id<MTLCommandQueue> queue=[metal newCommandQueue];
        MTLCompileOptions * options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        NSError * error=nil;
        id<MTLLibrary> lib=[metal newLibraryWithSource:[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error] options:options error:&error];
        if(!lib)throw std::runtime_error(error.description.UTF8String);
        id<MTLComputePipelineState> fused=[metal newComputePipelineStateWithFunction:[lib newFunctionWithName:@"affine_mm"] error:&error];
        id<MTLComputePipelineState> restore=[metal newComputePipelineStateWithFunction:[lib newFunctionWithName:@"affine_restore"] error:&error];
        if(!fused||!restore)throw std::runtime_error(error.description.UTF8String);
        ggml_backend_load_all();auto * dev=ggml_backend_dev_by_name("MTL0");check(dev!=nullptr,"MTL0 required");
        auto backend=ggml_backend_dev_init(dev,nullptr);check(backend!=nullptr,"Metal backend init failed");
        NSMutableArray * results=[NSMutableArray new];
        for(NSDictionary * c in manifest[@"cases"]) { @autoreleasepool {
            Args a={ [c[@"M"] unsignedIntValue],[c[@"N"] unsignedIntValue],[c[@"K"] unsignedIntValue],[c[@"bits"] unsignedIntValue] };
            check(a.K%32==0 && (a.bits==4||a.bits==8),"unsupported case");
            NSString * folder=c[@"folder"],* path=c[@"path"];
            NSData * codeData=bytes([folder stringByAppendingPathComponent:@"codes.bin"]);
            NSData * stepData=bytes([folder stringByAppendingPathComponent:@"step.bin"]);
            NSData * minData=bytes([folder stringByAppendingPathComponent:@"minimum.bin"]);
            NSData * denseData=bytes([folder stringByAppendingPathComponent:@"dense.bin"]);
            NSData * xData=bytes([path stringByAppendingPathComponent:@"x.bin"]);
            NSData * cpuData=bytes([path stringByAppendingPathComponent:@"cpu.bin"]);
            const size_t count=size_t(a.M)*a.N;
            check(denseData.length==size_t(a.N)*a.K*2 && xData.length==size_t(a.M)*a.K*4 && cpuData.length==count*4,"input shape mismatch");
            id<MTLBuffer> codes=[metal newBufferWithBytes:codeData.bytes length:codeData.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> steps=[metal newBufferWithBytes:stepData.bytes length:stepData.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> minima=[metal newBufferWithBytes:minData.bytes length:minData.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> x=[metal newBufferWithBytes:xData.bytes length:xData.length options:MTLResourceStorageModeShared];
            id<MTLBuffer> output=[metal newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> restored=[metal newBufferWithLength:denseData.length options:MTLResourceStorageModeShared];
            auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:restore];
            [enc setBuffer:codes offset:0 atIndex:0];[enc setBuffer:steps offset:0 atIndex:1];[enc setBuffer:minima offset:0 atIndex:2];[enc setBuffer:restored offset:0 atIndex:3];[enc setBytes:&a length:sizeof(a) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(size_t(a.N)*a.K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
            check(cb.status==MTLCommandBufferStatusCompleted,"restore GPU failure");
            check(std::memcmp(restored.contents,denseData.bytes,denseData.length)==0,"recipe reconstruction differs bitwise");
            ggml_init_params wp={ggml_tensor_overhead()*16,nullptr,true};auto * weights=ggml_init(wp);
            auto * wc=ggml_new_tensor_2d(weights,a.bits==4?GGML_TYPE_Q4_0:GGML_TYPE_Q8_0,a.K,a.N);
            auto * wd=ggml_new_tensor_2d(weights,GGML_TYPE_F16,a.K,a.N);
            auto * ws=ggml_new_tensor_1d(weights,GGML_TYPE_F32,a.N);auto * wm=ggml_new_tensor_1d(weights,GGML_TYPE_F32,a.N);
            auto * xx=ggml_new_tensor_2d(weights,GGML_TYPE_F32,a.K,a.M);
            auto wb=ggml_backend_alloc_ctx_tensors(weights,backend);check(wb!=nullptr,"weight allocation failed");
            for(auto p:{std::make_pair(wc,codeData),std::make_pair(wd,denseData),std::make_pair(ws,stepData),std::make_pair(wm,minData),std::make_pair(xx,xData)}){
                check(ggml_nbytes(p.first)==p.second.length,"GGML byte size mismatch");ggml_backend_tensor_set(p.first,p.second.bytes,0,p.second.length);
            }
            ggml_context * contexts[3];ggml_cgraph * graphs[3];ggml_tensor * outputs[3];ggml_gallocr_t allocs[3];
            for(int kind=0;kind<3;++kind){
                ggml_init_params gp={ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true};contexts[kind]=ggml_init(gp);graphs[kind]=ggml_new_graph_custom(contexts[kind],64,false);
                if(kind==0)outputs[kind]=ggml_mul_mat(contexts[kind],wd,xx);
                else {parakeet::PackedAffineWeights map{{wc,{wm,ws,a.bits==4?8:128}}};parakeet::PackedAffineGraphScope scope(contexts[kind],graphs[kind],map,kind==2);outputs[kind]=parakeet::affine_mul_mat(contexts[kind],wc,xx);}
                ggml_set_output(outputs[kind]);ggml_build_forward_expand(graphs[kind],outputs[kind]);
                for(int n=0;n<ggml_graph_n_nodes(graphs[kind]);++n)check(ggml_backend_supports_op(backend,ggml_graph_node(graphs[kind],n)),"CPU fallback prohibited");
                allocs[kind]=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));check(ggml_gallocr_alloc_graph(allocs[kind],graphs[kind]),"graph alloc failed");
            }
            auto execute=[&](int kind)->std::pair<double,double>{
                auto start=Clock::now();double gpu=0;
                if(kind<3){check(ggml_backend_graph_compute(backend,graphs[kind])==GGML_STATUS_SUCCESS,"GGML compute failed");ggml_backend_synchronize(backend);}
                else {
                    auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:fused];
                    [e setBuffer:codes offset:0 atIndex:0];[e setBuffer:steps offset:0 atIndex:1];[e setBuffer:minima offset:0 atIndex:2];[e setBuffer:x offset:0 atIndex:3];[e setBuffer:output offset:0 atIndex:4];[e setBytes:&a length:sizeof(a) atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake((a.M+31)/32,(a.N+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];
                    check(command.status==MTLCommandBufferStatusCompleted,"fused GPU failure");gpu=(command.GPUEndTime-command.GPUStartTime)*1000;
                }
                return {std::chrono::duration<double,std::milli>(Clock::now()-start).count(),gpu};
            };
            for(int i=0;i<5;++i)for(int kind=0;kind<4;++kind)execute(kind);
            std::vector<double> times[4],gpu;
            for(int rep=0;rep<20;++rep)for(int j=0;j<4;++j){int kind=(rep+j)%4;auto t=execute(kind);times[kind].push_back(t.first);if(kind==3)gpu.push_back(t.second);}
            std::vector<float> reference(count),old(count),native(count);ggml_backend_tensor_get(outputs[2],native.data(),0,count*4);ggml_backend_tensor_get(outputs[0],reference.data(),0,count*4);ggml_backend_tensor_get(outputs[1],old.data(),0,count*4);
            NSDictionary * parity=compare(native.data(),reference.data(),count);
            NSDictionary * standaloneParity=compare((const float *)output.contents,reference.data(),count);
            NSDictionary * cpuParity=compare(native.data(),(const float *)cpuData.bytes,count);
            NSDictionary * oldParity=compare(old.data(),reference.data(),count);
            NSMutableArray * samples=[NSMutableArray new];for(int kind=0;kind<4;++kind){NSMutableArray * v=[NSMutableArray new];for(double t:times[kind])[v addObject:@(t)];[samples addObject:v];}
            NSMutableArray * gpuSamples=[NSMutableArray new];for(double t:gpu)[gpuSamples addObject:@(t)];
            NSMutableDictionary * record=[c mutableCopy];record[@"dense_ms"]=@(median(times[0]));record[@"current_packed_ms"]=@(median(times[1]));record[@"fused_ms"]=@(median(times[2]));record[@"standalone_fused_ms"]=@(median(times[3]));record[@"standalone_vs_dense"]=standaloneParity;record[@"fused_gpu_ms"]=@(median(gpu));record[@"wall_samples_dense_packed_native_fused_standalone"]=samples;record[@"fused_gpu_samples"]=gpuSamples;record[@"fused_vs_dense"]=parity;record[@"fused_vs_cpu"]=cpuParity;record[@"old_vs_dense"]=oldParity;record[@"restored_weight_bitwise_exact"]=@YES;
            [[NSData dataWithBytes:native.data() length:count*4] writeToFile:[path stringByAppendingPathComponent:@"fused.f32"] atomically:YES];
            [results addObject:record];std::cout<<[c[@"name"] UTF8String]<<" bits="<<a.bits<<" M="<<a.M<<" dense="<<median(times[0])<<" packed="<<median(times[1])<<" fused="<<median(times[2])<<" ms"<<std::endl;
            for(int kind=0;kind<3;++kind){ggml_gallocr_free(allocs[kind]);ggml_free(contexts[kind]);}ggml_backend_buffer_free(wb);ggml_free(weights);
        }}
        NSDictionary * report=@{@"device":metal.name,@"repeats":@20,@"warmups":@5,@"results":results,@"scope":@"Real native GGUF affine carriers with synthetic FP16-representable activations. Rotating resident primitive wall times include submit/sync; GPU timestamps separate. Dense/native reconstruction use pinned GGML MTL0; native fused graph uses the new Metal-only GGML op; standalone fused uses one Metal dispatch. No ASR/whole-call conclusion. Exact weight restoration and relative L2 <0.001 versus dense GGML and CPU reference required."};
        check([[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error] writeToFile:@(argv[3]) atomically:YES],"write report failed");
        ggml_backend_free(backend);return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<std::endl;return 1;}}
}
