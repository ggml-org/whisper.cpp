#include "ggml.h"
#include "ggml-backend.h"
#include <iostream>
int main() {
    ggml_backend_load_all();
    auto * cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * metal=ggml_backend_dev_by_name("MTL0");
    if(!cpu||!metal)return 2;
    auto * ctx=ggml_init({ggml_tensor_overhead()*32,nullptr,true});
    for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0}) {
        auto * codes=ggml_new_tensor_2d(ctx,type,96,70);
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,96,7);
        auto * step=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,70);
        auto * minimum=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,70);
        auto * op=ggml_mul_mat_affine_v1(ctx,codes,x,step,minimum);
        if(op->src[2]!=step||op->src[3]!=minimum||op->ne[0]!=70||op->ne[1]!=7)return 3;
        if(ggml_backend_dev_supports_op(cpu,op)||!ggml_backend_dev_supports_op(metal,op))return 4;
        minimum->type=GGML_TYPE_F16;
        if(ggml_backend_dev_supports_op(metal,op))return 5;
        minimum->type=GGML_TYPE_F32;
        x->ne[2]=2;
        if(ggml_backend_dev_supports_op(metal,op))return 6;
        x->ne[2]=1;x->type=GGML_TYPE_F16;
        if(ggml_backend_dev_supports_op(metal,op))return 7;
    }
    ggml_free(ctx);std::cout<<"PASS: valid 4/8-bit shapes, CPU rejection, invalid metadata/input/batch rejection\n";
    return 0;
}
