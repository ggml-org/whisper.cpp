#include "pocket/ggml_utils.h"
#include <iostream>

int main() {
    try {
        ggml_backend_load_all();
        auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("CPU backend missing");
        {
            tts_cpp::pocket::detail::PocketGraph g(backend);
            auto * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, 64, 32);
            auto * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 16);
            auto * output = ggml_mul_mat(g.ctx, a, b);
            const auto bytes = g.prepare(output, true, 4);
            size_t arena = 0;
            ggml_gallocr_reserve_n_size(g.allocator, g.graph, nullptr, nullptr, &arena);
            if (bytes <= arena) throw std::runtime_error("CPU scratch omitted");
            if (a->data || b->data || output->data) throw std::runtime_error("preflight allocated tensors");
            const auto repeated = tts_cpp::pocket::detail::price_cpu_graph(backend, g.allocator, g.graph, 4);
            if (repeated != bytes) throw std::runtime_error("preflight changed on repeat");
        }
        {
            using namespace tts_cpp::pocket::detail;
            OpenCLScratch pools;
            // Generic F32 GEMM may need two independently retained copies.
            PocketGraph g(backend);
            auto * a_storage = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 64);
            auto * a = ggml_view_2d(g.ctx, a_storage, 64, 32, a_storage->nb[1]*2, 0);
            auto * storage = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 32);
            auto * b = ggml_view_2d(g.ctx, storage, 64, 16, storage->nb[1]*2, 0);
            auto * output = ggml_mul_mat(g.ctx, a, b);
            g.prepare(output, true);
            measure_opencl_scratch(g.graph, pools);
            if (pools.src0 < ggml_nbytes(a) || pools.src1 < ggml_nbytes(b))
                throw std::runtime_error("OpenCL strided GEMM scratch omitted");
            // A later graph adds a distinct persistent transpose pool. It
            // cannot replace the earlier graph's src0/src1 pool maxima.
            PocketGraph image(backend);
            auto * w = ggml_new_tensor_2d(image.ctx, GGML_TYPE_F32, 64, 128);
            auto * x = ggml_new_tensor_2d(image.ctx, GGML_TYPE_F32, 64, 16);
            image.prepare(ggml_mul_mat(image.ctx, w, x), true);
            OpenCLScratch transposes;
            measure_opencl_scratch(image.graph, transposes);
            const auto bound = opencl_scratch_bound(pools, transposes);
            if (bound < opencl_init_bytes + pools.src0 + pools.src1 + ggml_nbytes(w))
                throw std::runtime_error("OpenCL independently retained pools omitted");
            if (a->data || b->data || output->data || w->data || x->data)
                throw std::runtime_error("OpenCL scratch pricing allocated model/graph tensors");
        }
        ggml_backend_free(backend);
        std::cout << "Pocket graph memory: metadata-only CPU scratch included\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
