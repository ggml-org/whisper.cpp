// Standalone fake public backend API: no backend drivers or model payloads.
#include "whisper-fit-memory.h"

#include <cstdio>
#include <cstring>
#include <limits>

struct ggml_backend_reg { bool has_query = true; };
struct ggml_backend_device { ggml_backend_dev_props props = {}; ggml_backend_reg_t reg = nullptr; };
struct ggml_backend { ggml_backend_dev_t dev = nullptr; };

namespace {
ggml_backend_device cpu;
bool cpu_available = true;
int host_queries = 0;
bool query_ok = true;
ggml_backend_hexagon_runtime_info_v1 retained = {}, future = {};
}

void ggml_backend_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) { *props = dev->props; }
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_CPU && cpu_available ? &cpu : nullptr;
}
void ggml_backend_dev_memory(ggml_backend_dev_t dev, size_t * free_bytes, size_t * total_bytes) {
    ++host_queries;
    *free_bytes = dev->props.memory_free;
    *total_bytes = dev->props.memory_total;
}
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) { return backend->dev; }
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t dev) { return dev->reg; }

static bool runtime_query(ggml_backend_t, uint64_t nodes, ggml_backend_hexagon_runtime_info_v1 * out, size_t size) {
    if (!query_ok || size != sizeof(*out) || (nodes != 0 && nodes != 4096)) {
        return false;
    }
    *out = nodes == 0 ? retained : future;
    return true;
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return reg->has_query && std::strcmp(name, "ggml_backend_hexagon_get_runtime_info_v1") == 0 ?
        reinterpret_cast<void *>(runtime_query) : nullptr;
}

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    cpu.props.type = GGML_BACKEND_DEVICE_TYPE_CPU;
    cpu.props.memory_free = 800;
    cpu.props.memory_total = 1000;
    auto info = whisper_fit_query_memory(&cpu);
    CHECK(info.shares_host && info.free_bytes == 800 && info.total_bytes == 1000 && host_queries == 0);

    ggml_backend_reg reg;
    ggml_backend_device dev;
    dev.reg = &reg;
    dev.props.type = GGML_BACKEND_DEVICE_TYPE_GPU;
    dev.props.memory_unified = true;
    info = whisper_fit_query_memory(&dev);
    CHECK(info.shares_host && info.free_bytes == 800 && info.total_bytes == 1000 && host_queries == 1);
    // Zero free with a known capacity is actual pressure, not unknown VRAM.
    dev.props.memory_total = 500;
    info = whisper_fit_query_memory(&dev);
    CHECK(info.shares_host && info.free_bytes == 0 && info.total_bytes == 500 && host_queries == 1);
    dev.props.memory_free = 300;
    info = whisper_fit_query_memory(&dev);
    CHECK(info.free_bytes == 300 && info.total_bytes == 500 && host_queries == 1);
    dev.props.memory_total = dev.props.memory_free = 0;
    dev.props.memory_unified = false;
    info = whisper_fit_query_memory(&dev);
    CHECK(!info.shares_host && info.free_bytes == 0 && info.total_bytes == 0 && host_queries == 1);
    dev.props.type = GGML_BACKEND_DEVICE_TYPE_IGPU;
    info = whisper_fit_query_memory(&dev);
    CHECK(info.shares_host && info.free_bytes == 800 && host_queries == 2);
    cpu_available = false;
    info = whisper_fit_query_memory(&dev);
    CHECK(info.shares_host && info.total_bytes == 0 && host_queries == 2);

    ggml_backend backend;
    backend.dev = &dev;
    retained.version = future.version = GGML_HEXAGON_RUNTIME_INFO_VERSION;
    retained.session_count = future.session_count = 2;
    retained.host_bytes_bound = 100;
    future.host_bytes_bound = 150;
    retained.shared_bytes = future.shared_bytes = 900;
    uint64_t bytes = 0;
    CHECK(whisper_fit_query_backend_overhead(&backend, 4096, bytes) && bytes == 50);
    // Retained queues/metadata are already reflected in available RAM.
    future = retained;
    CHECK(whisper_fit_query_backend_overhead(&backend, 4096, bytes) && bytes == 0);
    future.shared_bytes += 10;
    CHECK(whisper_fit_query_backend_overhead(&backend, 4096, bytes) && bytes == 10);
    future.host_bytes_bound = 99;
    CHECK(!whisper_fit_query_backend_overhead(&backend, 4096, bytes));
    future = retained;
    future.session_count = 1;
    CHECK(!whisper_fit_query_backend_overhead(&backend, 4096, bytes));
    future = retained;
    future.version = 999;
    CHECK(!whisper_fit_query_backend_overhead(&backend, 4096, bytes));
    future = retained;
    query_ok = false;
    CHECK(!whisper_fit_query_backend_overhead(&backend, 4096, bytes));
    query_ok = true;
    retained.host_bytes_bound = retained.shared_bytes = 0;
    future.host_bytes_bound = future.shared_bytes = std::numeric_limits<uint64_t>::max();
    CHECK(whisper_fit_query_backend_overhead(&backend, 4096, bytes) && bytes == std::numeric_limits<uint64_t>::max());
    reg.has_query = false;
    CHECK(whisper_fit_query_backend_overhead(&backend, 4096, bytes) && bytes == 0);
    std::puts("test-whisper-fit-memory: all checks passed");
    return 0;
}
