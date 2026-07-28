#pragma once

// Cross-platform read-only memory-mapped file for the ACE-Step GGUF loaders.
// POSIX uses mmap(); Windows uses CreateFileMapping/MapViewOfFile. Header-only
// so both dit_gguf.cpp and vae_gguf.cpp share one implementation without a
// platform #ifdef sprinkled through each loader.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
// Without NOMINMAX, <windows.h> defines min/max as function-like macros and any
// later std::min(a, b) / std::max(a, b) in a TU that includes this header fails
// to compile under MSVC (C2589: '(' illegal token on right side of '::').
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace tts_cpp::acestep {

struct MappedFile {
    const uint8_t * data = nullptr;
    size_t          size = 0;
#ifdef _WIN32
    void * file_handle = nullptr;  // HANDLE
    void * map_handle  = nullptr;  // HANDLE
#else
    int fd = -1;
#endif
};

// Map `path` read-only. `tag` prefixes error logs (e.g. "acestep-dit").
inline bool mapped_file_open(MappedFile & m, const std::string & path, const char * tag) {
#ifdef _WIN32
    HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[%s] cannot open %s\n", tag, path.c_str());
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) {
        fprintf(stderr, "[%s] GetFileSizeEx failed\n", tag);
        CloseHandle(hf);
        return false;
    }
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) {
        fprintf(stderr, "[%s] CreateFileMapping failed\n", tag);
        CloseHandle(hf);
        return false;
    }
    void * base = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        fprintf(stderr, "[%s] MapViewOfFile failed\n", tag);
        CloseHandle(hm);
        CloseHandle(hf);
        return false;
    }
    m.file_handle = hf;
    m.map_handle  = hm;
    m.data        = (const uint8_t *) base;
    m.size        = (size_t) sz.QuadPart;
    return true;
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[%s] cannot open %s\n", tag, path.c_str());
        return false;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        fprintf(stderr, "[%s] fstat failed\n", tag);
        close(fd);
        return false;
    }
    void * base = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[%s] mmap failed\n", tag);
        close(fd);
        return false;
    }
    m.fd   = fd;
    m.data = (const uint8_t *) base;
    m.size = (size_t) sb.st_size;
    return true;
#endif
}

inline void mapped_file_close(MappedFile & m) {
#ifdef _WIN32
    if (m.data) UnmapViewOfFile((void *) m.data);
    if (m.map_handle) CloseHandle((HANDLE) m.map_handle);
    if (m.file_handle) CloseHandle((HANDLE) m.file_handle);
#else
    if (m.data) munmap((void *) m.data, m.size);
    if (m.fd >= 0) close(m.fd);
#endif
    m = {};
}

}  // namespace tts_cpp::acestep
