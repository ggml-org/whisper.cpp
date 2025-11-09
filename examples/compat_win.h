#pragma once

#ifdef _WIN32

// ----------------------------------------------------
// Windows compatibility layer for POSIX-style functions
// ----------------------------------------------------

// Suppress CRT security warnings for _open/_read/_write etc.
#define _CRT_SECURE_NO_WARNINGS

#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// ----------------------------------------------------
// GCC/Clang attribute compatibility
// ----------------------------------------------------

// Make GCC-style __attribute__((...)) a no-op on MSVC
#ifndef __attribute__
#define __attribute__(x)
#endif

// Handle '__packed__' keyword used in Linux/GCC code
#ifndef __packed__
#define __packed__
#endif

// ----------------------------------------------------
// POSIX I/O compatibility
// ----------------------------------------------------
#define open  _open
#define read  _read
#define write _write
#define close _close
#define lseek _lseek
#define stat  _stat64i32
#define fstat _fstat64i32

// Common file flags mapping
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_RDWR
#define O_RDWR _O_RDWR
#endif
#ifndef O_CREAT
#define O_CREAT _O_CREAT
#endif
#ifndef O_TRUNC
#define O_TRUNC _O_TRUNC
#endif

// ----------------------------------------------------
// Memory mapping stubs (mmap, munmap, etc.)
// ----------------------------------------------------
// These emulate minimal mmap behavior using Win32 APIs
// to allow code that expects mmap() to compile and run
// (but with real file I/O behind the scenes).
// ----------------------------------------------------

#define PROT_READ  1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_FAILED ((void *)-1)

static inline void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    HANDLE hMap = CreateFileMappingA(
        (HANDLE)_get_osfhandle(fd),
        NULL,
        PAGE_READWRITE,
        0,
        (DWORD)length,
        NULL
    );
    if (!hMap) {
        fprintf(stderr, "CreateFileMappingA failed: %lu\n", GetLastError());
        return MAP_FAILED;
    }

    void *map = MapViewOfFile(
        hMap,
        FILE_MAP_READ | FILE_MAP_WRITE,
        0,
        offset,
        length
    );

    CloseHandle(hMap);

    if (!map) {
        fprintf(stderr, "MapViewOfFile failed: %lu\n", GetLastError());
        return MAP_FAILED;
    }

    return map;
}

static inline int munmap(void *addr, size_t length) {
    if (!UnmapViewOfFile(addr)) {
        fprintf(stderr, "UnmapViewOfFile failed: %lu\n", GetLastError());
        return -1;
    }
    return 0;
}

// ----------------------------------------------------
// POSIX type aliases and missing symbols
// ----------------------------------------------------
typedef int mode_t;
typedef intptr_t ssize_t;

#endif // _WIN32
