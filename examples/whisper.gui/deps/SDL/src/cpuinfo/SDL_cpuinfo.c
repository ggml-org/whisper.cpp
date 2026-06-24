/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#ifdef TEST_MAIN
#include "SDL_config.h"
#else
#include "../SDL_internal.h"
#endif

#if defined(__WIN32__) || defined(__WINRT__) || defined(__GDK__)
#include "../core/windows/SDL_windows.h"
#endif
#if defined(__OS2__)
#undef HAVE_SYSCTLBYNAME
#define INCL_DOS
#include <os2.h>
#ifndef QSV_NUMPROCESSORS
#define QSV_NUMPROCESSORS 26
#endif
#endif

/* CPU feature detection for SDL */

#include "SDL_cpuinfo.h"
#include "SDL_assert.h"

#ifdef HAVE_SYSCONF
#include <unistd.h>
#endif
#ifdef HAVE_SYSCTLBYNAME
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if defined(__MACOSX__) && (defined(__ppc__) || defined(__ppc64__))
#include <sys/sysctl.h>         /* For AltiVec check */
#elif defined(__OpenBSD__) && defined(__powerpc__)
#include <sys/types.h>
#include <sys/sysctl.h> /* For AltiVec check */
#include <machine/cpu.h>
#elif defined(__FreeBSD__) && defined(__powerpc__)
#include <machine/cpu.h>
#include <sys/auxv.h>
#elif defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP)
#include <signal.h>
#include <setjmp.h>
#endif

#if defined(__QNXNTO__)
#include <sys/syspage.h>
#endif

#if (defined(__LINUX__) || defined(__ANDROID__)) && defined(__arm__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>

/*#include <asm/hwcap.h>*/
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef AT_PLATFORM
#define AT_PLATFORM 15
#endif
#ifndef HWCAP_NEON
#define HWCAP_NEON (1 << 12)
#endif
#endif

#if defined(__ANDROID__) && defined(__arm__) && !defined(HAVE_GETAUXVAL)
#include <cpu-features.h>
#endif

#if defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO)
#include <sys/auxv.h>
#endif

#ifdef __RISCOS__
#include <kernel.h>
#include <swis.h>
#endif

#ifdef __PS2__
#include <kernel.h>
#endif

#define CPU_HAS_RDTSC    (1 << 0)
#define CPU_HAS_ALTIVEC  (1 << 1)
#define CPU_HAS_MMX      (1 << 2)
#define CPU_HAS_3DNOW    (1 << 3)
#define CPU_HAS_SSE      (1 << 4)
#define CPU_HAS_SSE2     (1 << 5)
#define CPU_HAS_SSE3     (1 << 6)
#define CPU_HAS_SSE41    (1 << 7)
#define CPU_HAS_SSE42    (1 << 8)
#define CPU_HAS_AVX      (1 << 9)
#define CPU_HAS_AVX2     (1 << 10)
#define CPU_HAS_NEON     (1 << 11)
#define CPU_HAS_AVX512F  (1 << 12)
#define CPU_HAS_ARM_SIMD (1 << 13)
#define CPU_HAS_LSX      (1 << 14)
#define CPU_HAS_LASX     (1 << 15)

#define CPU_CFG2      0x2
#define CPU_CFG2_LSX  (1 << 6)
#define CPU_CFG2_LASX (1 << 7)

#if defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP) && !defined(__MACOSX__) && !defined(__OpenBSD__) && !defined(__FreeBSD__)
/* This is the brute force way of detecting instruction sets...
   the idea is borrowed from the libmpeg2 library - thanks!
 */
static jmp_buf jmpbuf;
static void illegal_instruction(int sig)
{
    longjmp(jmpbuf, 1);
}
#endif /* HAVE_SETJMP */

static int CPU_haveCPUID(void)
{
    int has_CPUID = 0;

/* *INDENT-OFF* */ /* clang-format off */
#ifndef SDL_CPUINFO_DISABLED
#if (defined(__GNUC__) || defined(__llvm__)) && defined(__i386__)
    __asm__ (
"        pushfl                      # Get original EFLAGS             \n"
"        popl    %%eax                                                 \n"
"        movl    %%eax,%%ecx                                           \n"
"        xorl    $0x200000,%%eax     # Flip ID bit in EFLAGS           \n"
"        pushl   %%eax               # Save new EFLAGS value on stack  \n"
"        popfl                       # Replace current EFLAGS value    \n"
"        pushfl                      # Get new EFLAGS                  \n"
"        popl    %%eax               # Store new EFLAGS in EAX         \n"
"        xorl    %%ecx,%%eax         # Can not toggle ID bit,          \n"
"        jz      1f                  # Processor=80486                 \n"
"        movl    $1,%0               # We have CPUID support           \n"
"1:                                                                    \n"
    : "=m" (has_CPUID)
    :
    : "%eax", "%ecx"
    );
#elif (defined(__GNUC__) || defined(__llvm__)) && defined(__x86_64__)
/* Technically, if this is being compiled under __x86_64__ then it has
   CPUid by definition.  But it's nice to be able to prove it.  :)      */
    __asm__ (
"        pushfq                      # Get original EFLAGS             \n"
"        popq    %%rax                                                 \n"
"        movq    %%rax,%%rcx                                           \n"
"        xorl    $0x200000,%%eax     # Flip ID bit in EFLAGS           \n"
"        pushq   %%rax               # Save new EFLAGS value on stack  \n"
"        popfq                       # Replace current EFLAGS value    \n"
"        pushfq                      # Get new EFLAGS                  \n"
"        popq    %%rax               # Store new EFLAGS in EAX         \n"
"        xorl    %%ecx,%%eax         # Can not toggle ID bit,          \n"
"        jz      1f                  # Processor=80486                 \n"
"        movl    $1,%0               # We have CPUID support           \n"
"1:                                                                    \n"
    : "=m" (has_CPUID)
    :
    : "%rax", "%rcx"
    );
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
    __asm {
        pushfd                      ; Get original EFLAGS
        pop     eax
        mov     ecx, eax
        xor     eax, 200000h        ; Flip ID bit in EFLAGS
        push    eax                 ; Save new EFLAGS value on stack
        popfd                       ; Replace current EFLAGS value
        pushfd                      ; Get new EFLAGS
        pop     eax                 ; Store new EFLAGS in EAX
        xor     eax, ecx            ; Can not toggle ID bit,
        jz      done                ; Processor=80486
        mov     has_CPUID,1         ; We have CPUID support
done:
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    has_CPUID = 1;
#elif defined(__sun) && defined(__i386)
    __asm (
"       pushfl                 \n"
"       popl    %eax           \n"
"       movl    %eax,%ecx      \n"
"       xorl    $0x200000,%eax \n"
"       pushl   %eax           \n"
"       popfl                  \n"
"       pushfl                 \n"
"       popl    %eax           \n"
"       xorl    %ecx,%eax      \n"
"       jz      1f             \n"
"       movl    $1,-8(%ebp)    \n"
"1:                            \n"
    );
#elif defined(__sun) && defined(__amd64)
    __asm (
"       pushfq                 \n"
"       popq    %rax           \n"
"       movq    %rax,%rcx      \n"
"       xorl    $0x200000,%eax \n"
"       pushq   %rax           \n"
"       popfq                  \n"
"       pushfq                 \n"
"       popq    %rax           \n"
"       xorl    %ecx,%eax      \n"
"       jz      1f             \n"
"       movl    $1,-8(%rbp)    \n"
"1:                            \n"
    );
#endif
#endif
/* *INDENT-ON* */ /* clang-format on */
    return has_CPUID;
}

#if (defined(__GNUC__) || defined(__llvm__)) && defined(__i386__)
#define cpuid(func, a, b, c, d)              \
    __asm__ __volatile__(                    \
        "        pushl %%ebx        \n"      \
        "        xorl %%ecx,%%ecx   \n"      \
        "        cpuid              \n"      \
        "        movl %%ebx, %%esi  \n"      \
        "        popl %%ebx         \n"      \
        : "=a"(a), "=S"(b), "=c"(c), "=d"(d) \
        : "a"(func))
#elif (defined(__GNUC__) || defined(__llvm__)) && defined(__x86_64__)
#define cpuid(func, a, b, c, d)              \
    __asm__ __volatile__(                    \
        "        pushq %%rbx        \n"      \
        "        xorq %%rcx,%%rcx   \n"      \
        "        cpuid              \n"      \
        "        movq %%rbx, %%rsi  \n"      \
        "        popq %%rbx         \n"      \
        : "=a"(a), "=S"(b), "=c"(c), "=d"(d) \
        : "a"(func))
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
#define cpuid(func, a, b, c, d) \
    __asm { \
        __asm mov eax, func \
        __asm xor ecx, ecx \
        __asm cpuid \
        __asm mov a, eax \
        __asm mov b, ebx \
        __asm mov c, ecx \
        __asm mov d, edx                   \
    }
#elif (defined(_MSC_VER) && defined(_M_X64))
/* Use __cpuidex instead of __cpuid because ICL does not clear ecx register */
#define cpuid(func, a, b, c, d)      \
    {                                \
        int CPUInfo[4];              \
        __cpuidex(CPUInfo, func, 0); \
        a = CPUInfo[0];              \
        b = CPUInfo[1];              \
        c = CPUInfo[2];              \
        d = CPUInfo[3];              \
    }
#else
#define cpuid(func, a, b, c, d) \
    do {                        \
        a = b = c = d = 0;      \
        (void)a;                \
        (void)b;                \
        (void)c;                \
        (void)d;                \
    } while (0)
#endif

static int CPU_CPUIDFeatures[4];
static int CPU_CPUIDMaxFunction = 0;
static SDL_bool CPU_OSSavesYMM = SDL_FALSE;
static SDL_bool CPU_OSSavesZMM = SDL_FALSE;

static void CPU_calcCPUIDFeatures(void)
{
    static SDL_bool checked = SDL_FALSE;
    if (!checked) {
        checked = SDL_TRUE;
        if (CPU_haveCPUID()) {
            int a, b, c, d;
            cpuid(0, a, b, c, d);
            CPU_CPUIDMaxFunction = a;
            if (CPU_CPUIDMaxFunction >= 1) {
                cpuid(1, a, b, c, d);
                CPU_CPUIDFeatures[0] = a;
                CPU_CPUIDFeatures[1] = b;
                CPU_CPUIDFeatures[2] = c;
                CPU_CPUIDFeatures[3] = d;

                /* Check to make sure we can call xgetbv */
                if (c & 0x08000000) {
                    /* Call xgetbv to see if YMM (etc) register state is saved */
#if (defined(__GNUC__) || defined(__llvm__)) && (defined(__i386__) || defined(__x86_64__))
                    __asm__(".byte 0x0f, 0x01, 0xd0"
                            : "=a"(a)
                            : "c"(0)
                            : "%edx");
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)) && (_MSC_FULL_VER >= 160040219) /* VS2010 SP1 */
                    a = (int)_xgetbv(0);
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
                    __asm
                        {
                        xor ecx, ecx
                        _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0
                        mov a, eax
                        }
#endif
                    CPU_OSSavesYMM = ((a & 6) == 6) ? SDL_TRUE : SDL_FALSE;
                    CPU_OSSavesZMM = (CPU_OSSavesYMM && ((a & 0xe0) == 0xe0)) ? SDL_TRUE : SDL_FALSE;
                }
            }
        }
    }
}

static int CPU_haveAltiVec(void)
{
    volatile int altivec = 0;
#ifndef SDL_CPUINFO_DISABLED
#if (defined(__MACOSX__) && (defined(__ppc__) || defined(__ppc64__))) || (defined(__OpenBSD__) && defined(__powerpc__))
#ifdef __OpenBSD__
    int selectors[2] = { CTL_MACHDEP, CPU_ALTIVEC };
#else
    int selectors[2] = { CTL_HW, HW_VECTORUNIT };
#endif
    int hasVectorUnit = 0;
    size_t length = sizeof(hasVectorUnit);
    int error = sysctl(selectors, 2, &hasVectorUnit, &length, NULL, 0);
    if (0 == error) {
        altivec = (hasVectorUnit != 0);
    }
#elif defined(__FreeBSD__) && defined(__powerpc__)
    unsigned long cpufeatures = 0;
    elf_aux_info(AT_HWCAP, &cpufeatures, sizeof(cpufeatures));
    altivec = cpufeatures & PPC_FEATURE_HAS_ALTIVEC;
    return altivec;
#elif defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP)
    void (*handler)(int sig);
    handler = signal(SIGILL, illegal_instruction);
    if (setjmp(jmpbuf) == 0) {
        asm volatile("mtspr 256, %0\n\t"
                     "vand %%v0, %%v0, %%v0" ::"r"(-1));
        altivec = 1;
    }
    signal(SIGILL, handler);
#endif
#endif
    return altivec;
}

#if (defined(__ARM_ARCH) && (__ARM_ARCH >= 6)) || defined(__aarch64__)
static int CPU_haveARMSIMD(void)
{
    return 1;
}

#elif !defined(__arm__)
static int CPU_haveARMSIMD(void)
{
    return 0;
}

#elif defined(__LINUX__)
static int CPU_haveARMSIMD(void)
{
    int arm_simd = 0;
    int fd;

    fd = open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        Elf32_auxv_t aux;
        while (read(fd, &aux, sizeof(aux)) == sizeof(aux)) {
            if (aux.a_type == AT_PLATFORM) {
                const char *plat = (const char *)aux.a_un.a_val;
                if (plat) {
                    arm_simd = SDL_strncmp(plat, "v6l", 3) == 0 ||
                               SDL_strncmp(plat, "v7l", 3) == 0;
                }
            }
        }
        close(fd);
    }
    return arm_simd;
}

#elif defined(__RISCOS__)
static int CPU_haveARMSIMD(void)
{
    _kernel_swi_regs regs;
    regs.r[0] = 0;
    if (_kernel_swi(OS_PlatformFeatures, &regs, &regs) != NULL) {
        return 0;
    }

    if (!(regs.r[0] & (1 << 31))) {
        return 0;
    }

    regs.r[0] = 34;
    regs.r[1] = 29;
    if (_kernel_swi(OS_PlatformFeatures, &regs, &regs) != NULL) {
        return 0;
    }

    return regs.r[0];
}

#else
static int CPU_haveARMSIMD(void)
{
#warning SDL_HasARMSIMD is not implemented for this ARM platform. Write me.
    return 0;
}
#endif

#if defined(__LINUX__) && defined(__arm__) && !defined(HAVE_GETAUXVAL)
static int readProcAuxvForNeon(void)
{
    int neon = 0;
    int fd;

    fd = open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        Elf32_auxv_t aux;
        while (read(fd, &aux, sizeof(aux)) == sizeof(aux)) {
            if (aux.a_type == AT_HWCAP) {
                neon = (aux.a_un.a_val & HWCAP_NEON) == HWCAP_NEON;
                break;
            }
        }
        close(fd);
    }
    return neon;
}
#endif

static int CPU_haveNEON(void)
{
/* The way you detect NEON is a privileged instruction on ARM, so you have
   query the OS kernel in a platform-specific way. :/ */
#if defined(SDL_CPUINFO_DISABLED)
    return 0; /* disabled */
#elif (defined(__WINDOWS__) || defined(__WINRT__) || defined(__GDK__)) && (defined(_M_ARM) || defined(_M_ARM64))
/* Visual Studio, for ARM, doesn't define __ARM_ARCH. Handle this first. */
/* Seems to have been removed */
#if !defined(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_NEON_INSTRUCTIONS_AVAILABLE 19
#endif
    /* All WinRT ARM devices are required to support NEON, but just in case. */
    return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;
#elif (defined(__ARM_ARCH) && (__ARM_ARCH >= 8)) || defined(__aarch64__)
    return 1; /* ARMv8 always has non-optional NEON support. */
#elif defined(__VITA__)
    return 1;
#elif defined(__3DS__)
    return 0;
#elif defined(__APPLE__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    /* (note that sysctlbyname("hw.optional.neon") doesn't work!) */
    return 1; /* all Apple ARMv7 chips and later have NEON. */
#elif defined(__APPLE__)
    return 0; /* assume anything else from Apple doesn't have NEON. */
#elif !defined(__arm__)
    return 0; /* not an ARM CPU at all. */
#elif defined(__OpenBSD__)
    return 1; /* OpenBSD only supports ARMv7 CPUs that have NEON. */
#elif defined(HAVE_ELF_AUX_INFO)
    unsigned long hasneon = 0;
    if (elf_aux_info(AT_HWCAP, (void *)&hasneon, (int)sizeof(hasneon)) != 0) {
        return 0;
    }
    return ((hasneon & HWCAP_NEON) == HWCAP_NEON);
#elif defined(__QNXNTO__)
    return SYSPAGE_ENTRY(cpuinfo)->flags & ARM_CPU_FLAG_NEON;
#elif (defined(__LINUX__) || defined(__ANDROID__)) && defined(HAVE_GETAUXVAL)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) == HWCAP_NEON;
#elif defined(__LINUX__)
    return readProcAuxvForNeon();
#elif defined(__ANDROID__)
    /* Use NDK cpufeatures to read either /proc/self/auxv or /proc/cpuinfo */
    {
        AndroidCpuFamily cpu_family = android_getCpuFamily();
        if (cpu_family == ANDROID_CPU_FAMILY_ARM) {
            uint64_t cpu_features = android_getCpuFeatures();
            if (cpu_features & ANDROID_CPU_ARM_FEATURE_NEON) {
                return 1;
            }
        }
        return 0;
    }
#elif defined(__RISCOS__)
    /* Use the VFPSupport_Features SWI to access the MVFR registers */
    {
        _kernel_swi_regs regs;
        regs.r[0] = 0;
        if (_kernel_swi(VFPSupport_Features, &regs, &regs) == NULL) {
            if ((regs.r[2] & 0xFFF000) == 0x111000) {
                return 1;
            }
        }
        return 0;
    }
#else
#warning SDL_HasNEON is not implemented for this ARM platform. Write me.
    return 0;
#endif
}

static int CPU_readCPUCFG(void)
{
    uint32_t cfg2 = 0;
#if defined __loongarch__
    __asm__ volatile(
        "cpucfg %0, %1 \n\t"
        : "+&r"(cfg2)
        : "r"(CPU_CFG2));
#endif
    return cfg2;
}

#define CPU_haveLSX()  (CPU_readCPUCFG() & CPU_CFG2_LSX)
#define CPU_haveLASX() (CPU_readCPUCFG() & CPU_CFG2_LASX)

#if defined(__e2k__)
inline int
CPU_have3DNow(void)
{
#if defined(__3dNOW__)
    return 1;
#else
    return 0;
#endif
}
#else
static int CPU_have3DNow(void)
{
    if (CPU_CPUIDMaxFunction > 0) { /* that is, do we have CPUID at all? */
        int a, b, c, d;
        cpuid(0x80000000, a, b, c, d);
        if (a >= 0x80000001) {
            cpuid(0x80000001, a, b, c, d);
            return d & 0x80000000;
        }
    }
    return 0;
}
#endif

#if defined(__e2k__)
#define CPU_haveRDTSC() (0)
#if defined(__MMX__)
#define CPU_haveMMX() (1)
#else
#define CPU_haveMMX() (0)
#endif
#if defined(__SSE__)
#define CPU_haveSSE() (1)
#else
#define CPU_haveSSE() (0)
#endif
#if defined(__SSE2__)
#define CPU_haveSSE2() (1)
#else
#define CPU_haveSSE2() (0)
#endif
#if defined(__SSE3__)
#define CPU_haveSSE3() (1)
#else
#define CPU_haveSSE3() (0)
#endif
#if defined(__SSE4_1__)
#define CPU_haveSSE41() (1)
#else
#define CPU_haveSSE41() (0)
#endif
#if defined(__SSE4_2__)
#define CPU_haveSSE42() (1)
#else
#define CPU_haveSSE42() (0)
#endif
#if defined(__AVX__)
#define CPU_haveAVX() (1)
#else
#define CPU_haveAVX() (0)
#endif
#else
#define CPU_haveRDTSC() (CPU_CPUIDFeatures[3] & 0x00000010)
#define CPU_haveMMX()   (CPU_CPUIDFeatures[3] & 0x00800000)
#define CPU_haveSSE()   (CPU_CPUIDFeatures[3] & 0x02000000)
#define CPU_haveSSE2()  (CPU_CPUIDFeatures[3] & 0x04000000)
#define CPU_haveSSE3()  (CPU_CPUIDFeatures[2] & 0x00000001)
#define CPU_haveSSE41() (CPU_CPUIDFeatures[2] & 0x00080000)
#define CPU_haveSSE42() (CPU_CPUIDFeatures[2] & 0x00100000)
#define CPU_haveAVX()   (CPU_OSSavesYMM && (CPU_CPUIDFeatures[2] & 0x10000000))
#endif

#if defined(__e2k__)
inline int
CPU_haveAVX2(void)
{
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
#else
static int CPU_haveAVX2(void)
{
    if (CPU_OSSavesYMM && (CPU_CPUIDMaxFunction >= 7)) {
        int a, b, c, d;
        (void)a;
        (void)b;
        (void)c;
        (void)d; /* compiler warnings... */
        cpuid(7, a, b, c, d);
        return b & 0x00000020;
    }
    return 0;
}
#endif

#if defined(__e2k__)
inline int
CPU_haveAVX512F(void)
{
    return 0;
}
#else
static int CPU_haveAVX512F(void)
{
    if (CPU_OSSavesZMM && (CPU_CPUIDMaxFunction >= 7)) {
        int a, b, c, d;
        (void)a;
        (void)b;
        (void)c;
        (void)d; /* compiler warnings... */
        cpuid(7, a, b, c, d);
        return b & 0x00010000;
    }
    return 0;
}
#endif

static int SDL_CPUCount = 0;

int SDL_GetCPUCount(void)
{
    if (!SDL_CPUCount) {
#ifndef SDL_CPUINFO_DISABLED
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_ONLN)
        if (SDL_CPUCount <= 0) {
            SDL_CPUCount = (int)sysconf(_SC_NPROCESSORS_ONLN);
        }
#endif
#ifdef HAVE_SYSCTLBYNAME
        if (SDL_CPUCount <= 0) {
            size_t size = sizeof(SDL_CPUCount);
            sysctlbyname("hw.ncpu", &SDL_CPUCount, &size, NULL, 0);
        }
#endif
#if defined(__WIN32__) || defined(__GDK__)
        if (SDL_CPUCount <= 0) {
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            SDL_CPUCount = info.dwNumberOfProcessors;
        }
#endif
#ifdef __OS2__
        if (SDL_CPUCount <= 0) {
            DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                            &SDL_CPUCount, sizeof(SDL_CPUCount) );
        }
#endif
#endif
        /* There has to be at least 1, right? :) */
        if (SDL_CPUCount <= 0) {
            SDL_CPUCount = 1;
        }
    }
    return SDL_CPUCount;
}

#if defined(__e2k__)
inline const char *
SDL_GetCPUType(void)
{
    static char SDL_CPUType[13];

    SDL_strlcpy(SDL_CPUType, "E2K MACHINE", sizeof(SDL_CPUType));

    return SDL_CPUType;
}
#else
/* Oh, such a sweet sweet trick, just not very useful. :) */
static const char *SDL_GetCPUType(void)
{
    static char SDL_CPUType[13];

    if (!SDL_CPUType[0]) {
        int i = 0;

        CPU_calcCPUIDFeatures();
        if (CPU_CPUIDMaxFunction > 0) { /* do we have CPUID at all? */
            int a, b, c, d;
            cpuid(0x00000000, a, b, c, d);
            (void)a;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);

            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);

            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
        }
        if (!SDL_CPUType[0]) {
            SDL_strlcpy(SDL_CPUType, "Unknown", sizeof(SDL_CPUType));
        }
    }
    return SDL_CPUType;
}
#endif

#ifdef TEST_MAIN /* !!! FIXME: only used for test at the moment. */
#if defined(__e2k__)
inline const char *
SDL_GetCPUName(void)
{
    static char SDL_CPUName[48];

    SDL_strlcpy(SDL_CPUName, __builtin_cpu_name(), sizeof(SDL_CPUName));

    return SDL_CPUName;
}
#else
static const char *SDL_GetCPUName(void)
{
    static char SDL_CPUName[48];

    if (!SDL_CPUName[0]) {
        int i = 0;
        int a, b, c, d;

        CPU_calcCPUIDFeatures();
        if (CPU_CPUIDMaxFunction > 0) { /* do we have CPUID at all? */
            cpuid(0x80000000, a, b, c, d);
            if (a >= 0x80000004) {
                cpuid(0x80000002, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                cpuid(0x80000003, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                cpuid(0x80000004, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
            }
        }
        if (!SDL_CPUName[0]) {
            SDL_strlcpy(SDL_CPUName, "Unknown", sizeof(SDL_CPUName));
        }
    }
    return SDL_CPUName;
}
#endif
#endif

int SDL_GetCPUCacheLineSize(void)
{
    const char *cpuType = SDL_GetCPUType();
    int a, b, c, d;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    if (SDL_strcmp(cpuType, "GenuineIntel") == 0 || SDL_strcmp(cpuType, "CentaurHauls") == 0 || SDL_strcmp(cpuType, "  Shanghai  ") == 0) {
        cpuid(0x00000001, a, b, c, d);
        return ((b >> 8) & 0xff) * 8;
    } else if (SDL_strcmp(cpuType, "AuthenticAMD") == 0 || SDL_strcmp(cpuType, "HygonGenuine") == 0) {
        cpuid(0x80000005, a, b, c, d);
        return c & 0xff;
    } else {
        /* Just make a guess here... */
        return SDL_CACHELINE_SIZE;
    }
}

static Uint32 SDL_CPUFeatures = 0xFFFFFFFF;
static Uint32 SDL_SIMDAlignment = 0xFFFFFFFF;

static Uint32 SDL_GetCPUFeatures(void)
{
    if (SDL_CPUFeatures == 0xFFFFFFFF) {
        CPU_calcCPUIDFeatures();
        SDL_CPUFeatures = 0;
        SDL_SIMDAlignment = sizeof(void *); /* a good safe base value */
        if (CPU_haveRDTSC()) {
            SDL_CPUFeatures |= CPU_HAS_RDTSC;
        }
        if (CPU_haveAltiVec()) {
            SDL_CPUFeatures |= CPU_HAS_ALTIVEC;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveMMX()) {
            SDL_CPUFeatures |= CPU_HAS_MMX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 8);
        }
        if (CPU_have3DNow()) {
            SDL_CPUFeatures |= CPU_HAS_3DNOW;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 8);
        }
        if (CPU_haveSSE()) {
            SDL_CPUFeatures |= CPU_HAS_SSE;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE2()) {
            SDL_CPUFeatures |= CPU_HAS_SSE2;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE3()) {
            SDL_CPUFeatures |= CPU_HAS_SSE3;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE41()) {
            SDL_CPUFeatures |= CPU_HAS_SSE41;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE42()) {
            SDL_CPUFeatures |= CPU_HAS_SSE42;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveAVX()) {
            SDL_CPUFeatures |= CPU_HAS_AVX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
        if (CPU_haveAVX2()) {
            SDL_CPUFeatures |= CPU_HAS_AVX2;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
        if (CPU_haveAVX512F()) {
            SDL_CPUFeatures |= CPU_HAS_AVX512F;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 64);
        }
        if (CPU_haveARMSIMD()) {
            SDL_CPUFeatures |= CPU_HAS_ARM_SIMD;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveNEON()) {
            SDL_CPUFeatures |= CPU_HAS_NEON;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveLSX()) {
            SDL_CPUFeatures |= CPU_HAS_LSX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveLASX()) {
            SDL_CPUFeatures |= CPU_HAS_LASX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
    }
    return SDL_CPUFeatures;
}

#define CPU_FEATURE_AVAILABLE(f) ((SDL_GetCPUFeatures() & f) ? SDL_TRUE : SDL_FALSE)

SDL_bool SDL_HasRDTSC(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_RDTSC);
}

SDL_bool SDL_HasAltiVec(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_ALTIVEC);
}

SDL_bool SDL_HasMMX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_MMX);
}

SDL_bool SDL_Has3DNow(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_3DNOW);
}

SDL_bool SDL_HasSSE(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE);
}

SDL_bool SDL_HasSSE2(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE2);
}

SDL_bool SDL_HasSSE3(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE3);
}

SDL_bool SDL_HasSSE41(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE41);
}

SDL_bool SDL_HasSSE42(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE42);
}

SDL_bool SDL_HasAVX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX);
}

SDL_bool SDL_HasAVX2(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX2);
}

SDL_bool SDL_HasAVX512F(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX512F);
}

SDL_bool SDL_HasARMSIMD(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_ARM_SIMD);
}

SDL_bool SDL_HasNEON(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_NEON);
}

SDL_bool SDL_HasLSX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_LSX);
}

SDL_bool SDL_HasLASX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_LASX);
}

static int SDL_SystemRAM = 0;

int SDL_GetSystemRAM(void)
{
    if (!SDL_SystemRAM) {
#ifndef SDL_CPUINFO_DISABLED
#if defined(HAVE_SYSCONF) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
        if (SDL_SystemRAM <= 0) {
            SDL_SystemRAM = (int)((Sint64)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE) / (1024 * 1024));
        }
#endif
#ifdef HAVE_SYSCTLBYNAME
        if (SDL_SystemRAM <= 0) {
#ifdef HW_PHYSMEM64
            /* (64-bit): NetBSD since 2003, OpenBSD */
            int mib[2] = { CTL_HW, HW_PHYSMEM64 };
#elif defined(HW_REALMEM)
            /* (64-bit): FreeBSD since 2005, DragonFly */
            int mib[2] = { CTL_HW, HW_REALMEM };
#elif defined(HW_MEMSIZE)
            /* (64-bit): Darwin */
            int mib[2] = { CTL_HW, HW_MEMSIZE };
#else
            /* (32-bit): very old BSD, might only report up to 2 GiB */
            int mib[2] = { CTL_HW, HW_PHYSMEM };
#endif /* HW_PHYSMEM64 */
            Uint64 memsize = 0;
            size_t len = sizeof(memsize);

            if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
                SDL_SystemRAM = (int)(memsize / (1024 * 1024));
            }
        }
#endif
#if defined(__WIN32__) || defined(__GDK__)
        if (SDL_SystemRAM <= 0) {
            MEMORYSTATUSEX stat;
            stat.dwLength = sizeof(stat);
            if (GlobalMemoryStatusEx(&stat)) {
                SDL_SystemRAM = (int)(stat.ullTotalPhys / (1024 * 1024));
            }
        }
#endif
#ifdef __OS2__
        if (SDL_SystemRAM <= 0) {
            Uint32 sysram = 0;
            DosQuerySysInfo(QSV_TOTPHYSMEM, QSV_TOTPHYSMEM, &sysram, 4);
            SDL_SystemRAM = (int) (sysram / 0x100000U);
        }
#endif
#ifdef __RISCOS__
        if (SDL_SystemRAM <= 0) {
            _kernel_swi_regs regs;
            regs.r[0] = 0x108;
            if (_kernel_swi(OS_Memory, &regs, &regs) == NULL) {
                SDL_SystemRAM = (int)(regs.r[1] * regs.r[2] / (1024 * 1024));
            }
        }
#endif
#ifdef __VITA__
        if (SDL_SystemRAM <= 0) {
            /* Vita has 512MiB on SoC, that's split into 256MiB(+109MiB in extended memory mode) for app
               +26MiB of physically continuous memory, +112MiB of CDRAM(VRAM) + system reserved memory. */
            SDL_SystemRAM = 536870912;
        }
#endif
#ifdef __PS2__
        if (SDL_SystemRAM <= 0) {
            /* PlayStation 2 has 32MiB however there are some special models with 64 and 128 */
            SDL_SystemRAM = GetMemorySize();
        }
#endif
#endif
    }
    return SDL_SystemRAM;
}

size_t SDL_SIMDGetAlignment(void)
{
    if (SDL_SIMDAlignment == 0xFFFFFFFF) {
        SDL_GetCPUFeatures(); /* make sure this has been calculated */
    }
    SDL_assert(SDL_SIMDAlignment != 0);
    return SDL_SIMDAlignment;
}

void *SDL_SIMDAlloc(const size_t len)
{
    const size_t alignment = SDL_SIMDGetAlignment();
    const size_t padding = (alignment - (len % alignment)) % alignment;
    Uint8 *retval = NULL;
    Uint8 *ptr;
    size_t to_allocate;

    /* alignment + padding + sizeof(void *) is bounded (a few hundred
     * bytes max), so no need to check for overflow within that argument */
    if (SDL_size_add_overflow(len, alignment + padding + sizeof(void *), &to_allocate)) {
        return NULL;
    }

    ptr = (Uint8 *)SDL_malloc(to_allocate);
    if (ptr) {
        /* store the actual allocated pointer right before our aligned pointer. */
        retval = ptr + sizeof(void *);
        retval += alignment - (((size_t)retval) % alignment);
        *(((void **)retval) - 1) = ptr;
    }
    return retval;
}

void *SDL_SIMDRealloc(void *mem, const size_t len)
{
    const size_t alignment = SDL_SIMDGetAlignment();
    const size_t padding = (alignment - (len % alignment)) % alignment;
    Uint8 *retval = (Uint8 *)mem;
    void *oldmem = mem;
    size_t memdiff = 0, ptrdiff;
    Uint8 *ptr;
    size_t to_allocate;

    /* alignment + padding + sizeof(void *) is bounded (a few hundred
     * bytes max), so no need to check for overflow within that argument */
    if (SDL_size_add_overflow(len, alignment + padding + sizeof(void *), &to_allocate)) {
        return NULL;
    }

    if (mem) {
        mem = *(((void **)mem) - 1);

        /* Check the delta between the real pointer and user pointer */
        memdiff = ((size_t)oldmem) - ((size_t)mem);
    }

    ptr = (Uint8 *)SDL_realloc(mem, to_allocate);

    if (!ptr) {
        return NULL; /* Out of memory, bail! */
    }

    /* Store the actual allocated pointer right before our aligned pointer. */
    retval = ptr + sizeof(void *);
    retval += alignment - (((size_t)retval) % alignment);

    /* Make sure the delta is the same! */
    if (mem) {
        ptrdiff = ((size_t)retval) - ((size_t)ptr);
        if (memdiff != ptrdiff) { /* Delta has changed, copy to new offset! */
            oldmem = (void *)(((uintptr_t)ptr) + memdiff);

            /* Even though the data past the old `len` is undefined, this is the
             * only length value we have, and it guarantees that we copy all the
             * previous memory anyhow.
             */
            SDL_memmove(retval, oldmem, len);
        }
    }

    /* Actually store the allocated pointer, finally. */
    *(((void **)retval) - 1) = ptr;
    return retval;
}

void SDL_SIMDFree(void *ptr)
{
    if (ptr) {
        SDL_free(*(((void **)ptr) - 1));
    }
}

#ifdef TEST_MAIN

#include <stdio.h>

int main(void)
{
    printf("CPU count: %d\n", SDL_GetCPUCount());
    printf("CPU type: %s\n", SDL_GetCPUType());
    printf("CPU name: %s\n", SDL_GetCPUName());
    printf("CacheLine size: %d\n", SDL_GetCPUCacheLineSize());
    printf("RDTSC: %d\n", SDL_HasRDTSC());
    printf("Altivec: %d\n", SDL_HasAltiVec());
    printf("MMX: %d\n", SDL_HasMMX());
    printf("3DNow: %d\n", SDL_Has3DNow());
    printf("SSE: %d\n", SDL_HasSSE());
    printf("SSE2: %d\n", SDL_HasSSE2());
    printf("SSE3: %d\n", SDL_HasSSE3());
    printf("SSE4.1: %d\n", SDL_HasSSE41());
    printf("SSE4.2: %d\n", SDL_HasSSE42());
    printf("AVX: %d\n", SDL_HasAVX());
    printf("AVX2: %d\n", SDL_HasAVX2());
    printf("AVX-512F: %d\n", SDL_HasAVX512F());
    printf("ARM SIMD: %d\n", SDL_HasARMSIMD());
    printf("NEON: %d\n", SDL_HasNEON());
    printf("LSX: %d\n", SDL_HasLSX());
    printf("LASX: %d\n", SDL_HasLASX());
    printf("RAM: %d MB\n", SDL_GetSystemRAM());
    return 0;
}

#endif /* TEST_MAIN */

/* vi: set ts=4 sw=4 expandtab: */
