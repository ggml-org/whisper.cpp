#!/usr/bin/env python
import argparse
import dataclasses
import fnmatch
from enum import Enum
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


class AppleArch(Enum):
    ARM64 = "arm64"
    X86_64 = "x86_64"


class MsvcArch(Enum):
    X86 = "x86"
    X64 = "x64"
    Arm32 = "arm"
    Arm64 = "arm64"


class JobOs(Enum):
    WindowsLatest = "windows-latest"
    UbuntuLatest = "ubuntu-latest"
    MacosLatest = "macos-latest"
    Ubuntu20_04 = "ubuntu-20.04"
    Ubuntu22_04 = "ubuntu-22.04"
    Ubuntu24_04 = "ubuntu-24.04"
    Macos13 = "macos-13"


class SdlPlatform(Enum):
    Android = "android"
    Emscripten = "emscripten"
    Haiku = "haiku"
    Msys2 = "msys2"
    Linux = "linux"
    MacOS = "macos"
    Ios = "ios"
    Tvos = "tvos"
    Msvc = "msvc"
    N3ds = "n3ds"
    Ps2 = "ps2"
    Psp = "psp"
    Vita = "vita"
    Riscos = "riscos"
    FreeBSD = "freebsd"
    NetBSD = "netbsd"
    Watcom = "watcom"


class Msys2Platform(Enum):
    Mingw32 = "mingw32"
    Mingw64 = "mingw64"
    Clang32 = "clang32"
    Clang64 = "clang64"
    Ucrt64 = "ucrt64"


class IntelCompiler(Enum):
    Icc = "icc"
    Icx = "icx"


class VitaGLES(Enum):
    Pib = "pib"
    Pvr = "pvr"

class WatcomPlatform(Enum):
    Windows = "windows"
    OS2 = "OS/2"


@dataclasses.dataclass(slots=True)
class JobSpec:
    name: str
    os: JobOs
    platform: SdlPlatform
    artifact: Optional[str]
    container: Optional[str] = None
    autotools: bool = False
    no_cmake: bool = False
    xcode: bool = False
    android_mk: bool = False
    lean: bool = False
    android_arch: Optional[str] = None
    android_abi: Optional[str] = None
    android_platform: Optional[int] = None
    msys2_platform: Optional[Msys2Platform] = None
    intel: Optional[IntelCompiler] = None
    apple_archs: Optional[set[AppleArch]] = None
    msvc_project: Optional[str] = None
    msvc_arch: Optional[MsvcArch] = None
    msvc_static_crt: bool = False
    clang_cl: bool = False
    gdk: bool = False
    uwp: bool = False
    vita_gles: Optional[VitaGLES] = None
    watcom_platform: Optional[WatcomPlatform] = None


JOB_SPECS = {
    "msys2-mingw32": JobSpec(name="Windows (msys2, mingw32)",               os=JobOs.WindowsLatest, platform=SdlPlatform.Msys2,       artifact="SDL-mingw32",            msys2_platform=Msys2Platform.Mingw32, ),
    "msys2-mingw64": JobSpec(name="Windows (msys2, mingw64)",               os=JobOs.WindowsLatest, platform=SdlPlatform.Msys2,       artifact="SDL-mingw64",            msys2_platform=Msys2Platform.Mingw64, ),
    "msys2-clang32": JobSpec(name="Windows (msys2, clang32)",               os=JobOs.WindowsLatest, platform=SdlPlatform.Msys2,       artifact="SDL-mingw32-clang",      msys2_platform=Msys2Platform.Clang32, ),
    "msys2-clang64": JobSpec(name="Windows (msys2, clang64)",               os=JobOs.WindowsLatest, platform=SdlPlatform.Msys2,       artifact="SDL-mingw64-clang",      msys2_platform=Msys2Platform.Clang64, ),
    "msys2-ucrt64": JobSpec(name="Windows (msys2, ucrt64)",                 os=JobOs.WindowsLatest, platform=SdlPlatform.Msys2,       artifact="SDL-mingw64-ucrt",       msys2_platform=Msys2Platform.Ucrt64, ),
    "msvc-x64": JobSpec(name="Windows (MSVC, x64)",                         os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-x64",             msvc_arch=MsvcArch.X64,   msvc_project="VisualC/SDL.sln", ),
    "msvc-x86": JobSpec(name="Windows (MSVC, x86)",                         os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-x86",             msvc_arch=MsvcArch.X86,   msvc_project="VisualC/SDL.sln", ),
    "msvc-arm64": JobSpec(name="Windows (MSVC, ARM64)",                     os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-arm64",           msvc_arch=MsvcArch.Arm64, ),
    "msvc-arm32": JobSpec(name="Windows (MSVC, ARM)",                       os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-arm32",           msvc_arch=MsvcArch.Arm32, ),
    "msvc-static-x64": JobSpec(name="Windows (MSVC, static VCRT, x64)",     os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-x64",             msvc_arch=MsvcArch.X64,   msvc_static_crt=True, ),
    "msvc-static-x86": JobSpec(name="Windows (MSVC, static VCRT, x86)",     os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-x86",             msvc_arch=MsvcArch.X86,   msvc_static_crt=True, ),
    "msvc-clang-x64": JobSpec(name="Windows (MSVC, clang-cl x64)",          os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-clang-cl-x64",       msvc_arch=MsvcArch.X64,   clang_cl=True, ),
    "msvc-clang-x86": JobSpec(name="Windows (MSVC, clang-cl x86)",          os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-clang-cl-x86",       msvc_arch=MsvcArch.X86,   clang_cl=True, ),
    "msvc-uwp-x64": JobSpec(name="UWP (MSVC, x64)",                         os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-UWP",             msvc_arch=MsvcArch.X64,   msvc_project="VisualC-WinRT/SDL-UWP.sln", uwp=True, ),
    "msvc-gdk-x64": JobSpec(name="GDK (MSVC, x64)",                         os=JobOs.WindowsLatest, platform=SdlPlatform.Msvc,        artifact="SDL-VC-GDK",             msvc_arch=MsvcArch.X64,   msvc_project="VisualC-GDK/SDL.sln", gdk=True, no_cmake=True, ),
    "ubuntu-20.04": JobSpec(name="Ubuntu 20.04",                            os=JobOs.Ubuntu20_04,   platform=SdlPlatform.Linux,       artifact="SDL-ubuntu20.04",        autotools=True),
    "ubuntu-22.04": JobSpec(name="Ubuntu 22.04",                            os=JobOs.Ubuntu22_04,   platform=SdlPlatform.Linux,       artifact="SDL-ubuntu22.04",        autotools=True),
    "steamrt-sniper": JobSpec(name="Steam Linux Runtime (Sniper)",          os=JobOs.UbuntuLatest,  platform=SdlPlatform.Linux,       artifact="SDL-slrsniper",          container="registry.gitlab.steamos.cloud/steamrt/sniper/sdk:beta", ),
    "ubuntu-intel-icx": JobSpec(name="Ubuntu 20.04 (Intel oneAPI)",         os=JobOs.Ubuntu20_04,   platform=SdlPlatform.Linux,       artifact="SDL-ubuntu20.04-oneapi", intel=IntelCompiler.Icx, ),
    "ubuntu-intel-icc": JobSpec(name="Ubuntu 20.04 (Intel Compiler)",       os=JobOs.Ubuntu20_04,   platform=SdlPlatform.Linux,       artifact="SDL-ubuntu20.04-icc",    intel=IntelCompiler.Icc, ),
    "macos-gnu-arm64-x64": JobSpec(name="MacOS (GNU prefix)",               os=JobOs.MacosLatest,   platform=SdlPlatform.MacOS,       artifact="SDL-macos-arm64-x64-gnu",autotools=True, apple_archs={AppleArch.X86_64, AppleArch.ARM64, },  ),
    "ios": JobSpec(name="iOS (CMake & xcode)",                              os=JobOs.MacosLatest,   platform=SdlPlatform.Ios,         artifact="SDL-ios-arm64",          xcode=True, ),
    "tvos": JobSpec(name="tvOS (CMake & xcode)",                            os=JobOs.MacosLatest,   platform=SdlPlatform.Tvos,        artifact="SDL-tvos-arm64",         xcode=True, ),
    "android-cmake": JobSpec(name="Android (CMake)",                        os=JobOs.UbuntuLatest,  platform=SdlPlatform.Android,     artifact="SDL-android-arm64",      android_abi="arm64-v8a", android_arch="aarch64", android_platform=23, ),
    "android-mk": JobSpec(name="Android (Android.mk)",                      os=JobOs.UbuntuLatest,  platform=SdlPlatform.Android,     artifact=None,                     no_cmake=True, android_mk=True, ),
    "emscripten": JobSpec(name="Emscripten",                                os=JobOs.UbuntuLatest,  platform=SdlPlatform.Emscripten,  artifact="SDL-emscripten", ),
    "haiku": JobSpec(name="Haiku",                                          os=JobOs.UbuntuLatest,  platform=SdlPlatform.Haiku,       artifact="SDL-haiku-x64",          container="ghcr.io/haiku/cross-compiler:x86_64-r1beta5", ),
    "n3ds": JobSpec(name="Nintendo 3DS",                                    os=JobOs.UbuntuLatest,  platform=SdlPlatform.N3ds,        artifact="SDL-n3ds",               container="devkitpro/devkitarm:latest", ),
    "ps2": JobSpec(name="Sony PlayStation 2",                               os=JobOs.UbuntuLatest,  platform=SdlPlatform.Ps2,         artifact="SDL-ps2",                container="ps2dev/ps2dev:latest", ),
    "psp": JobSpec(name="Sony PlayStation Portable",                        os=JobOs.UbuntuLatest,  platform=SdlPlatform.Psp,         artifact="SDL-psp",                container="pspdev/pspdev:latest", ),
    "vita-pib": JobSpec(name="Sony PlayStation Vita (GLES w/ pib)",         os=JobOs.UbuntuLatest,  platform=SdlPlatform.Vita,        artifact="SDL-vita-pib",           container="vitasdk/vitasdk:latest", vita_gles=VitaGLES.Pib,  ),
    "vita-pvr": JobSpec(name="Sony PlayStation Vita (GLES w/ PVR_PSP2)",    os=JobOs.UbuntuLatest,  platform=SdlPlatform.Vita,        artifact="SDL-vita-pvr",           container="vitasdk/vitasdk:latest", vita_gles=VitaGLES.Pvr, ),
    "riscos": JobSpec(name="RISC OS",                                       os=JobOs.UbuntuLatest,  platform=SdlPlatform.Riscos,      artifact="SDL-riscos",             container="riscosdotinfo/riscos-gccsdk-4.7:latest", ),
    "netbsd": JobSpec(name="NetBSD",                                        os=JobOs.UbuntuLatest,  platform=SdlPlatform.NetBSD,      artifact="SDL-netbsd-x64",  autotools=True, ),
    "freebsd": JobSpec(name="FreeBSD",                                      os=JobOs.UbuntuLatest,  platform=SdlPlatform.FreeBSD,     artifact="SDL-freebsd-x64", autotools=True, ),
    "watcom-win32": JobSpec(name="Watcom (Windows)",                        os=JobOs.WindowsLatest, platform=SdlPlatform.Watcom,      artifact="SDL-watcom-win32",  no_cmake=True, watcom_platform=WatcomPlatform.Windows ),
    "watcom-os2": JobSpec(name="Watcom (OS/2)",                             os=JobOs.WindowsLatest, platform=SdlPlatform.Watcom,      artifact="SDL-watcom-win32",  no_cmake=True, watcom_platform=WatcomPlatform.OS2 ),
    # "watcom-win32"
    # "watcom-os2"
}


class StaticLibType(Enum):
    MSVC = "SDL2-static.lib"
    A = "libSDL2.a"


class SharedLibType(Enum):
    WIN32 = "SDL2.dll"
    SO_0 = "libSDL2-2.0.so.0"
    SO = "libSDL2.so"
    DYLIB = "libSDL2-2.0.0.dylib"
    FRAMEWORK = "SDL2.framework/Versions/A/SDL2"


@dataclasses.dataclass(slots=True)
class JobDetails:
    name: str
    key: str
    os: str
    platform: str
    artifact: str
    no_autotools: bool
    no_cmake: bool
    build_autotools_tests: bool = True
    build_tests: bool = True
    container: str = ""
    cmake_build_type: str = "RelWithDebInfo"
    shell: str = "sh"
    sudo: str = "sudo"
    cmake_config_emulator: str = ""
    apk_packages: list[str] = dataclasses.field(default_factory=list)
    apt_packages: list[str] = dataclasses.field(default_factory=list)
    brew_packages: list[str] = dataclasses.field(default_factory=list)
    cmake_toolchain_file: str = ""
    cmake_arguments: list[str] = dataclasses.field(default_factory=list)
    cmake_build_arguments: list[str] = dataclasses.field(default_factory=list)
    cppflags: list[str] = dataclasses.field(default_factory=list)
    cc: str = ""
    cxx: str = ""
    cflags: list[str] = dataclasses.field(default_factory=list)
    cxxflags: list[str] = dataclasses.field(default_factory=list)
    ldflags: list[str] = dataclasses.field(default_factory=list)
    pollute_directories: list[str] = dataclasses.field(default_factory=list)
    use_cmake: bool = True
    shared: bool = True
    static: bool = True
    shared_lib: Optional[SharedLibType] = None
    static_lib: Optional[StaticLibType] = None
    run_tests: bool = True
    test_pkg_config: bool = True
    cc_from_cmake: bool = False
    source_cmd: str = ""
    pretest_cmd: str = ""
    android_apks: list[str] = dataclasses.field(default_factory=list)
    android_ndk: bool = False
    android_mk: bool = False
    minidump: bool = False
    intel: bool = False
    msys2_msystem: str = ""
    msys2_env: str = ""
    msys2_no_perl: bool = False
    werror: bool = True
    msvc_vcvars_arch: str = ""
    msvc_vcvars_sdk: str = ""
    msvc_project: str = ""
    msvc_project_flags: list[str] = dataclasses.field(default_factory=list)
    setup_ninja: bool = False
    setup_libusb_arch: str = ""
    xcode_sdk: str = ""
    xcode_target: str = ""
    setup_gdk_folder: str = ""
    cpactions: bool = False
    cpactions_os: str = ""
    cpactions_version: str = ""
    cpactions_arch: str = ""
    cpactions_setup_cmd: str = ""
    cpactions_install_cmd: str = ""
    setup_vita_gles_type: str = ""
    check_sources: bool = False
    watcom_makefile: str = ""

    def to_workflow(self, enable_artifacts: bool) -> dict[str, str|bool]:
        data = {
            "name": self.name,
            "key": self.key,
            "os": self.os,
            "container": self.container if self.container else "",
            "platform": self.platform,
            "artifact": self.artifact,
            "enable-artifacts": enable_artifacts,
            "shell": self.shell,
            "msys2-msystem": self.msys2_msystem,
            "msys2-env": self.msys2_env,
            "msys2-no-perl": self.msys2_no_perl,
            "android-ndk": self.android_ndk,
            "intel": self.intel,
            "apk-packages": my_shlex_join(self.apk_packages),
            "apt-packages": my_shlex_join(self.apt_packages),
            "test-pkg-config": self.test_pkg_config,
            "brew-packages": my_shlex_join(self.brew_packages),
            "pollute-directories": my_shlex_join(self.pollute_directories),
            "no-autotools": self.no_autotools,
            "no-cmake": self.no_cmake,
            "build-autotools-tests": self.build_autotools_tests,
            "build-tests": self.build_tests,
            "source-cmd": self.source_cmd,
            "pretest-cmd": self.pretest_cmd,
            "cmake-config-emulator": self.cmake_config_emulator,
            "cc": self.cc,
            "cxx": self.cxx,
            "cflags": my_shlex_join(self.cppflags + self.cflags),
            "cxxflags": my_shlex_join(self.cppflags + self.cxxflags),
            "ldflags": my_shlex_join(self.ldflags),
            "cmake-toolchain-file": self.cmake_toolchain_file,
            "cmake-arguments": my_shlex_join(self.cmake_arguments),
            "cmake-build-arguments": my_shlex_join(self.cmake_build_arguments),
            "shared": self.shared,
            "static": self.static,
            "shared-lib": self.shared_lib.value if self.shared_lib else None,
            "static-lib": self.static_lib.value if self.static_lib else None,
            "cmake-build-type": self.cmake_build_type,
            "run-tests": self.run_tests,
            "android-apks": my_shlex_join(self.android_apks),
            "android-mk": self.android_mk,
            "werror": self.werror,
            "sudo": self.sudo,
            "msvc-vcvars-arch": self.msvc_vcvars_arch,
            "msvc-vcvars-sdk": self.msvc_vcvars_sdk,
            "msvc-project": self.msvc_project,
            "msvc-project-flags": my_shlex_join(self.msvc_project_flags),
            "setup-ninja": self.setup_ninja,
            "setup-libusb-arch": self.setup_libusb_arch,
            "cc-from-cmake": self.cc_from_cmake,
            "xcode-sdk": self.xcode_sdk,
            "xcode-target": self.xcode_target,
            "cpactions": self.cpactions,
            "cpactions-os": self.cpactions_os,
            "cpactions-version": self.cpactions_version,
            "cpactions-arch": self.cpactions_arch,
            "cpactions-setup-cmd": self.cpactions_setup_cmd,
            "cpactions-install-cmd": self.cpactions_install_cmd,
            "setup-vita-gles-type": self.setup_vita_gles_type,
            "setup-gdk-folder": self.setup_gdk_folder,
            "check-sources": self.check_sources,
            "watcom-makefile": self.watcom_makefile,
        }
        return {k: v for k, v in data.items() if v != ""}


def my_shlex_join(s):
    def escape(s):
        if s[:1] == "'" and s[-1:] == "'":
            return s
        if set(s).intersection(set("; \t")):
            s = s.replace("\\", "\\\\").replace("\"", "\\\"")
            return f'"{s}"'
        return s

    return " ".join(escape(e) for e in s)


def spec_to_job(spec: JobSpec, key: str, trackmem_symbol_names: bool) -> JobDetails:
    job = JobDetails(
        name=spec.name,
        key=key,
        os=spec.os.value,
        artifact=spec.artifact or "",
        container=spec.container or "",
        platform=spec.platform.value,
        sudo="sudo",
        no_autotools=not spec.autotools,
        no_cmake=spec.no_cmake,
    )
    if job.os.startswith("ubuntu"):
        job.apt_packages.extend([
            "ninja-build",
            "pkg-config",
        ])
    if spec.msvc_static_crt:
        job.cmake_arguments.append("-DSDL_FORCE_STATIC_VCRT=ON")
    pretest_cmd = []
    if trackmem_symbol_names:
        pretest_cmd.append("export SDL_TRACKMEM_SYMBOL_NAMES=1")
    else:
        pretest_cmd.append("export SDL_TRACKMEM_SYMBOL_NAMES=0")
    win32 = spec.platform in (SdlPlatform.Msys2, SdlPlatform.Msvc)
    fpic = None
    build_parallel = True
    if spec.lean:
        job.cppflags.append("-DSDL_LEAN_AND_MEAN=1")
    if win32:
        job.cmake_arguments.append("-DSDLTEST_PROCDUMP=ON")
        job.minidump = True
    if spec.intel is not None:
        match spec.intel:
            case IntelCompiler.Icx:
                job.cc = "icx"
                job.cxx = "icpx"
            case IntelCompiler.Icc:
                job.cc = "icc"
                job.cxx = "icpc"
                job.cppflags.append("-diag-disable=10441")
            case _:
                raise ValueError(f"Invalid intel={spec.intel}")
        job.source_cmd = f"source /opt/intel/oneapi/setvars.sh;"
        job.intel = True
        job.shell = "bash"
        job.cmake_arguments.extend((
            f"-DCMAKE_C_COMPILER={job.cc}",
            f"-DCMAKE_CXX_COMPILER={job.cxx}",
            "-DCMAKE_SYSTEM_NAME=Linux",
        ))
    match spec.platform:
        case SdlPlatform.Msvc:
            job.setup_ninja = not spec.gdk
            job.msvc_project = spec.msvc_project if spec.msvc_project else ""
            job.msvc_project_flags.append("-p:TreatWarningsAsError=true")
            job.test_pkg_config = False
            job.shared_lib = SharedLibType.WIN32
            job.static_lib = StaticLibType.MSVC
            job.cmake_arguments.extend((
                "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase",
                "-DCMAKE_EXE_LINKER_FLAGS=-DEBUG",
                "-DCMAKE_SHARED_LINKER_FLAGS=-DEBUG",
            ))
            if spec.uwp:
                job.cmake_arguments.append("'-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>DLL'")
            else:
                job.cmake_arguments.append("'-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>'")

            if spec.clang_cl:
                job.cmake_arguments.extend((
                    "-DCMAKE_C_COMPILER=clang-cl",
                    "-DCMAKE_CXX_COMPILER=clang-cl",
                ))
                match spec.msvc_arch:
                    case MsvcArch.X86:
                        job.cflags.append("/clang:-m32")
                        job.cxxflags.append("/clang:-m32")
                        job.ldflags.append("/MACHINE:X86")
                    case MsvcArch.X64:
                        job.cflags.append("/clang:-m64")
                        job.cxxflags.append("/clang:-m64")
                        job.ldflags.append("/MACHINE:X64")
                    case _:
                        raise ValueError(f"Unsupported clang-cl architecture (arch={spec.msvc_arch})")
            if spec.msvc_project:
                match spec.msvc_arch:
                    case MsvcArch.X86:
                        msvc_platform = "Win32"
                    case MsvcArch.X64:
                        msvc_platform = "x64"
                    case _:
                        raise ValueError(f"Unsupported vcxproj architecture (arch={spec.msvc_arch})")
                if spec.gdk:
                    msvc_platform = f"Gaming.Desktop.{msvc_platform}"
                job.msvc_project_flags.append(f"-p:Platform={msvc_platform}")
            match spec.msvc_arch:
                case MsvcArch.X86:
                    job.msvc_vcvars_arch = "x64_x86"
                case MsvcArch.X64:
                    job.msvc_vcvars_arch = "x64"
                case MsvcArch.Arm32:
                    job.msvc_vcvars_arch = "x64_arm"
                    job.msvc_vcvars_sdk = "10.0.22621.0"  # 10.0.26100.0 dropped ARM32 um and ucrt libraries
                    job.run_tests = False
                case MsvcArch.Arm64:
                    job.msvc_vcvars_arch = "x64_arm64"
                    job.run_tests = False
            if spec.uwp:
                job.build_tests = False
                job.cmake_arguments.extend((
                    "-DCMAKE_SYSTEM_NAME=WindowsStore",
                    "-DCMAKE_SYSTEM_VERSION=10.0",
                ))
                job.msvc_project_flags.append("-p:WindowsTargetPlatformVersion=10.0.17763.0")
            elif spec.gdk:
                job.setup_gdk_folder = "VisualC-GDK"
            else:
                match spec.msvc_arch:
                    case MsvcArch.X86:
                        job.setup_libusb_arch = "x86"
                    case MsvcArch.X64:
                        job.setup_libusb_arch = "x64"
        case SdlPlatform.Linux:
            if spec.name.startswith("Ubuntu"):
                assert spec.os.value.startswith("ubuntu-")
                job.apt_packages.extend((
                    "gnome-desktop-testing",
                    "libasound2-dev",
                    "libpulse-dev",
                    "libaudio-dev",
                    "libjack-dev",
                    "libsndio-dev",
                    "libusb-1.0-0-dev",
                    "libx11-dev",
                    "libxext-dev",
                    "libxrandr-dev",
                    "libxcursor-dev",
                    "libxfixes-dev",
                    "libxi-dev",
                    "libxss-dev",
                    "libwayland-dev",
                    "libxkbcommon-dev",
                    "libdrm-dev",
                    "libgbm-dev",
                    "libgl1-mesa-dev",
                    "libgles2-mesa-dev",
                    "libegl1-mesa-dev",
                    "libdbus-1-dev",
                    "libibus-1.0-dev",
                    "libudev-dev",
                    "fcitx-libs-dev",
                ))
                ubuntu_year, ubuntu_month = [int(v) for v in spec.os.value.removeprefix("ubuntu-").split(".", 1)]
                if ubuntu_year >= 22:
                    job.apt_packages.extend(("libpipewire-0.3-dev", "libdecor-0-dev"))
                job.apt_packages.extend((
                    "libunwind-dev",  # For SDL_test memory tracking
                ))
            if trackmem_symbol_names:
                # older libunwind is slow
                job.cmake_arguments.append("-DSDLTEST_TIMEOUT_MULTIPLIER=2")
            job.shared_lib = SharedLibType.SO_0
            job.static_lib = StaticLibType.A
            fpic = True
        case SdlPlatform.Ios | SdlPlatform.Tvos:
            job.brew_packages.extend([
                "ninja",
            ])
            job.run_tests = False
            job.test_pkg_config = False
            job.shared_lib = SharedLibType.DYLIB
            job.static_lib = StaticLibType.A
            match spec.platform:
                case SdlPlatform.Ios:
                    if spec.xcode:
                        job.xcode_sdk = "iphoneos"
                        job.xcode_target = "Static Library-iOS"
                    job.cmake_arguments.extend([
                        "-DCMAKE_SYSTEM_NAME=iOS",
                        "-DCMAKE_OSX_ARCHITECTURES=\"arm64\"",
                        "-DCMAKE_OSX_DEPLOYMENT_TARGET=9.0",
                    ])
                case SdlPlatform.Tvos:
                    if spec.xcode:
                        job.xcode_sdk = "appletvos"
                        job.xcode_target = "Static Library-tvOS"
                    job.cmake_arguments.extend([
                        "-DCMAKE_SYSTEM_NAME=tvOS",
                        "-DCMAKE_OSX_ARCHITECTURES=\"arm64\"",
                        "-DCMAKE_OSX_DEPLOYMENT_TARGET=9.0",
                    ])
        case SdlPlatform.MacOS:
            osx_arch = ";".join(e.value for e in spec.apple_archs) if spec.apple_archs else "arm64"
            job.cmake_arguments.extend((
                f"'-DCMAKE_OSX_ARCHITECTURES={osx_arch}'",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.11",
            ))
            job.build_autotools_tests = False
            job.shared_lib = SharedLibType.DYLIB
            job.static_lib = StaticLibType.A
            job.apt_packages = []
            job.brew_packages.extend((
                "autoconf",
                "ninja",
            ))
            if spec.xcode:
                job.xcode_sdk = "macosx"
        case SdlPlatform.Android:
            job.android_mk = spec.android_mk
            job.run_tests = False
            job.shared_lib = SharedLibType.SO
            job.static_lib = StaticLibType.A
            if spec.android_mk or not spec.no_cmake:
                job.android_ndk = True
            if spec.android_mk:
                job.apt_packages = []
            if not spec.no_cmake:
                job.cmake_arguments.extend((
                    f"-DANDROID_PLATFORM={spec.android_platform}",
                    f"-DANDROID_ABI={spec.android_abi}",
                ))
                job.cmake_toolchain_file = "${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"
                job.cc = f"${{ANDROID_NDK_HOME}}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target={spec.android_arch}-none-linux-androideabi{spec.android_platform}"

                job.android_apks = [
                    "testaudiorecording-apk",
                    "testautomation-apk",
                    "testcontroller-apk",
                    "testmultiaudio-apk",
                    "testsprite-apk",
                ]
        case SdlPlatform.Emscripten:
            job.run_tests = False
            job.shared = False
            job.cmake_config_emulator = "emcmake"
            job.cmake_build_type = "Debug"
            job.test_pkg_config = False
            job.cmake_arguments.extend((
                "-DSDLTEST_BROWSER=chrome",
                "-DSDLTEST_TIMEOUT_MULTIPLIER=4",
                "-DSDLTEST_CHROME_BINARY=${CHROME_BINARY}",
            ))
            job.static_lib = StaticLibType.A
        case SdlPlatform.Ps2:
            build_parallel = False
            job.shared = False
            job.sudo = ""
            job.apt_packages = []
            job.apk_packages = ["cmake", "gmp", "mpc1", "mpfr4", "ninja", "pkgconf", "git", ]
            job.cmake_toolchain_file = "${PS2DEV}/ps2sdk/ps2dev.cmake"
            job.run_tests = False
            job.shared = False
            job.cc = "mips64r5900el-ps2-elf-gcc"
            job.ldflags = ["-L${PS2DEV}/ps2sdk/ee/lib", "-L${PS2DEV}/gsKit/lib", "-L${PS2DEV}/ps2sdk/ports/lib", ]
            job.static_lib = StaticLibType.A
        case SdlPlatform.Psp:
            build_parallel = False
            job.sudo = ""
            job.apt_packages = []
            job.apk_packages = ["cmake", "gmp", "mpc1", "mpfr4", "ninja", "pkgconf", ]
            job.cmake_toolchain_file = "${PSPDEV}/psp/share/pspdev.cmake"
            job.run_tests = False
            job.shared = False
            job.cc = "psp-gcc"
            job.ldflags = ["-L${PSPDEV}/lib", "-L${PSPDEV}/psp/lib", "-L${PSPDEV}/psp/sdk/lib", ]
            job.pollute_directories = ["${PSPDEV}/include", "${PSPDEV}/psp/include", "${PSPDEV}/psp/sdk/include", ]
            job.static_lib = StaticLibType.A
        case SdlPlatform.Vita:
            job.sudo = ""
            job.apt_packages = []
            job.apk_packages = ["cmake", "ninja", "pkgconf", "bash", "tar"]
            job.cmake_toolchain_file = "${VITASDK}/share/vita.toolchain.cmake"
            assert spec.vita_gles is not None
            job.setup_vita_gles_type = {
                VitaGLES.Pib: "pib",
                VitaGLES.Pvr: "pvr",
            }[spec.vita_gles]
            job.cmake_arguments.extend((
                f"-DVIDEO_VITA_PIB={ 'true' if spec.vita_gles == VitaGLES.Pib else 'false' }",
                f"-DVIDEO_VITA_PVR={ 'true' if spec.vita_gles == VitaGLES.Pvr else 'false' }",
                "-DSDL_ARMNEON=ON",
                "-DSDL_ARMSIMD=ON",
                ))
            # Fix vita.toolchain.cmake (https://github.com/vitasdk/vita-toolchain/pull/253)
            job.source_cmd = r"""sed -i -E "s#set\\( PKG_CONFIG_EXECUTABLE \"\\$\\{VITASDK}/bin/arm-vita-eabi-pkg-config\" \\)#set\\( PKG_CONFIG_EXECUTABLE \"${VITASDK}/bin/arm-vita-eabi-pkg-config\" CACHE PATH \"Path of pkg-config executable\" \\)#" ${VITASDK}/share/vita.toolchain.cmake"""
            job.run_tests = False
            job.shared = False
            job.cc = "arm-vita-eabi-gcc"
            job.static_lib = StaticLibType.A
        case SdlPlatform.Haiku:
            fpic = False
            job.build_autotools_tests = False
            job.run_tests = False
            job.cc = "x86_64-unknown-haiku-gcc"
            job.cxx = "x86_64-unknown-haiku-g++"
            job.sudo = ""
            job.cmake_arguments.extend((
                f"-DCMAKE_C_COMPILER={job.cc}",
                f"-DCMAKE_CXX_COMPILER={job.cxx}",
                "-DSDL_UNIX_CONSOLE_BUILD=ON",
            ))
            job.shared_lib = SharedLibType.SO_0
            job.static_lib = StaticLibType.A
        case SdlPlatform.N3ds:
            job.shared = False
            job.apt_packages = ["ninja-build", "binutils"]
            job.run_tests = False
            job.cc_from_cmake = True
            job.cmake_toolchain_file = "${DEVKITPRO}/cmake/3DS.cmake"
            job.static_lib = StaticLibType.A
        case SdlPlatform.Msys2:
            job.shell = "msys2 {0}"
            assert spec.msys2_platform
            job.msys2_msystem = spec.msys2_platform.value
            job.msys2_env = {
                "mingw32": "mingw-w64-i686",
                "mingw64": "mingw-w64-x86_64",
                "clang32": "mingw-w64-clang-i686",
                "clang64": "mingw-w64-clang-x86_64",
                "ucrt64": "mingw-w64-ucrt-x86_64",
            }[spec.msys2_platform.value]
            job.msys2_no_perl = spec.msys2_platform in (Msys2Platform.Mingw32, Msys2Platform.Clang32)
            job.shared_lib = SharedLibType.WIN32
            job.static_lib = StaticLibType.A
        case SdlPlatform.Riscos:
            # FIXME: Enable SDL_WERROR
            job.werror = False
            job.build_autotools_tests = False
            job.apt_packages = ["cmake", "ninja-build"]
            job.test_pkg_config = False
            job.shared = False
            job.run_tests = False
            job.sudo = ""
            job.cmake_arguments.extend((
                "-DRISCOS:BOOL=ON",
                "-DCMAKE_DISABLE_PRECOMPILE_HEADERS:BOOL=ON",
                "-DSDL_GCC_ATOMICS:BOOL=OFF",
            ))
            job.cmake_toolchain_file = "/home/riscos/env/toolchain-riscos.cmake"
            job.static_lib = StaticLibType.A
        case SdlPlatform.FreeBSD | SdlPlatform.NetBSD:
            job.build_autotools_tests = False
            job.cpactions = True
            job.no_cmake = True
            job.run_tests = False
            job.apt_packages = []
            job.shared_lib = SharedLibType.SO_0
            job.static_lib = StaticLibType.A
            match spec.platform:
                case SdlPlatform.FreeBSD:
                    job.cpactions_os = "freebsd"
                    job.cpactions_version = "13.3"
                    job.cpactions_arch = "x86-64"
                    job.cpactions_setup_cmd = "sudo pkg update"
                    job.cpactions_install_cmd = "sudo pkg install -y cmake ninja pkgconf libXcursor libXext libXinerama libXi libXfixes libXrandr libXScrnSaver libXxf86vm wayland wayland-protocols libxkbcommon mesa-libs libglvnd evdev-proto libinotify alsa-lib jackit pipewire pulseaudio sndio dbus zh-fcitx ibus libudev-devd"
                    job.cmake_arguments.extend((
                        "-DSDL_CHECK_REQUIRED_INCLUDES=/usr/local/include",
                        "-DSDL_CHECK_REQUIRED_LINK_OPTIONS=-L/usr/local/lib",
                    ))
                case SdlPlatform.NetBSD:
                    job.cpactions_os = "netbsd"
                    job.cpactions_version = "10.0"
                    job.cpactions_arch = "x86-64"
                    job.cpactions_setup_cmd = "export PATH=\"/usr/pkg/sbin:/usr/pkg/bin:/sbin:$PATH\"; export PKG_CONFIG_PATH=\"/usr/pkg/lib/pkgconfig\";export PKG_PATH=\"https://cdn.netBSD.org/pub/pkgsrc/packages/NetBSD/$(uname -p)/$(uname -r|cut -f \"1 2\" -d.)/All/\";echo \"PKG_PATH=$PKG_PATH\";echo \"uname -a -> \"$(uname -a)\"\";sudo -E sysctl -w security.pax.aslr.enabled=0;sudo -E sysctl -w security.pax.aslr.global=0;sudo -E pkgin clean;sudo -E pkgin update"
                    job.cpactions_install_cmd = "sudo -E pkgin -y install cmake dbus pkgconf ninja-build pulseaudio libxkbcommon wayland wayland-protocols libinotify libusb1"
        case SdlPlatform.Watcom:
            match spec.watcom_platform:
                case WatcomPlatform.OS2:
                    job.watcom_makefile = "Makefile.os2"
                    job.run_tests = False
                case WatcomPlatform.Windows:
                    job.watcom_makefile = "Makefile.w32"
                    job.run_tests = True
                case _:
                    raise ValueError(f"Unsupported watcom_platform=${spec.watcom_platform}")
        case _:
            raise ValueError(f"Unsupported platform={spec.platform}")

    if "ubuntu" in spec.name.lower():
        job.check_sources = True

    if not build_parallel:
        job.cmake_build_arguments.append("-j1")
    if job.cflags:
        job.cmake_arguments.append(f"-DCMAKE_C_FLAGS={my_shlex_join(job.cflags)}")
    if job.cxxflags:
        job.cmake_arguments.append(f"-DCMAKE_CXX_FLAGS={my_shlex_join(job.cxxflags)}")
    if job.ldflags:
        job.cmake_arguments.append(f"-DCMAKE_SHARED_LINKER_FLAGS={my_shlex_join(job.ldflags)}")
        job.cmake_arguments.append(f"-DCMAKE_EXE_LINKER_FLAGS={my_shlex_join(job.ldflags)}")
    job.pretest_cmd = "\n".join(pretest_cmd)
    def tf(b):
        return "ON" if b else "OFF"

    if fpic is not None:
        job.cmake_arguments.append(f"-DCMAKE_POSITION_INDEPENDENT_CODE={tf(fpic)}")

    if job.no_cmake:
        job.cmake_arguments = []

    return job


def spec_to_platform(spec: JobSpec, key: str, enable_artifacts: bool, trackmem_symbol_names: bool) -> dict[str, str|bool]:
    logger.info("spec=%r", spec)
    job = spec_to_job(spec, key=key, trackmem_symbol_names=trackmem_symbol_names)
    logger.info("job=%r", job)
    platform = job.to_workflow(enable_artifacts=enable_artifacts)
    logger.info("platform=%r", platform)
    return platform


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--github-variable-prefix", default="platforms")
    parser.add_argument("--github-ci", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--commit-message-file")
    parser.add_argument("--no-artifact", dest="enable_artifacts", action="store_false")
    parser.add_argument("--trackmem-symbol-names", dest="trackmem_symbol_names", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    remaining_keys = set(JOB_SPECS.keys())

    all_level_keys = (
        # Level 1
        (
            "haiku",
        ),
    )

    filters = []
    if args.commit_message_file:
        with open(args.commit_message_file, "r") as f:
            commit_message = f.read()
            for m in re.finditer(r"\[sdl-ci-filter (.*)]", commit_message, flags=re.M):
                filters.append(m.group(1).strip(" \t\n\r\t'\""))

            if re.search(r"\[sdl-ci-artifacts?]", commit_message, flags=re.M):
                args.enable_artifacts = True

            if re.search(r"\[sdl-ci-(full-)?trackmem(-symbol-names)?]", commit_message, flags=re.M):
                args.trackmem_symbol_names = True

    if not filters:
        filters.append("*")

    logger.info("filters: %r", filters)

    all_level_platforms = {}

    all_platforms = {key: spec_to_platform(spec, key=key, enable_artifacts=args.enable_artifacts, trackmem_symbol_names=args.trackmem_symbol_names) for key, spec in JOB_SPECS.items()}

    for level_i, level_keys in enumerate(all_level_keys, 1):
        level_key = f"level{level_i}"
        logger.info("Level %d: keys=%r", level_i, level_keys)
        assert all(k in remaining_keys for k in level_keys)
        level_platforms = tuple(all_platforms[key] for key in level_keys)
        remaining_keys.difference_update(level_keys)
        all_level_platforms[level_key] = level_platforms
        logger.info("=" * 80)

    logger.info("Keys before filter: %r", remaining_keys)

    filtered_remaining_keys = set()
    for filter in filters:
        filtered_remaining_keys.update(fnmatch.filter(remaining_keys, filter))

    logger.info("Keys after filter: %r", filtered_remaining_keys)

    remaining_keys = filtered_remaining_keys

    logger.info("Remaining: %r", remaining_keys)
    all_level_platforms["others"] = tuple(all_platforms[key] for key in remaining_keys)

    if args.github_ci:
        for level, platforms in all_level_platforms.items():
            platforms_json = json.dumps(platforms)
            txt = f"{args.github_variable_prefix}-{level}={platforms_json}"
            logger.info("%s", txt)
            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(txt)
                    f.write("\n")
            else:
                logger.warning("GITHUB_OUTPUT not defined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
