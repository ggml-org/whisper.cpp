project('whisper.cpp', ['cpp', 'c'], version: '1.5.0')

cc = meson.get_compiler('c')

if build_machine.system() == 'linux'
    add_project_arguments('-D_GNU_SOURCE', language: 'c')
endif

math = cc.find_library('m', required: true)
threads = dependency('threads')

if not get_option('no_avx')
    add_project_arguments('-mavx', language: 'c')
endif

if not get_option('no_avx2')
    add_project_arguments('-mavx2', language: 'c')
endif

if not get_option('no_fma')
    add_project_arguments('-mfma', language: 'c')
endif

if not get_option('no_f16C')
    add_project_arguments('-mf16c', language: 'c')
endif

ggml_lib = library(
    'ggml',
    'ggml.c',
    'ggml-alloc.c',
    'ggml-backend.c',
    'ggml-quants.c',
    dependencies: [math, threads],
)
ggml_dep = declare_dependency(
    dependencies: [math, threads],
    include_directories: '.',
    link_with: ggml_lib,
)

whisper_lib = library(
    'whisper',
    'whisper.cpp',
    dependencies: [ggml_dep],
)
whisper_dep = declare_dependency(
    include_directories: '.',
    dependencies: [ggml_dep],
    link_with: whisper_lib,
)

if get_option('build_examples')
    subdir('examples')
endif
