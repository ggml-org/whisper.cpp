class Options
  def initialize
    @options = {}
    @pending_options = []
    @ignored_options = []

    configure
  end

  def help
    @options
      .collect_concat {|name, (type, value)|
        option = option_name(name)
        if type == :bool
          ["--enable-#{option}", "--disable-#{option}"]
        else
          "--#{option}=#{type.upcase}"
        end
      }
      .join($/)
  end

  def to_s
    @options
      .reject {|name, (type, value)| value.nil?}
      .collect {|name, (type, value)| "-D #{name}=#{value == true ? "ON" : value == false ? "OFF" : value.shellescape}"}
      .join(" ")
  end

  def cmake_options
    return @cmake_options if @cmake_options

    output = nil
    Dir.chdir __dir__ do
      output = `cmake -S sources -B build -L`
    end
    started = false
    @cmake_options = output.lines.filter_map {|line|
      if line.chomp == "-- Cache values"
        started = true
        next
      end
      next unless started
      option, value = line.chomp.split("=", 2)
      name, type = option.split(":", 2)
      [name, type, value]
    }
  end

  def missing_options
    cmake_options.collect {|name, type, value| name} -
      @options.keys - @pending_options - @ignored_options
  end

  def extra_options
    @options.keys + @pending_options + @ignored_options -
      cmake_options.collect {|name, type, value| name}
  end

  private

  def configure
    filepath "ACCELERATE_FRAMEWORK"
    ignored "BUILD_SHARED_LIBS"
    ignored "BUILD_TESTING"
    ignored "CMAKE_BUILD_TYPE"
    ignored "CMAKE_INSTALL_PREFIX"
    string "CMAKE_OSX_ARCHITECTURES"
    ignored "CMAKE_OSX_DEPLOYMENT_TARGET"
    string "CMAKE_OSX_SYSROOT"
    filepath "FOUNDATION_LIBRARY"
    bool "GGML_ACCELERATE"
    bool "GGML_ALL_WARNINGS_3RD_PARTY"
    bool "GGML_AMX_BF16"
    bool "GGML_AMX_INT8"
    bool "GGML_AMX_TILE"
    bool "GGML_AVX"
    bool "GGML_AVX2"
    bool "GGML_AVX512"
    bool "GGML_AVX512_BF16"
    bool "GGML_AVX512_VBMI"
    bool "GGML_AVX512_VNNI"
    bool "GGML_AVX_VNNI"
    ignored "GGML_BACKEND_DL"
    ignored "GGML_BIN_INSTALL_DIR"
    bool "GGML_BLAS"
    string "GGML_BLAS_VENDOR"
    bool "GGML_BMI2"
    ignored "GGML_BUILD_EXAMPLES"
    ignored "GGML_BUILD_TESTS"
    bool "GGML_CCACHE"
    filepath "GGML_CCACHE_FOUND"
    bool "GGML_CPU"
    bool "GGML_CPU_AARCH64"
    ignored "GGML_CPU_ALL_VARIANTS"
    string "GGML_CPU_ARM_ARCH"
    bool "GGML_CPU_HBM"
    bool "GGML_CPU_KLEIDIAI"
    string "GGML_CPU_POWERPC_CPUTYPE"
    bool "GGML_CUDA"
    string "GGML_CUDA_COMPRESSION_MODE"
    bool "GGML_CUDA_F16"
    bool "GGML_CUDA_FA"
    bool "GGML_CUDA_FA_ALL_QUANTS"
    bool "GGML_CUDA_FORCE_CUBLAS"
    bool "GGML_CUDA_FORCE_MMQ"
    ignored "GGML_CUDA_GRAPHS"
    bool "GGML_CUDA_NO_PEER_COPY"
    bool "GGML_CUDA_NO_VMM"
    string "GGML_CUDA_PEER_MAX_BATCH_SIZE"
    bool "GGML_F16C"
    bool "GGML_FMA"
    bool "GGML_GPROF"
    bool "GGML_HIP"
    bool "GGML_HIP_GRAPHS"
    bool "GGML_HIP_NO_VMM"
    bool "GGML_HIP_ROCWMMA_FATTN"
    ignored "GGML_INCLUDE_INSTALL_DIR"
    bool "GGML_KOMPUTE"
    bool "GGML_LASX"
    ignored "GGML_LIB_INSTALL_DIR"
    ignored "GGML_LLAMAFILE"
    bool "GGML_LSX"
    bool "GGML_LTO"
    bool "GGML_METAL"
    bool "GGML_METAL_EMBED_LIBRARY"
    string "GGML_METAL_MACOSX_VERSION_MIN"
    bool "GGML_METAL_NDEBUG"
    bool "GGML_METAL_SHADER_DEBUG"
    string "GGML_METAL_STD"
    bool "GGML_METAL_USE_BF16"
    bool "GGML_MUSA"
    bool "GGML_NATIVE"
    bool "GGML_OPENCL"
    bool "GGML_OPENCL_EMBED_KERNELS"
    bool "GGML_OPENCL_PROFILING"
    string "GGML_OPENCL_TARGET_VERSION"
    bool "GGML_OPENCL_USE_ADRENO_KERNELS"
    bool "GGML_OPENMP"
    bool "GGML_RPC"
    bool "GGML_RVV"
    bool "GGML_RV_ZFH"
    pending "GGML_SCCACHE_FOUND"
    string "GGML_SCHED_MAX_COPIES"
    bool "GGML_SSE42"
    ignored "GGML_STATIC"
    bool "GGML_SYCL"
    string "GGML_SYCL_DEVICE_ARCH"
    bool "GGML_SYCL_F16"
    bool "GGML_SYCL_GRAPH"
    string "GGML_SYCL_TARGET"
    bool "GGML_SYCL_DNN"
    bool "GGML_VULKAN"
    bool "GGML_VULKAN_CHECK_RESULTS"
    bool "GGML_VULKAN_DEBUG"
    bool "GGML_VULKAN_MEMORY_DEBUG"
    bool "GGML_VULKAN_PERF"
    ignored "GGML_VULKAN_RUN_TESTS"
    filepath "GGML_VULKAN_SHADERS_GEN_TOOLCHAIN"
    bool "GGML_VULKAN_SHADER_DEBUG_INFO"
    pending "GGML_VULKAN_VALIDATE"
    bool "GGML_VXE"
    filepath "GIT_EXE"
    filepath "MATH_LIBRARY"
    filepath "METALKIT_FRAMEWORK"
    filepath "METAL_FRAMEWORK"
    bool "WHISPER_ALL_WARNINGS"
    bool "WHISPER_ALL_WARNINGS_3RD_PARTY"
    ignored "WHISPER_BIN_INSTALL_DIR"
    ignored "WHISPER_BUILD_EXAMPLES"
    ignored "WHISPER_BUILD_SERVER"
    ignored"WHISPER_BUILD_TESTS"
    bool "WHISPER_COREML"
    bool "WHISPER_COREML_ALLOW_FALLBACK"
    ignored "WHISPER_CURL"
    bool "WHISPER_FATAL_WARNINGS"
    ignored "WHISPER_FFMPEG"
    ignored "WHISPER_INCLUDE_INSTALL_DIR"
    ignored "WHISPER_LIB_INSTALL_DIR"
    bool "WHISPER_OPENVINO"
    bool "WHISPER_SANITIZE_ADDRESS"
    bool "WHISPER_SANITIZE_THREAD"
    bool "WHISPER_SANITIZE_UNDEFINED"
    ignored "WHISPER_SDL2"
    pending "WHISPER_USE_SYSTEM_GGML"
  end

  def option_name(name)
    name.downcase.gsub("_", "-")
  end

  def bool(name)
    option = option_name(name)
    value = enable_config(option)
    @options[name] = [:bool, value]
  end

  def string(name, type=:string)
    option = "--#{option_name(name)}"
    value = arg_config(option)
    raise "String expected for #{option}" if value == true || value&.empty?
    @options[name] = [type, value]
  end

  def path(name)
    string(name, :path)
  end

  def filepath(name)
    string(name, :filepath)
  end

  def pending(name)
    @pending_options << name
  end

  def ignored(name)
    @ignored_options << name
  end
end
