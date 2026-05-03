require_relative "dependencies"

class DependenciesForWindows < Dependencies
  def local_libs
    libs.collect {|lib| %|"#{lib_path(lib)}"|}.join(" ")
  end

  private

  def prefix(lib)
    lib.start_with?("ggml") ? "" : "lib"
  end

  def lib_path(lib)
    File.join(__dir__, lib).tr("\\", "/")
  end
end
