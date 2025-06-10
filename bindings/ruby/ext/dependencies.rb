require "tsort"

class Dependencies
  include TSort

  def initialize(cmake, options)
    @cmake = cmake
    @options = options
    @static_lib_shape = nil
    @nodes = {}
    @graph = Hash.new {|h, k| h[k] = []}

    generate_dot
    parse_dot
  end

  def libs
    tsort.filter_map {|node|
      label, shape = @nodes[node]
      if shape == @static_lib_shape
        label.gsub(/\\n\([^)]+\)/, '')
      else
        nil
      end
    }.reverse.collect {|lib| "lib#{lib}.a"}
  end

  def to_s
    libs.join(" ")
  end

  private

  def dot_path
    File.join(__dir__, "build", "whisper.cpp.dot")
  end

  def generate_dot
    args = ["-S", "sources", "-B", "build", "--graphviz", dot_path, "-D", "BUILD_SHARED_LIBS=OFF"]
    args << @options.to_s unless @options.to_s.empty?
    system @cmake, *args, exception: true
  end

  def parse_dot
    File.open(dot_path).each_line do |line|
      case line
      when /\[\s*label\s*=\s*"Static Library"\s*,\s*shape\s*=\s*(?<shape>\w+)\s*\]/
        @static_lib_shape = $~[:shape]
      when /\A\s*"(?<node>\w+)"\s*\[\s*label\s*=\s*"(?<label>\S+)"\s*,\s*shape\s*=\s*(?<shape>\w+)\s*\]\s*;\s*\z/
        node = $~[:node]
        label = $~[:label]
        shape = $~[:shape]
        @nodes[node] = [label, shape]
      when /\A\s*"(?<depender>\w+)"\s*->\s*"(?<dependee>\w+)"/
        depender = $~[:depender]
        dependee = $~[:dependee]
        @graph[depender] << dependee
      end
    end
  end

  def tsort_each_node
    @nodes.each_key do |node|
      yield node
    end
  end

  def tsort_each_child(node)
    @graph[node].each do |child|
      yield child
    end
  end
end
