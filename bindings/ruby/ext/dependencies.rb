require "tsort"

class Dependencies
  def initialize(cmake, options)
    @cmake = cmake
    @options = options

    generate_dot
    @libs = parse_dot
  end

  def to_s
    @libs.join(" ")
  end

  private

  def dot_path
    File.join(__dir__, "build", "whisper.cpp.dot")
  end

  def generate_dot
    system @cmake, "-S", "sources", "-B", "build", "--graphviz", dot_path, "-D", "BUILD_SHARED_LIBS=OFF", @options.to_s, exception: true
  end

  def parse_dot
    static_lib_shape = nil
    nodes = {}
    depends = Hash.new {|h, k| h[k] = []}

    class << depends
      include TSort
      alias tsort_each_node each_key
      def tsort_each_child(node, &block)
        fetch(node, []).each(&block)
      end
    end

    File.open(dot_path).each_line do |line|
      case line
      when /\[\s*label\s*=\s*"Static Library"\s*,\s*shape\s*=\s*(?<shape>\w+)\s*\]/
        static_lib_shape = $~[:shape]
      when /\A\s*"(?<node>\w+)"\s*\[\s*label\s*=\s*"(?<label>\S+)"\s*,\s*shape\s*=\s*(?<shape>\w+)\s*\]\s*;\s*\z/
        node = $~[:node]
        label = $~[:label]
        shape = $~[:shape]
        nodes[node] = [label, shape]
      when /\A\s*"(?<depender>\w+)"\s*->\s*"(?<dependee>\w+)"/
        depender = $~[:depender]
        dependee = $~[:dependee]
        depends[depender] ||= []
        depends[depender] << dependee
      end
    end
    depends.tsort.filter_map {|node|
      label, shape = nodes[node]
      shape == static_lib_shape ? label : nil
    }.collect {|lib| "lib#{lib}.a"}
      .reverse
  end
end
