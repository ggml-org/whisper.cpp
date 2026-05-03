require "pathname"

root = Pathname("..")/".."
ignored_dirs = %w[
  .devops
  .github
  ci
  examples/addon.node
  examples/bench.wasm
  examples/command
  examples/command.wasm
  examples/lsp
  examples/main
  examples/python
  examples/stream
  examples/stream.wasm
  examples/sycl
  examples/talk-llama
  examples/wchess
  examples/whisper.android
  examples/whisper.android.java
  examples/whisper.nvim
  examples/whisper.objc
  examples/whisper.swiftui
  examples/whisper.wasm
  grammars
  models
  samples
  scripts
  tests
].collect {|dir| root/dir}
ignored_files = %w[
  AUTHORS
  Makefile
  .gitignore
  .gitmodules
  .dockerignore
]
ignored_exts = %w[
  .yml
  .sh
  .md
  .py
  .js
  .nvim
]

EXTSOURCES =
  `git ls-files -z #{root}`.split("\x0")
    .collect {|file| Pathname(file)}
    .reject {|file|
      ignored_exts.include?(file.extname) ||
        ignored_files.include?(file.basename.to_path) ||
        ignored_dirs.any? {|dir| file.descend.any? {|desc| desc == dir}} ||
        (file.descend.to_a[1] != root && file != Pathname("..")/"javascript"/"package-tmpl.json")
    }
    .collect(&:to_path)
