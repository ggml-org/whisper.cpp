ignored_dirs = %w[
  .devops
  examples/wchess/wchess.wasm
  examples/whisper.android
  examples/whisper.android.java
  examples/whisper.objc
  examples/whisper.swiftui
  grammars
  models
  samples
  scripts
]
ignored_files = %w[
  AUTHORS
  Makefile
  README.md
  README_sycl.md
  .gitignore
  .gitmodules
  whisper.nvim
  twitch.sh
  yt-wsp.sh
]

EXTSOURCES =
  `git ls-files -z ../..`.split("\x0")
    .select {|file|
      basename = File.basename(file)

      ignored_dirs.all? {|dir| !file.start_with?("../../#{dir}")} &&
        !ignored_files.include?(basename) &&
        (file.start_with?("../..") || file.start_with?("../javascript")) &&
        (!file.start_with?("../../.github/") || basename == "bindings-ruby.yml")
    }
