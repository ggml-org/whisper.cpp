ignored_dirs = %w[
  .devops
  ci
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
  .dockerignore
  whisper.nvim
  twitch.sh
  yt-wsp.sh
  close-issue.yml
]

EXTSOURCES =
  `git ls-files -z ../..`.split("\x0")
    .reject {|file|
      ignored_dirs.any? {|dir| file.start_with?("../../#{dir}")} ||
        ignored_files.include?(File.basename(file)) ||
        (!file.start_with?("../..") && !file.start_with?("../javascript")) ||
        file.start_with?("../../.github/")
    }
