require "pathname"

root = Pathname("..")/".."
ignored_dirs = %w[
  .devops
  .github
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
].collect {|dir| root/dir}
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
  `git ls-files -z #{root}`.split("\x0")
    .collect {|file| Pathname(file)}
    .reject {|file|
      ignored_dirs.any? {|dir| file.descend.any? {|desc| desc == dir}} ||
        ignored_files.include?(file.basename.to_path) ||
        (file.descend.to_a[1] != root && file.descend.to_a[1] != Pathname("..")/"javascript")
    }
    .collect(&:to_path)
