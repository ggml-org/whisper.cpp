# Syncing `third_party/whisper.cpp` with upstream

`third_party/whisper.cpp` is a **git subtree** of
[ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), pinned at the
tag recorded in `third_party/whisper.cpp/UPSTREAM_PIN`. Divergence policy:
*upstream-tracking, minimally divergent* — every deliberate delta is listed in
`third_party/whisper.cpp/PATCHES.md`, and the `whisper divergence guard` CI
job fails any PR whose subtree differs from the pin outside that manifest.

## Pulling a new upstream release

```sh
git remote add upstream https://github.com/ggml-org/whisper.cpp.git 2>/dev/null
git fetch upstream --tags

git checkout -b feat/whisper-sync-vX.Y.Z master
git subtree pull --prefix third_party/whisper.cpp upstream vX.Y.Z --squash
```

`git subtree pull` performs a real 3-way merge against the previous squashed
snapshot, so **conflicts appear only in the files we patch** (see PATCHES.md —
typically `src/whisper.cpp`, `src/CMakeLists.txt`, `CMakeLists.txt`,
`include/whisper.h`). Resolve them keeping both upstream's changes and the
QVAC delta; the `QVAC` comment markers show which side is ours.

## Post-pull checklist

1. `echo vX.Y.Z > third_party/whisper.cpp/UPSTREAM_PIN`
2. Re-verify every PATCHES.md row still applies (upstream may have absorbed or
   moved things — e.g. `WHISPER_BUILD_PARAKEET` is an upstreaming candidate
   that may land upstream eventually). Update the manifest if the delta
   changed shape; delete rows upstream made redundant.
3. `ggml/` inside the subtree must remain byte-identical to the new upstream
   tag — **never** resolve a conflict by editing vendored ggml. ggml changes
   go to [qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) `speech`.
4. Local sanity: umbrella configure/build + `ctest -LE 'gpu|perf'` + the
   whisper-cli smoke (see `.github/workflows/whisper-ci.yml` for the exact
   commands).
5. Open the PR and let the full matrix run (whisper CI, engine lanes,
   windows/android/ios cross lanes, divergence guard). The guard re-derives
   the allowed-file set from PATCHES.md, so a stale manifest fails loudly.
6. Coordinate the registry pin bump (the `whisper-cpp` port tracks
   `third_party/whisper.cpp`).

## Rules of thumb

- Never hand-edit `third_party/whisper.cpp` outside a sync PR or a PR that
  updates PATCHES.md in the same change.
- Keep the QVAC delta small: prefer `engines/whisper` glue, upstream PRs, or
  qvac-ext-ggml over growing the manifest.
- One sync PR per upstream release; don't batch unrelated work into it.
