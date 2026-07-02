#!/bin/bash

# Offline test for whisper-cli's -hf / --hf-file HuggingFace cache resolution.
#
# It seeds a temporary HF hub cache (HF_HUB_CACHE) with the
# models--org--repo/{refs,snapshots} layout that the `hf` CLI / huggingface_hub
# produces, using an existing local `for-tests` model as the payload, then checks:
#   1. `-hf <repo> --hf-file <file>` resolves the cached snapshot and runs (exit 0)
#   2. a missing --hf-file prints the "not found in HF cache" error and exits 3
#   3. `-m <path>` regression: an explicit model path still works unchanged
#   4. bare invocation (no -hf/-m) still uses the models/ggml-base.en.bin default
#   5. (optional) a no-OpenSSL build attempting an https resolve with an empty
#      cache prints the "rebuild with -DWHISPER_OPENSSL=ON" hint and exits non-zero
#
# HF_HUB_OFFLINE=1 forces the resolver to skip the network path (Phase 2), so the
# warm-cache cases resolve deterministically from the seeded cache with no network.
#
# Usage:
#   ./tests/test-hf-resolve.sh
#   WHISPER_CLI=build-ssl/bin/whisper-cli \
#     WHISPER_CLI_NOSSL=build-nossl/bin/whisper-cli ./tests/test-hf-resolve.sh

set -u

cd "$(dirname "$0")/.."

main="${WHISPER_CLI:-./build/bin/whisper-cli}"
main_nossl="${WHISPER_CLI_NOSSL:-}"
sample="samples/jfk.wav"
seed_model="models/for-tests-ggml-base.en.bin"
repo="ggerganov/whisper.cpp"
hf_file="ggml-base.en.bin"

for f in "$main" "$sample" "$seed_model"; do
    if [ ! -e "$f" ]; then
        printf "required fixture not found: %s\n" "$f"
        printf "build whisper-cli and ensure test models/samples are present first.\n"
        exit 1
    fi
done

tmp_cache="$(mktemp -d)"
trap 'rm -rf "$tmp_cache"' EXIT

commit="$(printf '%040d' 1 | tr '0' 'a')"
snapshot_dir="$tmp_cache/models--ggerganov--whisper.cpp/snapshots/$commit"
refs_dir="$tmp_cache/models--ggerganov--whisper.cpp/refs"
mkdir -p "$snapshot_dir" "$refs_dir"
printf '%s' "$commit" > "$refs_dir/main"
cp "$seed_model" "$snapshot_dir/$hf_file"

fail=0

# 1. cache resolution succeeds (offline: network path skipped, falls back to cache)
if HF_HUB_OFFLINE=1 HF_HUB_CACHE="$tmp_cache" "$main" -hf "$repo" --hf-file "$hf_file" -f "$sample" >/tmp/hf_resolve_ok.log 2>&1; then
    if grep -qi "failed to open" /tmp/hf_resolve_ok.log; then
        printf "FAIL: -hf resolved but model failed to open\n"; fail=1
    else
        printf "PASS: -hf %s --hf-file %s resolved from cache (offline, exit 0)\n" "$repo" "$hf_file"
    fi
else
    printf "FAIL: -hf offline resolution exited non-zero\n"; cat /tmp/hf_resolve_ok.log; fail=1
fi

# 2. missing file -> exit 3 with clear error
HF_HUB_OFFLINE=1 HF_HUB_CACHE="$tmp_cache" "$main" -hf "$repo" --hf-file ggml-missing.bin -f "$sample" >/tmp/hf_resolve_miss.log 2>&1
rc=$?
if [ "$rc" -eq 3 ] && grep -qi "not found in HF cache" /tmp/hf_resolve_miss.log; then
    printf "PASS: missing --hf-file reports 'not found in HF cache' and exits 3\n"
else
    printf "FAIL: missing --hf-file expected exit 3 + error message, got exit %s\n" "$rc"; fail=1
fi

# 3. -m regression: explicit path still works
if "$main" -m "$seed_model" -f "$sample" >/tmp/hf_resolve_m.log 2>&1; then
    printf "PASS: -m %s still works (exit 0)\n" "$seed_model"
else
    printf "FAIL: -m regression exited non-zero\n"; cat /tmp/hf_resolve_m.log; fail=1
fi

# 4. bare default unchanged: still points at models/ggml-base.en.bin
"$main" -f "$sample" >/tmp/hf_resolve_bare.log 2>&1
if grep -qi "models/ggml-base.en.bin" /tmp/hf_resolve_bare.log; then
    printf "PASS: bare invocation still uses models/ggml-base.en.bin default\n"
else
    printf "FAIL: bare default no longer references models/ggml-base.en.bin\n"; cat /tmp/hf_resolve_bare.log; fail=1
fi

# 5. no-OpenSSL build: an https resolve against an empty cache prints the rebuild
#    hint and exits non-zero. Only runs if a no-OpenSSL binary is provided.
if [ -n "$main_nossl" ] && [ -e "$main_nossl" ]; then
    empty_cache="$(mktemp -d)"
    HF_HUB_CACHE="$empty_cache" "$main_nossl" -hf "$repo" --hf-file "$hf_file" -f "$sample" >/tmp/hf_resolve_nossl.log 2>&1
    rc=$?
    rm -rf "$empty_cache"
    if [ "$rc" -ne 0 ] && grep -qi "rebuild with -DWHISPER_OPENSSL=ON" /tmp/hf_resolve_nossl.log; then
        printf "PASS: no-OpenSSL https attempt prints rebuild hint and exits non-zero\n"
    else
        printf "FAIL: no-OpenSSL https attempt expected rebuild hint + non-zero exit, got exit %s\n" "$rc"
        cat /tmp/hf_resolve_nossl.log; fail=1
    fi
else
    printf "SKIP: no-OpenSSL rebuild-hint check (set WHISPER_CLI_NOSSL to enable)\n"
fi

if [ "$fail" -ne 0 ]; then
    printf "\ntest-hf-resolve: FAILED\n"
    exit 1
fi

printf "\ntest-hf-resolve: all checks passed\n"
