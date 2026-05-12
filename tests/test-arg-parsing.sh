#!/bin/bash
# Test that command-line argument parsing handles missing values gracefully
# (exits with code 1, not segfault 139)

WHISPER_STREAM="${1:-../build/bin/whisper-stream}"

failed=0

test_missing_arg() {
    local flag=$1
    local output
    local exit_code

    output=$("$WHISPER_STREAM" "$flag" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 1 ]; then
        echo "PASS: $flag (exit code 1)"
    elif [ $exit_code -eq 139 ]; then
        echo "FAIL: $flag caused segfault (exit code 139)"
        failed=1
    else
        echo "FAIL: $flag unexpected exit code $exit_code"
        failed=1
    fi

    # Also verify error message mentions the flag
    if ! echo "$output" | grep -q "requires an argument"; then
        echo "WARN: $flag did not show expected error message"
    fi
}

echo "Testing whisper-stream argument parsing..."
echo

# Test all flags that require arguments
test_missing_arg "-l"
test_missing_arg "-m"
test_missing_arg "-t"
test_missing_arg "-f"
test_missing_arg "-c"
test_missing_arg "--language"
test_missing_arg "--model"
test_missing_arg "--threads"

echo
if [ $failed -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
