# Reference Transcripts

This directory contains canonical reference transcripts used by the benchmark correctness gate.

Expected files:

- `short.txt`
- `medium.txt`
- `long.txt`

How they are used:

- `benchmark/parse_results.py` extracts transcript text from each measured run log.
- Text is normalized (case, punctuation, spacing).
- WER and CER are computed against these reference files.
- `benchmark/bench.sh` enforces correctness thresholds by default:
  - `MAX_WER=0.02`
  - `MAX_CER=0.02`

Notes:

- Keep references fixed once baseline is established.
- If audio inputs change, regenerate references intentionally and document why.
