#!/bin/bash
set -euo pipefail

# ====== 設定部分 ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WHISPER_CLI="${WHISPER_CLI:-"$SCRIPT_DIR/build/bin/whisper-cli"}"
MODEL="${MODEL:-"$SCRIPT_DIR/models/ggml-large-v3-turbo.bin"}"
SEGMENT_TIME="${SEGMENT_TIME:-0}"  # 0 のままなら分割せず、環境変数で秒数を上書き
LANG="${WHISPER_LANG:-ja}"
LANG_LOWER=$(printf '%s' "$LANG" | tr '[:upper:]' '[:lower:]')
SAVED_RECORDING=""
BUNDLE_DIR=""
# 録音関連の環境変数（必要に応じて呼び出し前に設定）
#   AUDIO_DEVICE=:0             # ffmpeg avfoundation の音声デバイス指定
#   RECORD_OUTPUT_DIR=$HOME/Desktop  # 録音ファイルと文字起こしの保存先
#   KEEP_RECORDING=1            # 0 にすると録音済みファイルを保存しない
#   RECORD_BASENAME=custom_name # 録音保存時のファイル名（接尾辞は自動で付与）
#   MAX_RECORD_SECONDS=5400     # 録音の自動停止時間（秒）。90分授業を想定したデフォルト
#   RECORD_STATUS_INTERVAL=5    # 録音中ステータスの更新間隔（秒）。0 にすると通知を停止
#   RECORD_SAMPLE_RATE=48000    # 録音時のサンプリングレート。未指定なら 48000。
#   RECORD_CHANNELS=1           # 録音時のチャンネル数。未指定ならモノラル。
#   RECORD_QUEUE_SIZE=32768     # ffmpeg のスレッドキューサイズ。バッファ不足によるプツプツ対策。
#   RECORD_DISABLE_FILTER=0     # 1 にすると録音時の aresample フィルタを無効化。
#   RECORD_FILTER=...           # 録音時の追加フィルタ。未指定時は非同期リサンプルを適用。
#   BEAM_SIZE=8                 # beam search の幅。精度向上のための既定値
#   BEST_OF=8                   # 生成候補数。精度/速度のバランスで調整
#   FAST_MODE=0                # 1 にすると速度優先設定を適用（beam/best-of 縮小など）
#   FAST_BEAM_SIZE=1           # FAST_MODE 時の beam search 幅
#   FAST_BEST_OF=1             # FAST_MODE 時の生成候補数
#   FAST_DISABLE_FALLBACK=1    # FAST_MODE 時に温度フォールバックを無効化（0 で保持）
#   HIGH_PERF_MODE=0           # 1 にすると CPU/GPU を積極利用する高負荷モード
#   PROCESSORS=1               # whisper-cli の --processors（HIGH_PERF_MODE=1 時は未指定なら自動）
#   GPU_LAYERS=                # whisper-cli の --gpu-layers（オフロード層数）。空なら無効
#   HIGH_PERF_GPU_LAYERS=      # HIGH_PERF_MODE=1 かつ GPU_LAYERS 未指定時に用いる推奨層数
#   SEGMENT_WORKERS=           # セグメント並列処理の最大同時実行数（未指定時は自動）
#   FFMPEG_RESAMPLE_THREADS=   # ffmpeg 整形時に利用するスレッド数（HIGH_PERF_MODE=1 で未指定なら自動）
#   PROMPT_HINT=               # whisper-cli に渡す初期プロンプト文字列 (--prompt)
#   PROMPT_FILE=               # whisper-cli に渡すプロンプトファイルパス (--prompt-file)
#   AUTO_JA_PROMPT=1           # ja 言語時に漢字・句読点を促す既定プロンプトを付与（0 で無効化）
#   STABILITY_MODE=auto       # 0:無効 / 1:常時有効 / auto: ja 系言語のみ精度重視の安全策を適用
#   TEMPERATURE=               # --temperature を上書き
#   TEMPERATURE_INC=           # --temperature-inc を上書き
#   ENTROPY_THOLD=             # --entropy-thold を上書き
#   LOGPROB_THOLD=             # --logprob-thold を上書き
#   NO_SPEECH_THOLD=0.58       # --no-speech-thold を上書き
#   MAX_LEN=                   # --max-len を上書き（0/未設定で無効）
#   SPLIT_ON_WORD=             # 1 で --split-on-word を付与（--max-len と併用推奨）
#   DEDUP_CONSECUTIVE=0        # 連続する同一行を除去（必要なら 1 / auto を指定）
#   REDUCE_CHAR_RUNS=0         # 連続文字の畳み込み。1 で有効、auto で ja 系のみ有効
#   REDUCE_CHAR_RUN_MAX=12     # 連続文字を残す最大長（REDUCE_CHAR_RUNS 使用時）
#   DISABLE_FALLBACK=0        # 温度フォールバックを抑止（必要なら 1 / auto を指定）
#   MAX_CONTEXT=auto          # whisper-cli の --max-context。auto は ja 系で 0 を採用
#   ACCURACY_MODE=auto        # 1 で高精度優先（beam/best-of を拡大し速度低下を許容）
#   ACCURACY_BEAM=12          # ACCURACY_MODE 時に用いる beam 幅
#   ACCURACY_BEST_OF=12       # ACCURACY_MODE 時に生成する候補数
MAX_RECORD_SECONDS=${MAX_RECORD_SECONDS:-5400}
RECORD_STATUS_INTERVAL=${RECORD_STATUS_INTERVAL:-5}
RECORD_STATUS_PID=""
RECORD_STOP_KEY_PID=""
RECORD_TIMEOUT_PID=""
RECORD_ACTIVE_PID=""
RECORD_TIMEOUT_FLAG=""
RECORD_STOP_KEY_FLAG=""
RECORD_START_TS=0
FINAL_TIMED_TXT=""

log_record_debug() {
    if [ "${RECORD_DEBUG:-0}" = "0" ]; then
        return
    fi
    printf '[record-debug] %s\n' "$1"
}

# whisper-cli のリソース関連スイッチ（精度は維持しつつ負荷が増えるもの）
#   --threads / -t         : CPU スレッド数。大きくすると計算が速くなるが CPU を占有する。
#   --processors / -p      : 音声を分割し複数ワーカーで同時処理する。CPU/RAM を追加で消費。
#   --flash-attn / -fa     : 対応 GPU で FlashAttention を使用。Metal では描画ドロップが報告されているため既定で無効。
#   --gpu-layers / --ngl   : Metal/OpenCL でエンコード層を GPU にオフロード。VRAM 使用量と帯域が増える。
# HIGH_PERF_MODE=1 を指定すると、これらのスイッチに対して高負荷寄りの既定値を適用する（個別の環境変数で上書き可）。

format_duration() {
    local total="$1"
    if [ "$total" -lt 0 ]; then
        total=0
    fi
    local hours=$(( total / 3600 ))
    local minutes=$(( (total % 3600) / 60 ))
    local seconds=$(( total % 60 ))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

is_pid_alive() {
    local pid="$1"
    if [ -z "$pid" ]; then
        return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi
    if command -v ps >/dev/null 2>&1; then
        local state
        state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
        if [ -n "$state" ]; then
            case "$state" in
                [Zz]*)
                    return 1
                    ;;
            esac
        fi
    fi
    return 0
}

wait_for_pid_exit() {
    local target_pid="$1"
    local timeout_seconds="${2:-5}"
    if [ -z "$target_pid" ]; then
        return 0
    fi
    if ! is_pid_alive "$target_pid"; then
        return 0
    fi
    local start_ts
    start_ts=$(date +%s)
    while is_pid_alive "$target_pid"; do
        local now_ts
        now_ts=$(date +%s)
        if [ $(( now_ts - start_ts )) -ge "$timeout_seconds" ]; then
            return 1
        fi
        sleep 0.2
    done
    return 0
}

start_recording_status_loop() {
    local target_pid="$1"
    if [ "${RECORD_STATUS_INTERVAL:-0}" -le 0 ]; then
        return
    fi
    log_record_debug "status loop start pid=$target_pid"
    RECORD_START_TS=$(date +%s)
    (
        trap '' INT TERM
        while is_pid_alive "$target_pid"; do
            sleep "$RECORD_STATUS_INTERVAL"
            if ! is_pid_alive "$target_pid"; then
                break
            fi
            local now
            now=$(date +%s)
            local elapsed=$(( now - RECORD_START_TS ))
            if [ "$elapsed" -lt 0 ]; then
                elapsed=0
            fi
            printf "\r録音中... 経過 %s" "$(format_duration "$elapsed")"
        done
    ) &
    RECORD_STATUS_PID=$!
}

stop_recording_status_loop() {
    local exit_code="${1:-0}"
    if [ -z "${RECORD_STATUS_PID:-}" ]; then
        return
    fi
    if kill -0 "$RECORD_STATUS_PID" 2>/dev/null; then
        kill "$RECORD_STATUS_PID" 2>/dev/null || true
    fi
    wait "$RECORD_STATUS_PID" 2>/dev/null || true
    RECORD_STATUS_PID=""
    if [ "${RECORD_STATUS_INTERVAL:-0}" -le 0 ]; then
        printf "\n"
        return
    fi
    local end_ts
    end_ts=$(date +%s)
    local elapsed=$(( end_ts - RECORD_START_TS ))
    if [ "$elapsed" -lt 0 ]; then
        elapsed=0
    fi
    local label="録音終了"
    if [ "$exit_code" -ne 0 ]; then
        label="録音が中断されました"
    fi
    printf "\r%-60s\r" ""
    printf "%s（経過 %s）\n" "$label" "$(format_duration "$elapsed")"
}

signal_process_tree() {
    local signal="$1"
    local pid="$2"
    if [ -z "$pid" ]; then
        return
    fi
    if ! is_pid_alive "$pid"; then
        return
    fi
    log_record_debug "signal_process_tree send $signal to pid=$pid"
    # signal the entire process group first to cover helpers that detach from the parent
    kill "-$signal" "-$pid" 2>/dev/null || true
    if command -v pgrep >/dev/null 2>&1; then
        while IFS= read -r child_pid; do
            if [ -n "$child_pid" ] && [ "$child_pid" != "$pid" ]; then
                signal_process_tree "$signal" "$child_pid"
            fi
        done < <(pgrep -P "$pid" 2>/dev/null || true)
    fi
    kill "-$signal" "$pid" 2>/dev/null || true
}

terminate_recording_process() {
    local target_pid="$1"
    if [ -z "$target_pid" ]; then
        return
    fi
    if ! is_pid_alive "$target_pid"; then
        return
    fi
    log_record_debug "terminate_recording_process begin pid=$target_pid"
    signal_process_tree INT "$target_pid"
    if wait_for_pid_exit "$target_pid" 6; then
        log_record_debug "terminate_recording_process pid=$target_pid exited after INT"
        wait "$target_pid" 2>/dev/null || true
        return
    fi
    signal_process_tree TERM "$target_pid"
    if wait_for_pid_exit "$target_pid" 4; then
        log_record_debug "terminate_recording_process pid=$target_pid exited after TERM"
        wait "$target_pid" 2>/dev/null || true
        return
    fi
    log_record_debug "terminate_recording_process escalating to KILL pid=$target_pid"
    signal_process_tree KILL "$target_pid"
    wait "$target_pid" 2>/dev/null || true
}

start_recording_stop_key_loop() {
    local target_pid="$1"
    log_record_debug "stop key loop start pid=$target_pid"
    (
        trap '' INT
        local read_device="/dev/tty"
        if [ ! -t 0 ] && [ ! -t 1 ]; then
            if [ -n "${TTY:-}" ]; then
                read_device="$TTY"
            fi
        fi
        if [ ! -r "$read_device" ]; then
            exit 0
        fi
        while is_pid_alive "$target_pid"; do
            if read -r -s -n 1 key < "$read_device"; then
                case "$key" in
                    q|Q)
                        printf "\n'q' が押されたため録音を停止します...\n"
                        log_record_debug "stop key detected for pid=$target_pid"
                        if [ -n "${RECORD_STOP_KEY_FLAG:-}" ]; then
                            : > "$RECORD_STOP_KEY_FLAG" || true
                        fi
                        if is_pid_alive "$target_pid"; then
                            log_record_debug "stop key terminating pid=$target_pid"
                            terminate_recording_process "$target_pid" || true
                        fi
                        log_record_debug "stop key loop break pid=$target_pid"
                        break
                        ;;
                esac
            else
                log_record_debug "stop key read failed, exiting"
                break
            fi
        done
    ) &
    RECORD_STOP_KEY_PID=$!
}

stop_recording_stop_key_loop() {
    if [ -z "${RECORD_STOP_KEY_PID:-}" ]; then
        return
    fi
    if kill -0 "$RECORD_STOP_KEY_PID" 2>/dev/null; then
        kill "$RECORD_STOP_KEY_PID" 2>/dev/null || true
        if kill -0 "$RECORD_STOP_KEY_PID" 2>/dev/null; then
            kill -KILL "$RECORD_STOP_KEY_PID" 2>/dev/null || true
        fi
    fi
    wait "$RECORD_STOP_KEY_PID" 2>/dev/null || true
    RECORD_STOP_KEY_PID=""
}

start_recording_timeout_guard() {
    local target_pid="$1"
    local limit_seconds="$2"
    local timeout_flag="${RECORD_TIMEOUT_FLAG:-}"
    if [ -z "$target_pid" ] || [ -z "$limit_seconds" ]; then
        return
    fi
    case "$limit_seconds" in
        (''|*[!0-9]*)
            return
            ;;
        (0)
            return
            ;;
    esac
    log_record_debug "timeout guard start pid=$target_pid limit=$limit_seconds"
    (
        trap "log_record_debug 'timeout guard received signal pid=$target_pid'; exit 0" INT TERM
        sleep "$limit_seconds"
        if is_pid_alive "$target_pid"; then
            if [ -n "$timeout_flag" ]; then
                : > "$timeout_flag"
            fi
            printf "\n録音が最大秒数 %s に到達したため停止します...\n" "$limit_seconds"
            log_record_debug "timeout guard terminating pid=$target_pid"
            terminate_recording_process "$target_pid"
        fi
    ) &
    RECORD_TIMEOUT_PID=$!
}

stop_recording_timeout_guard() {
    if [ -z "${RECORD_TIMEOUT_PID:-}" ]; then
        return
    fi
    if is_pid_alive "$RECORD_TIMEOUT_PID"; then
        log_record_debug "timeout guard stop requested pid=$RECORD_TIMEOUT_PID"
        kill "$RECORD_TIMEOUT_PID" 2>/dev/null || true
        sleep 0.05
    fi
    if is_pid_alive "$RECORD_TIMEOUT_PID"; then
        log_record_debug "timeout guard forcing kill pid=$RECORD_TIMEOUT_PID"
        kill -KILL "$RECORD_TIMEOUT_PID" 2>/dev/null || true
    fi
    wait "$RECORD_TIMEOUT_PID" 2>/dev/null || true
    RECORD_TIMEOUT_PID=""
}

generate_timestamped_txt() {
    local json_path="$1"
    local txt_path="$2"
    local offset_seconds="${3:-}"
    if [ ! -f "$json_path" ]; then
        return 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        return 1
    fi
    local tmp_output
    tmp_output="${txt_path}.tmp.$$"
    local offset_arg
    offset_arg="${offset_seconds:-0}"
    if python3 - "$json_path" "$tmp_output" "$offset_arg" <<'PY'
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

json_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
offset_raw = sys.argv[3] if len(sys.argv) > 3 else "0"

def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None

def parse_timecode(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", ".")
        parts = text.split(":")
        if len(parts) != 3:
            return None
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except (TypeError, ValueError):
        return None

def iter_segments(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        segments = payload.get("segments")
        if isinstance(segments, list) and segments:
            for entry in segments:
                if isinstance(entry, dict):
                    yield entry
            return
        transcription = payload.get("transcription")
        if isinstance(transcription, list) and transcription:
            for entry in transcription:
                if isinstance(entry, dict):
                    yield entry
            return
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                yield entry

try:
    data = json.loads(json_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

segments = list(iter_segments(data))

try:
    base_offset = float(offset_raw)
except (TypeError, ValueError):
    base_offset = 0.0

def segment_start_seconds(segment: Dict[str, Any]) -> float:
    start = coerce_float(segment.get("start"))
    if start is not None:
        return max(0.0, start)
    offsets = segment.get("offsets")
    if isinstance(offsets, dict):
        start_ms = coerce_float(offsets.get("from"))
        if start_ms is not None:
            return max(0.0, start_ms / 1000.0)
    timestamps = segment.get("timestamps")
    if isinstance(timestamps, dict):
        start_ts = parse_timecode(timestamps.get("from"))
        if start_ts is not None:
            return max(0.0, start_ts)
    return 0.0

def format_ts(value: float) -> str:
    if not math.isfinite(value):
        safe_value = 0.0
    else:
        safe_value = max(0.0, value)
    total_ms = int(round(safe_value * 1000.0))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) / 1000.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

lines = []
for segment in segments:
    text = str(segment.get("text", "")).strip()
    if not text:
        continue
    start_seconds = segment_start_seconds(segment)
    start_ts = format_ts(base_offset + start_seconds)
    lines.append(f"{{{start_ts}}}\u3000{text}")

if not lines:
    sys.exit(2)

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
    then
        mv "$tmp_output" "$txt_path"
        return 0
    fi
    rm -f "$tmp_output"
    return 1
}

# 高速化設定
HIGH_PERF_MODE="${HIGH_PERF_MODE:-0}"

LOGICAL_CPU_DEFAULT="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
LOGICAL_CPU_MAX="$(sysctl -n hw.logicalcpu_max 2>/dev/null || echo "$LOGICAL_CPU_DEFAULT")"

if [ -z "${THREADS-}" ]; then
    if [ "$HIGH_PERF_MODE" = "1" ]; then
        THREADS="$LOGICAL_CPU_MAX"
    else
        THREADS="$LOGICAL_CPU_DEFAULT"
    fi
fi
case "$THREADS" in
    (''|*[!0-9]*) THREADS="$LOGICAL_CPU_DEFAULT" ;;
    (0) THREADS="$LOGICAL_CPU_DEFAULT" ;;
esac

if [ -z "${PROCESSORS+x}" ]; then
    PROCESSORS_DEFINED=0
else
    PROCESSORS_DEFINED=1
fi
if [ -z "${PROCESSORS-}" ]; then
    if [ "$HIGH_PERF_MODE" = "1" ] && [ "$LOGICAL_CPU_MAX" -ge 8 ]; then
        PROCESSORS=2
    else
        PROCESSORS=1
    fi
fi
case "$PROCESSORS" in
    (''|*[!0-9]*) PROCESSORS=1 ;;
    (0) PROCESSORS=1 ;;
esac

HIGH_PERF_GPU_LAYERS="${HIGH_PERF_GPU_LAYERS:-}"
if [ -z "${GPU_LAYERS-}" ]; then
    if [ "$HIGH_PERF_MODE" = "1" ] && [ -n "$HIGH_PERF_GPU_LAYERS" ]; then
        GPU_LAYERS="$HIGH_PERF_GPU_LAYERS"
    fi
fi
GPU_LAYERS="${GPU_LAYERS:-}"
case "$GPU_LAYERS" in
    (*[!0-9]*) GPU_LAYERS="" ;;
esac

if [ -z "${BEAM_SIZE+x}" ]; then
    BEAM_SIZE_DEFINED=0
else
    BEAM_SIZE_DEFINED=1
fi
if [ -z "${BEST_OF+x}" ]; then
    BEST_OF_DEFINED=0
else
    BEST_OF_DEFINED=1
fi

if [ -z "${FAST_MODE+x}" ]; then
    FAST_MODE_DEFINED=0
else
    FAST_MODE_DEFINED=1
fi
FAST_MODE=${FAST_MODE:-0}
FAST_BEAM_SIZE=${FAST_BEAM_SIZE:-1}
FAST_BEST_OF=${FAST_BEST_OF:-1}
FAST_DISABLE_FALLBACK=${FAST_DISABLE_FALLBACK:-1}

ACCURACY_MODE="${ACCURACY_MODE:-auto}"
ACCURACY_MODE_ACTIVE=0
ACCURACY_BEAM=${ACCURACY_BEAM:-16}
ACCURACY_BEST_OF=${ACCURACY_BEST_OF:-16}
case "$ACCURACY_MODE" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                ACCURACY_MODE_ACTIVE=1
                ;;
        esac
        ;;
    (1|true|TRUE|yes|YES)
        ACCURACY_MODE_ACTIVE=1
        ;;
esac
if [ "$ACCURACY_MODE_ACTIVE" = "1" ] && [ "$FAST_MODE_DEFINED" -eq 0 ]; then
    FAST_MODE=0
fi

if [ "$FAST_MODE" = "1" ]; then
    if [ "$BEAM_SIZE_DEFINED" -eq 0 ]; then
        BEAM_SIZE=$FAST_BEAM_SIZE
    fi
    if [ "$BEST_OF_DEFINED" -eq 0 ]; then
        BEST_OF=$FAST_BEST_OF
    fi
fi

if [ "$ACCURACY_MODE_ACTIVE" = "1" ]; then
    if [ "$BEAM_SIZE_DEFINED" -eq 0 ]; then
        BEAM_SIZE=$ACCURACY_BEAM
    fi
    if [ "$BEST_OF_DEFINED" -eq 0 ]; then
        BEST_OF=$ACCURACY_BEST_OF
    fi
fi

BEAM_SIZE=${BEAM_SIZE:-8}
BEST_OF=${BEST_OF:-8}

# 温度・しきい値などの安定化設定
TEMPERATURE="${TEMPERATURE:-}"
TEMPERATURE_INC="${TEMPERATURE_INC:-}"
ENTROPY_THOLD="${ENTROPY_THOLD:-}"
LOGPROB_THOLD="${LOGPROB_THOLD:-}"
NO_SPEECH_THOLD="${NO_SPEECH_THOLD:-0.58}"
MAX_LEN="${MAX_LEN:-}"
SPLIT_ON_WORD="${SPLIT_ON_WORD:-}"
if [ -z "${MAX_CONTEXT+x}" ]; then
    MAX_CONTEXT="auto"
fi
MAX_CONTEXT_TOKENS=""
case "$MAX_CONTEXT" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                MAX_CONTEXT_TOKENS=0
                ;;
        esac
        ;;
    ('')
        MAX_CONTEXT_TOKENS=""
        ;;
    (*[!0-9-]*)
        echo "警告: MAX_CONTEXT=$MAX_CONTEXT は数値として解釈できません。既定を使用します。" >&2
        MAX_CONTEXT_TOKENS=""
        ;;
    (*)
        MAX_CONTEXT_TOKENS="$MAX_CONTEXT"
        ;;
esac
STABILITY_MODE="${STABILITY_MODE:-auto}"
STABILITY_MODE_ACTIVE=0

case "$STABILITY_MODE" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                STABILITY_MODE_ACTIVE=1
                ;;
        esac
        ;;
    (1|true|TRUE|yes|YES)
        STABILITY_MODE_ACTIVE=1
        ;;
    (*)
        STABILITY_MODE_ACTIVE=0
        ;;
esac

if [ "$STABILITY_MODE_ACTIVE" = "1" ]; then
    if [ -z "$TEMPERATURE" ]; then
        TEMPERATURE="0"
    fi
    if [ -z "$TEMPERATURE_INC" ]; then
        TEMPERATURE_INC="0.2"
    fi
    if [ -z "$ENTROPY_THOLD" ]; then
        ENTROPY_THOLD="1.6"
    fi
    if [ -z "$LOGPROB_THOLD" ]; then
        LOGPROB_THOLD="-1.0"
    fi
    if [ -z "$NO_SPEECH_THOLD" ]; then
        NO_SPEECH_THOLD="0.58"
    fi
    if [ -z "$MAX_LEN" ]; then
        MAX_LEN="80"
    fi
    if [ -z "$SPLIT_ON_WORD" ]; then
        SPLIT_ON_WORD="1"
    fi
    if [ "$BEAM_SIZE_DEFINED" -eq 0 ]; then
        BEAM_SIZE=8
    fi
    if [ "$BEST_OF_DEFINED" -eq 0 ]; then
        BEST_OF=8
    fi
    if [ "$PROCESSORS_DEFINED" -eq 0 ]; then
        PROCESSORS=1
    fi
    echo "安定化モードを適用しました（STABILITY_MODE: ${STABILITY_MODE}）。"
fi

if [ "$ACCURACY_MODE_ACTIVE" = "1" ]; then
    if [ -z "$TEMPERATURE" ]; then
        TEMPERATURE="0"
    fi
    if [ -z "$TEMPERATURE_INC" ]; then
        TEMPERATURE_INC="0.2"
    fi
    if [ -z "$ENTROPY_THOLD" ]; then
        ENTROPY_THOLD="1.6"
    fi
    if [ -z "$LOGPROB_THOLD" ]; then
        LOGPROB_THOLD="-1.0"
    fi
    if [ -z "$NO_SPEECH_THOLD" ]; then
        NO_SPEECH_THOLD="0.58"
    fi
    if [ -z "$MAX_LEN" ]; then
        MAX_LEN="80"
    fi
    if [ -z "$SPLIT_ON_WORD" ]; then
        SPLIT_ON_WORD="1"
    fi
    echo "高精度モードを適用しました（ACCURACY_MODE: ${ACCURACY_MODE}）。"
fi

if [ -z "${DISABLE_FALLBACK+x}" ]; then
    DISABLE_FALLBACK=0
fi
DISABLE_FALLBACK="${DISABLE_FALLBACK:-0}"
DISABLE_FALLBACK_ACTIVE=0
case "$DISABLE_FALLBACK" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                DISABLE_FALLBACK_ACTIVE=1
                ;;
        esac
        ;;
    (1|true|TRUE|yes|YES)
        DISABLE_FALLBACK_ACTIVE=1
        ;;
esac
if [ "$DISABLE_FALLBACK_ACTIVE" = "1" ] && [ -z "$TEMPERATURE_INC" ]; then
    TEMPERATURE_INC="0"
fi
if [ "$DISABLE_FALLBACK_ACTIVE" = "1" ]; then
    echo "温度フォールバックを無効化しました（DISABLE_FALLBACK: ${DISABLE_FALLBACK}）。"
fi

CLI_OPTS=(-t "$THREADS" -p "$PROCESSORS" --suppress-nst -bs "$BEAM_SIZE" -bo "$BEST_OF")
NF_REQUESTED=0
if [ "$FAST_MODE" = "1" ] && [ "$FAST_DISABLE_FALLBACK" = "1" ]; then
    NF_REQUESTED=1
fi
if [ "$DISABLE_FALLBACK_ACTIVE" = "1" ]; then
    NF_REQUESTED=1
fi
if [ "$NF_REQUESTED" = "1" ]; then
    CLI_OPTS+=(-nf)
fi

if [ -n "$MAX_CONTEXT_TOKENS" ]; then
    CLI_OPTS+=(-mc "$MAX_CONTEXT_TOKENS")
    echo "max-context を適用しました: $MAX_CONTEXT_TOKENS"
fi

if [ -n "$TEMPERATURE" ]; then
    CLI_OPTS+=(--temperature "$TEMPERATURE")
fi
if [ -n "$TEMPERATURE_INC" ]; then
    CLI_OPTS+=(--temperature-inc "$TEMPERATURE_INC")
fi
if [ -n "$ENTROPY_THOLD" ]; then
    CLI_OPTS+=(--entropy-thold "$ENTROPY_THOLD")
fi
if [ -n "$LOGPROB_THOLD" ]; then
    CLI_OPTS+=(--logprob-thold "$LOGPROB_THOLD")
fi
if [ -n "$NO_SPEECH_THOLD" ]; then
    CLI_OPTS+=(--no-speech-thold "$NO_SPEECH_THOLD")
fi
case "$MAX_LEN" in
    (''|*[!0-9]*) MAX_LEN="" ;;
esac
if [ -n "$MAX_LEN" ] && [ "$MAX_LEN" -gt 0 ]; then
    CLI_OPTS+=(--max-len "$MAX_LEN")
fi
case "$SPLIT_ON_WORD" in
    (1|true|TRUE|yes|YES)
        CLI_OPTS+=(--split-on-word)
        ;;
esac

# 言語別プロンプト（日本語の漢字・句読点を促す）
PROMPT_HINT="${PROMPT_HINT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
AUTO_JA_PROMPT="${AUTO_JA_PROMPT:-1}"
DEFAULT_JA_PROMPT="以下は漢字かな交じり文で句読点を適切に含めた自然な日本語の書き起こしです。"
DEFAULT_JA_PROMPT_ACTIVE=0

if [ -n "$PROMPT_FILE" ]; then
    if [ -f "$PROMPT_FILE" ]; then
        CLI_OPTS+=(--prompt-file "$PROMPT_FILE")
        echo "プロンプトファイルを適用します: $PROMPT_FILE"
    else
        echo "警告: PROMPT_FILE=$PROMPT_FILE が見つかりません。--prompt は付与されません。" >&2
    fi
elif [ -z "$PROMPT_HINT" ] && [ "$AUTO_JA_PROMPT" = "1" ]; then
    case "$LANG_LOWER" in
        (ja|ja-jp|japanese)
            PROMPT_HINT="$DEFAULT_JA_PROMPT"
            DEFAULT_JA_PROMPT_ACTIVE=1
            ;;
    esac
fi

if [ -z "$PROMPT_FILE" ] && [ -n "$PROMPT_HINT" ]; then
    CLI_OPTS+=(--prompt "$PROMPT_HINT")
    if [ "$DEFAULT_JA_PROMPT_ACTIVE" = "1" ]; then
        echo "日本語向けの既定プロンプトを自動適用しました。"
    else
        echo "環境変数 PROMPT_HINT に基づくプロンプトを適用します。"
    fi
fi

# フラッシュアテンションは Metal 環境で出力が途切れる既知の不具合があるため既定では無効化
if [ "$HIGH_PERF_MODE" = "1" ] && [ -z "${USE_FLASH_ATTENTION+x}" ]; then
    USE_FLASH_ATTENTION=1
fi
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-0}"
if [ "${FORCE_CPU:-0}" = "1" ]; then
    USE_FLASH_ATTENTION=0
fi
if [ "$USE_FLASH_ATTENTION" = "1" ]; then
    CLI_OPTS+=(-fa)
fi

# GPU が不安定な場合は FORCE_CPU=1 を指定して CPU 実行にフォールバック
if [ "${FORCE_CPU:-0}" = "1" ]; then
    CLI_OPTS+=(-ng)
fi

if [ "${FORCE_CPU:-0}" != "1" ] && [ -n "$GPU_LAYERS" ]; then
    CLI_OPTS+=(--gpu-layers "$GPU_LAYERS")
fi

# 進捗表示（デフォルト: 有効）
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
if [ "$SHOW_PROGRESS" = "1" ]; then
    CLI_OPTS+=(-pp)
fi

# オプション: 無音スキップ用VAD（無効化するには USE_VAD=0 を環境変数で指定）
USE_VAD="${USE_VAD:-1}"
if [ "$USE_VAD" = "1" ]; then
    VAD_MODEL="${VAD_MODEL:-"$SCRIPT_DIR/models/for-tests-silero-v5.1.2-ggml.bin"}"
    if [ -f "$VAD_MODEL" ]; then
        CLI_OPTS+=(--vad -vm "$VAD_MODEL" -vt 0.6 -vsd 200 -vmsd 60 -vp 40)
    else
        echo "警告: VAD_MODEL=$VAD_MODEL が見つからないため VAD を無効化します。" >&2
        USE_VAD=0
    fi
fi

FFMPEG_RESAMPLE_THREADS="${FFMPEG_RESAMPLE_THREADS:-}"
FFMPEG_RESAMPLE_ARGS=()
FFMPEG_SEGMENT_ARGS=()
if [ "$HIGH_PERF_MODE" = "1" ] && [ -z "$FFMPEG_RESAMPLE_THREADS" ]; then
    FFMPEG_RESAMPLE_THREADS="$THREADS"
fi
case "$FFMPEG_RESAMPLE_THREADS" in
    (''|*[!0-9]*) FFMPEG_RESAMPLE_THREADS="" ;;
    (0) FFMPEG_RESAMPLE_THREADS="" ;;
esac
if [ -n "$FFMPEG_RESAMPLE_THREADS" ]; then
    FFMPEG_RESAMPLE_ARGS=(-threads "$FFMPEG_RESAMPLE_THREADS")
fi

FFMPEG_SEGMENT_THREADS="${FFMPEG_SEGMENT_THREADS:-$FFMPEG_RESAMPLE_THREADS}"
case "$FFMPEG_SEGMENT_THREADS" in
    (''|*[!0-9]*) FFMPEG_SEGMENT_THREADS="" ;;
    (0) FFMPEG_SEGMENT_THREADS="" ;;
esac
if [ -n "$FFMPEG_SEGMENT_THREADS" ]; then
    FFMPEG_SEGMENT_ARGS=(-threads "$FFMPEG_SEGMENT_THREADS")
fi

# 依存関係チェック
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg が見つかりません。Homebrew でインストール: brew install ffmpeg"
    exit 1
fi
if [ ! -x "$WHISPER_CLI" ]; then
    echo "whisper-cli が見つからないか実行できません: $WHISPER_CLI"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "モデルファイルが見つかりません: $MODEL"
    exit 1
fi
if ! command -v osascript >/dev/null 2>&1; then
    echo "osascript が利用できないため、GUI ダイアログを表示できません。"
    exit 1
fi

# ====== 一時ディレクトリとファイル設定 ======
SLEEP_GUARD="${SLEEP_GUARD:-1}"
if [ "$SLEEP_GUARD" = "1" ] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu -w $$ &
    CAFFEINATE_PID=$!
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/whisper.XXXXXX")"
cleanup() {
    if [ -n "${RECORD_ACTIVE_PID:-}" ]; then
        terminate_recording_process "$RECORD_ACTIVE_PID"
    fi
    stop_recording_timeout_guard || true
    stop_recording_stop_key_loop || true
    if [ -n "${TMP_DIR:-}" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
    if [ -n "${CAFFEINATE_PID:-}" ] && kill -0 "$CAFFEINATE_PID" 2>/dev/null; then
        kill "$CAFFEINATE_PID" 2>/dev/null || true
        wait "$CAFFEINATE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

PROCESSED_WAV="$TMP_DIR/processed.wav"
RAW_RECORDING="$TMP_DIR/raw_record.wav"
RECORD_TIMEOUT_FLAG="$TMP_DIR/record_timeout_triggered"
RECORD_STOP_KEY_FLAG="$TMP_DIR/record_stop_key_triggered"
rm -f "$RECORD_TIMEOUT_FLAG" 2>/dev/null || true
rm -f "$RECORD_STOP_KEY_FLAG" 2>/dev/null || true

# ====== 録音またはファイル選択 ======
RECORD_CHOICE="録音しない"
RECORD_CHOICE=$(osascript \
    -e 'tell application "System Events"' \
    -e 'activate' \
    -e 'display dialog "新しく録音しますか？" buttons {"録音しない","録音する"} default button "録音する"' \
    -e 'return button returned of result' \
    -e 'end tell' 2>/dev/null) || RECORD_CHOICE="録音しない"

if [ "$RECORD_CHOICE" = "録音する" ]; then
    AUDIO_DEVICE="${AUDIO_DEVICE:-":0"}"
    if [ -z "${RECORD_OUTPUT_DIR:-}" ]; then
        RECORD_OUTPUT_DIR=$(osascript -e 'try' \
            -e 'POSIX path of (choose folder with prompt "録音ファイルの保存先フォルダを選んでください")' \
            -e 'on error number -128' \
            -e 'return ""' \
            -e 'end try' 2>/dev/null || true)
        if [ -z "$RECORD_OUTPUT_DIR" ]; then
            RECORD_OUTPUT_DIR="$HOME/Desktop"
        fi
    fi
    if [ -n "$RECORD_OUTPUT_DIR" ]; then
        while [ "${#RECORD_OUTPUT_DIR}" -gt 1 ] && [ "${RECORD_OUTPUT_DIR%/}" != "$RECORD_OUTPUT_DIR" ]; do
            RECORD_OUTPUT_DIR="${RECORD_OUTPUT_DIR%/}"
        done
    fi
    mkdir -p "$RECORD_OUTPUT_DIR"
    timestamp="$(date +%Y%m%d)"
    RECORD_BASENAME="${RECORD_BASENAME:-${timestamp}_whisper_recording_$(date +%H%M%S)}"
    BUNDLE_DIR="$RECORD_OUTPUT_DIR/${RECORD_BASENAME}"
    echo "録音を開始します（終了するには 'q' キー、または ${MAX_RECORD_SECONDS} 秒で自動停止）。デバイス: $AUDIO_DEVICE"
    RECORD_SAMPLE_RATE="${RECORD_SAMPLE_RATE:-48000}"
    case "$RECORD_SAMPLE_RATE" in
        (''|*[!0-9]*) RECORD_SAMPLE_RATE=48000 ;;
    esac
    RECORD_CHANNELS="${RECORD_CHANNELS:-1}"
    case "$RECORD_CHANNELS" in
        (''|*[!0-9]*) RECORD_CHANNELS=1 ;;
    esac
    RECORD_QUEUE_SIZE="${RECORD_QUEUE_SIZE:-32768}"
    case "$RECORD_QUEUE_SIZE" in
        (''|*[!0-9]*) RECORD_QUEUE_SIZE=0 ;;
    esac
    if [ "${RECORD_DISABLE_FILTER:-0}" = "1" ]; then
        RECORD_FILTER=""
    else
        if [ -z "${RECORD_FILTER:-}" ]; then
            RECORD_FILTER="aresample=resampler=soxr:async=1:first_pts=0"
        else
            RECORD_FILTER="$RECORD_FILTER"
        fi
    fi

    RECORD_BACKEND="${RECORD_BACKEND:-auto}"
    RECORD_BACKEND_LOWER=$(printf '%s' "$RECORD_BACKEND" | tr '[:upper:]' '[:lower:]')
    capture_name="ffmpeg"
    capture_cmd=()

    REC_BIN=""
    if command -v rec >/dev/null 2>&1; then
        REC_BIN="$(command -v rec)"
    else
        for candidate in /opt/homebrew/bin/rec /usr/local/bin/rec; do
            if [ -x "$candidate" ]; then
                REC_BIN="$candidate"
                break
            fi
        done
    fi

    use_sox=0
    if [ "$RECORD_BACKEND_LOWER" = "sox" ]; then
        use_sox=1
    elif [ "$RECORD_BACKEND_LOWER" = "auto" ] && [ -n "$REC_BIN" ]; then
        use_sox=1
    fi

    if [ "$use_sox" = "1" ]; then
        if [ -n "$REC_BIN" ]; then
            capture_name="sox"
            capture_cmd=("$REC_BIN" -q -c "$RECORD_CHANNELS" -r "$RECORD_SAMPLE_RATE" -b 16 -e signed-integer)
            capture_cmd+=("$RAW_RECORDING")
            if [ -n "${RECORD_SOX_EFFECTS:-}" ]; then
                # shellcheck disable=SC2206
                sox_effects=( ${RECORD_SOX_EFFECTS} )
                capture_cmd+=("${sox_effects[@]}")
            fi
        else
            if [ "$RECORD_BACKEND_LOWER" = "sox" ]; then
                echo "警告: RECORD_BACKEND=sox が指定されましたが rec コマンドが見つかりません。ffmpeg にフォールバックします。" >&2
            fi
        fi
    fi

    if [ "${#capture_cmd[@]}" -eq 0 ]; then
        capture_cmd=(ffmpeg -y -nostdin -hide_banner -loglevel error -nostats -f avfoundation)
        if [ "$RECORD_QUEUE_SIZE" -gt 0 ]; then
            capture_cmd+=(-thread_queue_size "$RECORD_QUEUE_SIZE")
        fi
        if [ "$MAX_RECORD_SECONDS" -gt 0 ]; then
            capture_cmd+=(-t "$MAX_RECORD_SECONDS" -timelimit "$MAX_RECORD_SECONDS")
        fi
        capture_cmd+=(
            -i "$AUDIO_DEVICE"
            -ac "$RECORD_CHANNELS"
            -ar "$RECORD_SAMPLE_RATE"
            -c:a pcm_s16le
        )
        if [ -n "${RECORD_FILTER:-}" ]; then
            capture_cmd+=(-af "$RECORD_FILTER")
        fi
        capture_cmd+=("$RAW_RECORDING")
    fi

    "${capture_cmd[@]}" &
    capture_pid=$!
    RECORD_ACTIVE_PID="$capture_pid"
    log_record_debug "capture start pid=$capture_pid backend=$capture_name"
    start_recording_status_loop "$capture_pid"
    start_recording_stop_key_loop "$capture_pid"
    start_recording_timeout_guard "$capture_pid" "$MAX_RECORD_SECONDS"

    timeout_triggered=0
    manual_stop_triggered=0
    while is_pid_alive "$capture_pid"; do
        if [ -n "${RECORD_TIMEOUT_FLAG:-}" ] && [ -f "$RECORD_TIMEOUT_FLAG" ]; then
            timeout_triggered=1
            log_record_debug "main loop timeout flag hit pid=$capture_pid"
            terminate_recording_process "$capture_pid"
            break
        fi
        if [ -n "${RECORD_STOP_KEY_FLAG:-}" ] && [ -f "$RECORD_STOP_KEY_FLAG" ]; then
            manual_stop_triggered=1
            rm -f "$RECORD_STOP_KEY_FLAG" 2>/dev/null || true
            log_record_debug "main loop stop key flag hit pid=$capture_pid"
            terminate_recording_process "$capture_pid"
            break
        fi
        sleep 0.2
    done

    if [ "$manual_stop_triggered" = "0" ] && [ -n "${RECORD_STOP_KEY_FLAG:-}" ] && [ -f "$RECORD_STOP_KEY_FLAG" ]; then
        manual_stop_triggered=1
        rm -f "$RECORD_STOP_KEY_FLAG" 2>/dev/null || true
    fi

    if ! wait "$capture_pid" 2>/dev/null; then
        capture_status=$?
        log_record_debug "wait on capture pid=$capture_pid returned status=$capture_status"
    else
        capture_status=0
        log_record_debug "wait on capture pid=$capture_pid completed cleanly"
    fi
    RECORD_ACTIVE_PID=""
    stop_recording_timeout_guard
    rm -f "$RECORD_TIMEOUT_FLAG" 2>/dev/null || true
    stop_recording_stop_key_loop
    if [ "$capture_name" = "ffmpeg" ] && [ "$capture_status" -ne 0 ] && [ -s "$RAW_RECORDING" ]; then
        echo "ffmpeg が中断されました。録音ファイルを修復しています..." >&2
        repaired_recording="${RAW_RECORDING%.wav}_repaired.wav"
        if ffmpeg -y -nostdin -hide_banner -loglevel warning -fflags +discardcorrupt -i "$RAW_RECORDING" -c copy "$repaired_recording" 2>/dev/null; then
            mv "$repaired_recording" "$RAW_RECORDING"
            capture_status=0
        else
            rm -f "$repaired_recording"
        fi
    fi
    effective_status=$capture_status
    if [ "$timeout_triggered" = "1" ] || [ "$manual_stop_triggered" = "1" ]; then
        effective_status=0
    elif [ "$capture_status" -ne 0 ]; then
        if [ -s "$RAW_RECORDING" ]; then
            effective_status=0
        elif [ -f "$RAW_RECORDING" ] && command -v ffprobe >/dev/null 2>&1; then
            duration_raw=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$RAW_RECORDING" 2>/dev/null || true)
            if [ -n "${duration_raw:-}" ]; then
                duration_int=$(printf '%.0f' "$duration_raw" 2>/dev/null || echo 0)
                if [ "$duration_int" -gt 0 ]; then
                    effective_status=0
                fi
            fi
        fi
    fi
    stop_recording_status_loop "$effective_status"
    if [ "$effective_status" -ne 0 ]; then
        echo "録音に失敗しました。既存のファイル選択に切り替えます。"
        RECORD_CHOICE="録音しない"
    else
        mkdir -p "$BUNDLE_DIR"
        if [ "${KEEP_RECORDING:-1}" = "1" ]; then
            SAVED_RECORDING="$BUNDLE_DIR/${RECORD_BASENAME}.wav"
            cp "$RAW_RECORDING" "$SAVED_RECORDING"
        fi
        FILE="$RAW_RECORDING"
        OUT_PREFIX="$BUNDLE_DIR/${RECORD_BASENAME}"
        FINAL_TXT="${OUT_PREFIX}.txt"
    fi
fi

if [ "${RECORD_CHOICE:-録音しない}" != "録音する" ]; then
    echo "文字起こししたい音声ファイルを Finder で選択してください..."
    FILE=$(osascript -e 'POSIX path of (choose file with prompt "音声ファイルを選択してください")' || true)
    if [ -z "$FILE" ]; then
        echo "ファイルが選択されませんでした。処理を中止します。"
        exit 1
    fi
    OUT_PREFIX="${FILE%.*}_full"
    FINAL_TXT="${OUT_PREFIX}.txt"
fi

# ====== 音声の整形・WAV変換 ======
echo "音声を解析用に整形しています..."
ADVANCED_FILTER="highpass=f=120,lowpass=f=7800,afftdn=nr=12,aresample=16000:resampler=soxr,dynaudnorm=f=75:g=15,loudnorm=I=-18:LRA=11:TP=-1.5"
SIMPLE_FILTER="aresample=16000:resampler=soxr,dynaudnorm=f=75:g=15"
FALLBACK_FILTER="$SIMPLE_FILTER"
PRIMARY_FILTER="$ADVANCED_FILTER"
if [ "$FAST_MODE" = "1" ]; then
    PRIMARY_FILTER="$SIMPLE_FILTER"
    FALLBACK_FILTER="aresample=16000"
fi

FFMPEG_RESAMPLE_RETRY=0
if [ "${#FFMPEG_RESAMPLE_ARGS[@]}" -gt 0 ]; then
    if ! ffmpeg "${FFMPEG_RESAMPLE_ARGS[@]}" -y -nostdin -hide_banner -loglevel warning -nostats \
        -i "$FILE" -vn -sn -dn -ac 1 -ar 16000 -af "$PRIMARY_FILTER" "$PROCESSED_WAV"; then
        FFMPEG_RESAMPLE_RETRY=1
    fi
else
    if ! ffmpeg -y -nostdin -hide_banner -loglevel warning -nostats \
        -i "$FILE" -vn -sn -dn -ac 1 -ar 16000 -af "$PRIMARY_FILTER" "$PROCESSED_WAV"; then
        FFMPEG_RESAMPLE_RETRY=1
    fi
fi

if [ "$FFMPEG_RESAMPLE_RETRY" -ne 0 ]; then
    if [ "$FAST_MODE" = "1" ]; then
        echo "フィルタ処理に失敗したため、最低限の整形で再試行します..."
    else
        echo "高度なフィルタの適用に失敗したため、簡易整形で再試行します..."
    fi
    if [ "${#FFMPEG_RESAMPLE_ARGS[@]}" -gt 0 ]; then
        ffmpeg "${FFMPEG_RESAMPLE_ARGS[@]}" -y -nostdin -hide_banner -loglevel warning -nostats \
            -i "$FILE" -vn -sn -dn -ac 1 -ar 16000 -af "$FALLBACK_FILTER" "$PROCESSED_WAV"
    else
        ffmpeg -y -nostdin -hide_banner -loglevel warning -nostats \
            -i "$FILE" -vn -sn -dn -ac 1 -ar 16000 -af "$FALLBACK_FILTER" "$PROCESSED_WAV"
    fi
fi


# ====== 分割判定（90分授業を想定） ======
SEGMENT_FILES=()
DURATION_SECONDS=""
if command -v ffprobe >/dev/null 2>&1; then
    duration_raw=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$PROCESSED_WAV" 2>/dev/null || true)
    if [ -n "${duration_raw:-}" ]; then
        DURATION_SECONDS=$(echo "$duration_raw" | awk '{printf("%d", ($1+0.5))}' 2>/dev/null || true)
    fi
fi

if [ -n "$DURATION_SECONDS" ] && [ "$SEGMENT_TIME" -gt 0 ] && [ "$DURATION_SECONDS" -gt "$SEGMENT_TIME" ]; then
    approx_minutes=$(( (DURATION_SECONDS + 30) / 60 ))
    echo "約 ${approx_minutes} 分の音声を検出。${SEGMENT_TIME} 秒ごとに分割して処理します。"
    FFMPEG_SEGMENT_SUCCESS=0
    if [ "${#FFMPEG_SEGMENT_ARGS[@]}" -gt 0 ]; then
        if ffmpeg "${FFMPEG_SEGMENT_ARGS[@]}" -y -nostdin -hide_banner -loglevel info -stats \
            -i "$PROCESSED_WAV" -f segment -segment_time "$SEGMENT_TIME" -c copy "$TMP_DIR/segment_%03d.wav"; then
            FFMPEG_SEGMENT_SUCCESS=1
        fi
    else
        if ffmpeg -y -nostdin -hide_banner -loglevel info -stats \
            -i "$PROCESSED_WAV" -f segment -segment_time "$SEGMENT_TIME" -c copy "$TMP_DIR/segment_%03d.wav"; then
            FFMPEG_SEGMENT_SUCCESS=1
        fi
    fi
    if [ "$FFMPEG_SEGMENT_SUCCESS" -eq 1 ]; then
        SEGMENT_FILES=()
        while IFS= read -r segment_path; do
            SEGMENT_FILES+=("$segment_path")
        done < <(find "$TMP_DIR" -maxdepth 1 -type f -name 'segment_*.wav' -print | sort)
        if [ ${#SEGMENT_FILES[@]} -eq 0 ]; then
            echo "セグメントファイルが見つかりませんでした。全体を一括で処理します。"
            SEGMENT_FILES=()
        fi
    else
        echo "セグメント化に失敗したため、全体を一括で処理します。"
        SEGMENT_FILES=()
    fi
fi


# ====== 文字起こし ======
SOURCE_DISPLAY="$FILE"
if [ "$RECORD_CHOICE" = "録音する" ]; then
    if [ -n "$SAVED_RECORDING" ]; then
        SOURCE_DISPLAY="$SAVED_RECORDING"
    else
        SOURCE_DISPLAY="録音データ（一時ファイル）"
    fi
fi

if [ ${#SEGMENT_FILES[@]} -gt 0 ]; then
    echo "文字起こし対象: $SOURCE_DISPLAY"

    if [ -z "${BUNDLE_DIR:-}" ]; then
        base_dir=$(dirname "$OUT_PREFIX")
        base_name=$(basename "$OUT_PREFIX")
        BUNDLE_DIR="$base_dir/${base_name}_bundle"
    fi
    mkdir -p "$BUNDLE_DIR"

    segment_base_name=$(basename "$OUT_PREFIX")
    segment_result_prefix="$BUNDLE_DIR/$segment_base_name"
    FINAL_TXT="${segment_result_prefix}.txt"
    : > "$FINAL_TXT"

    segment_total=${#SEGMENT_FILES[@]}
    SEGMENT_WORKERS="${SEGMENT_WORKERS:-}"
    if [ -z "$SEGMENT_WORKERS" ]; then
        SEGMENT_WORKERS="$PROCESSORS"
        case "$SEGMENT_WORKERS" in
            (''|*[!0-9]*) SEGMENT_WORKERS=0 ;;
        esac
        if [ "$SEGMENT_WORKERS" -lt 1 ]; then
            SEGMENT_WORKERS=1
        fi
        if [ "$HIGH_PERF_MODE" = "1" ] && [ "$SEGMENT_WORKERS" -lt 2 ] && [ "$segment_total" -gt 1 ]; then
            SEGMENT_WORKERS=2
        fi
    fi
    case "$SEGMENT_WORKERS" in
        (''|*[!0-9]*) SEGMENT_WORKERS=1 ;;
        (0) SEGMENT_WORKERS=1 ;;
    esac
    if [ "$SEGMENT_WORKERS" -gt "$segment_total" ]; then
        SEGMENT_WORKERS="$segment_total"
    fi

    echo "分割処理を ${SEGMENT_WORKERS} 並列で実行します。"

    # 背景ワーカープールを組んで whisper-cli を並列実行する
    segment_plain_outputs=()
    segment_elapsed_outputs=()
    segment_running_pids=()
    segment_running_indexes=()
    segment_index=0
    SEGMENT_FAILURE=0
    running_offset="0"

    for segment in "${SEGMENT_FILES[@]}"; do
        part_id=$(printf "%02d" $((segment_index + 1)))
        segment_tmp=$(mktemp -d "$TMP_DIR/segment_${part_id}.XXXXXX")
        segment_prefix="$segment_tmp/out"
        dest_plain="$BUNDLE_DIR/${segment_base_name}_part${part_id}.txt"
        dest_elapsed="$BUNDLE_DIR/${segment_base_name}_part${part_id}_elapsed.txt"
        segment_offset="$running_offset"

        segment_plain_outputs[$segment_index]="$dest_plain"
        segment_elapsed_outputs[$segment_index]="$dest_elapsed"

        echo "分割 ${part_id} をキューに投入（スレッド: ${THREADS}）..."
        (
            set -e
            tmp_prefix="$segment_prefix"
            final_plain="$dest_plain"
            final_elapsed="$dest_elapsed"
            segment_offset_value="$segment_offset"
            "$WHISPER_CLI" -m "$MODEL" -f "$segment" -l "$LANG" -otxt -oj -of "$tmp_prefix" "${CLI_OPTS[@]}"
            final_dir=$(dirname "$final_plain")
            mkdir -p "$final_dir"
            tmp_json="${tmp_prefix}.json"
            tmp_txt="${tmp_prefix}.txt"
            if [ -f "$tmp_txt" ]; then
                mv "$tmp_txt" "${final_plain}.partial"
                mv "${final_plain}.partial" "$final_plain"
            else
                : > "$final_plain"
            fi
            if ! generate_timestamped_txt "$tmp_json" "$final_elapsed" "$segment_offset_value"; then
                if [ -f "$final_plain" ]; then
                    cp "$final_plain" "$final_elapsed" 2>/dev/null || : > "$final_elapsed"
                else
                    : > "$final_elapsed"
                fi
            fi
            if [ -f "$tmp_json" ]; then
                mv "$tmp_json" "${final_plain%.txt}.json"
            fi
            if [ -f "$tmp_txt" ]; then
                rm -f "$tmp_txt"
            fi
        ) &
        pid=$!
        segment_running_pids+=("$pid")
        segment_running_indexes+=("$segment_index")

        if [ "${#segment_running_pids[@]}" -ge "$SEGMENT_WORKERS" ]; then
            pid_to_wait="${segment_running_pids[0]}"
            idx_to_wait="${segment_running_indexes[0]}"
            if ! wait "$pid_to_wait"; then
                SEGMENT_FAILURE=1
            fi
            segment_running_pids=("${segment_running_pids[@]:1}")
            segment_running_indexes=("${segment_running_indexes[@]:1}")
        fi

        if command -v python3 >/dev/null 2>&1; then
            segment_duration=""
            if command -v ffprobe >/dev/null 2>&1; then
                duration_raw=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$segment" 2>/dev/null || true)
                if [ -n "${duration_raw:-}" ]; then
                    segment_duration="$duration_raw"
                fi
            fi
            if [ -z "${segment_duration:-}" ] && [ -n "${SEGMENT_TIME:-}" ]; then
                segment_duration="$SEGMENT_TIME"
            fi
            if [ -n "${segment_duration:-}" ]; then
                running_offset=$(python3 - "$running_offset" "$segment_duration" <<'PY'
import sys

def parse(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

args = sys.argv
base = parse(args[1]) if len(args) > 1 else 0.0
increment = parse(args[2]) if len(args) > 2 else 0.0
print(f"{base + increment:.6f}")
PY
                )
            fi
        fi

        segment_index=$((segment_index + 1))
    done

    for pending_index in "${!segment_running_pids[@]}"; do
        pid_to_wait="${segment_running_pids[$pending_index]}"
        idx_to_wait="${segment_running_indexes[$pending_index]}"
        if ! wait "$pid_to_wait"; then
            SEGMENT_FAILURE=1
        fi
    done

    if [ "$SEGMENT_FAILURE" -ne 0 ]; then
        echo "一部のセグメント処理でエラーが発生しました。ログを確認してください。" >&2
    fi

    FINAL_TIMED_TXT="${segment_result_prefix}_elapsed.txt"
    : > "$FINAL_TIMED_TXT"

    for idx in "${!segment_plain_outputs[@]}"; do
        part_id=$(printf "%02d" $((idx + 1)))
        part_plain="${segment_plain_outputs[$idx]}"
        part_elapsed="${segment_elapsed_outputs[$idx]}"
        if [ -f "$part_plain" ]; then
            printf "[Part %s]\n" "$part_id" >> "$FINAL_TXT"
            cat "$part_plain" >> "$FINAL_TXT"
            printf "\n" >> "$FINAL_TXT"
        fi
        if [ -f "$part_elapsed" ]; then
            printf "[Part %s]\n" "$part_id" >> "$FINAL_TIMED_TXT"
            cat "$part_elapsed" >> "$FINAL_TIMED_TXT"
            printf "\n" >> "$FINAL_TIMED_TXT"
        fi
    done
    echo "分割済みテキストは $BUNDLE_DIR/${segment_base_name}_partXX.txt として個別に保存されています。"
    echo "経過時間付きの個別テキストは $BUNDLE_DIR/${segment_base_name}_partXX_elapsed.txt に保存されています。"
else
    echo "文字起こし中（スレッド: ${THREADS}）: $SOURCE_DISPLAY"
    "$WHISPER_CLI" -m "$MODEL" -f "$PROCESSED_WAV" -l "$LANG" -otxt -oj -of "$OUT_PREFIX" "${CLI_OPTS[@]}"
    main_json="${OUT_PREFIX}.json"
    main_plain="${OUT_PREFIX}.txt"
    FINAL_TIMED_TXT="${OUT_PREFIX}_elapsed.txt"
    if ! generate_timestamped_txt "$main_json" "$FINAL_TIMED_TXT"; then
        if [ -f "$main_plain" ]; then
            if ! cp "$main_plain" "$FINAL_TIMED_TXT" 2>/dev/null; then
                FINAL_TIMED_TXT=""
            fi
        else
            FINAL_TIMED_TXT=""
        fi
    fi
    FINAL_TXT="$main_plain"
fi


# ====== 自動重複除去 ======
DEDUP_CONSECUTIVE="${DEDUP_CONSECUTIVE:-0}"
case "$DEDUP_CONSECUTIVE" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                DEDUP_CONSECUTIVE=1
                ;;
            (*)
                DEDUP_CONSECUTIVE=0
                ;;
        esac
        ;;
    (1|true|TRUE|yes|YES)
        DEDUP_CONSECUTIVE=1
        ;;
    (*)
        DEDUP_CONSECUTIVE=0
        ;;
esac
if [ "$DEDUP_CONSECUTIVE" = "1" ] && [ -f "$FINAL_TXT" ]; then
    DEDUP_TMP="$TMP_DIR/dedup.txt"
    awk 'NR==1 {print; prev=$0; next} {if ($0 == prev && $0 != "") next; print; prev=$0}' "$FINAL_TXT" > "$DEDUP_TMP" && mv "$DEDUP_TMP" "$FINAL_TXT"
fi
if [ "$DEDUP_CONSECUTIVE" = "1" ] && [ -n "${FINAL_TIMED_TXT:-}" ] && [ -f "$FINAL_TIMED_TXT" ]; then
    DEDUP_TS_TMP="$TMP_DIR/dedup_elapsed.txt"
    awk 'NR==1 {print; prev=$0; next} {if ($0 == prev && $0 != "") next; print; prev=$0}' "$FINAL_TIMED_TXT" > "$DEDUP_TS_TMP" && mv "$DEDUP_TS_TMP" "$FINAL_TIMED_TXT"
fi

# ====== 改行の整形 ======
REDUCE_CHAR_RUNS="${REDUCE_CHAR_RUNS:-0}"
REDUCE_CHAR_RUNS_ACTIVE=0
case "$REDUCE_CHAR_RUNS" in
    (auto)
        case "$LANG_LOWER" in
            (ja|ja-jp|japanese)
                REDUCE_CHAR_RUNS_ACTIVE=1
                ;;
        esac
        ;;
    (1|true|TRUE|yes|YES)
        REDUCE_CHAR_RUNS_ACTIVE=1
        ;;
esac
REDUCE_CHAR_RUN_MAX="${REDUCE_CHAR_RUN_MAX:-12}"
case "$REDUCE_CHAR_RUN_MAX" in
    (''|*[!0-9]*) REDUCE_CHAR_RUN_MAX=12 ;;
    (*)
        if [ "$REDUCE_CHAR_RUN_MAX" -lt 2 ]; then
            REDUCE_CHAR_RUN_MAX=2
        fi
        ;;
esac

if [ -f "$FINAL_TXT" ] && command -v python3 >/dev/null 2>&1; then
    # python スクリプトは stdin から読み込み、対象ファイル名を argv[1] で渡す
    REDUCE_CHAR_RUNS_ACTIVE="$REDUCE_CHAR_RUNS_ACTIVE" \
    REDUCE_CHAR_RUN_MAX="$REDUCE_CHAR_RUN_MAX" \
    python3 - "$FINAL_TXT" <<'PY'
import sys
from pathlib import Path
import os
import re

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

out_lines = []
buffer = []

reduce_runs = os.environ.get("REDUCE_CHAR_RUNS_ACTIVE") == "1"
try:
    max_run = int(os.environ.get("REDUCE_CHAR_RUN_MAX", "12"))
except ValueError:
    max_run = 12
max_run = max(2, max_run)

if reduce_runs:
    char_run_pattern = re.compile(r"([^\s])\1{" + str(max_run) + r",}")
    def collapse_runs(value: str) -> str:
        # 異常な長さの連続文字列を最大長に抑える
        def _repl(match: re.Match[str]) -> str:
            ch = match.group(1)
            return ch * max_run
        return char_run_pattern.sub(_repl, value)
else:
    def collapse_runs(value: str) -> str:
        return value

for line in text.splitlines():
    stripped = line.strip()

    # セグメント見出しはそのまま保持
    if stripped.startswith("[Part "):
        if buffer:
            out_lines.append("".join(buffer))
            buffer.clear()
        out_lines.append(stripped)
        continue

    if not stripped:
        if buffer:
            out_lines.append("".join(buffer))
            buffer.clear()
        # 空行を畳み過ぎないように 1 行だけ追加
        if not out_lines or out_lines[-1] != "":
            out_lines.append("")
        continue

    buffer.append(collapse_runs(stripped))

if buffer:
    out_lines.append("".join(buffer))

path.write_text("\n".join(out_lines).rstrip("\n") + "\n", encoding="utf-8")
PY
fi

if [ -n "${FINAL_TIMED_TXT:-}" ] && [ -f "$FINAL_TIMED_TXT" ] && command -v python3 >/dev/null 2>&1; then
    REDUCE_CHAR_RUNS_ACTIVE="$REDUCE_CHAR_RUNS_ACTIVE" \
    REDUCE_CHAR_RUN_MAX="$REDUCE_CHAR_RUN_MAX" \
    python3 - "$FINAL_TIMED_TXT" <<'PY'
import sys
from pathlib import Path
import os
import re

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

out_lines = []
buffer = []

reduce_runs = os.environ.get("REDUCE_CHAR_RUNS_ACTIVE") == "1"
try:
    max_run = int(os.environ.get("REDUCE_CHAR_RUN_MAX", "12"))
except ValueError:
    max_run = 12
max_run = max(2, max_run)

if reduce_runs:
    char_run_pattern = re.compile(r"([^\s])\1{" + str(max_run) + r",}")
    def collapse_runs(value: str) -> str:
        def _repl(match: re.Match[str]) -> str:
            ch = match.group(1)
            return ch * max_run
        return char_run_pattern.sub(_repl, value)
else:
    def collapse_runs(value: str) -> str:
        return value

for line in text.splitlines():
    stripped = line.strip()

    if stripped.startswith("[Part "):
        if buffer:
            out_lines.append("".join(buffer))
            buffer.clear()
        out_lines.append(stripped)
        continue

    if not stripped:
        if buffer:
            out_lines.append("".join(buffer))
            buffer.clear()
        if not out_lines or out_lines[-1] != "":
            out_lines.append("")
        continue

    buffer.append(collapse_runs(stripped))

if buffer:
    out_lines.append("".join(buffer))

path.write_text("\n".join(out_lines).rstrip("\n") + "\n", encoding="utf-8")
PY
fi

if [ -n "${FINAL_TIMED_TXT:-}" ] && [ -f "$FINAL_TIMED_TXT" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "$FINAL_TIMED_TXT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
lines = text.splitlines()

out_lines = []

def push_blank():
    if out_lines and out_lines[-1] == "":
        return
    out_lines.append("")

idx = 0
total = len(lines)
while idx < total:
    raw_line = lines[idx]
    stripped = raw_line.rstrip("
")
    trimmed = stripped.strip()

    if not trimmed:
        push_blank()
        idx += 1
        continue

    if trimmed.startswith("[Part "):
        if out_lines and out_lines[-1] != "":
            out_lines.append("")
        out_lines.append(trimmed)
        idx += 1
        continue

    if trimmed.startswith("{") and "}" in trimmed:
        ts, _, rest = trimmed.partition("}")
        ts = ts + "}"
        text_body = rest
        if text_body.startswith("　"):
            formatted = text_body
        else:
            formatted = ("　" + text_body.lstrip()) if text_body else "　"
        out_lines.append(ts)
        if formatted:
            out_lines.append(formatted)
        idx += 1
        continue

    out_lines.append(trimmed)
    idx += 1

while out_lines and out_lines[-1] == "":
    out_lines.pop()

path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
PY
fi



echo "文字起こし完了！ 出力ファイル：$FINAL_TXT"
if [ -n "${FINAL_TIMED_TXT:-}" ] && [ -f "$FINAL_TIMED_TXT" ]; then
    echo "経過時間付き出力：$FINAL_TIMED_TXT"
fi
if [ -n "$SAVED_RECORDING" ]; then
    if [ -n "$BUNDLE_DIR" ]; then
        echo "録音した音声と書き起こしを保存しました：$BUNDLE_DIR"
    else
        echo "録音した音声ファイルを保存しました：$SAVED_RECORDING"
    fi
elif [ "$RECORD_CHOICE" = "録音する" ]; then
    echo "録音データは一時ファイルとして削除されました。"
fi
