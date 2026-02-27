# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas==2.2.3",
# ]
# ///

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SilenceRegion:
    region_id: int
    region_type: str
    start_ms: int
    end_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify timestamp silence violations using synthetic timeline ground truth. "
            "Checks segment and word/token [t0, t1] overlap against known silence regions."
        )
    )
    parser.add_argument("--timeline-csv", required=True, help="Timeline CSV from stable-timestamps_005.py")
    parser.add_argument("--baseline-json", required=True, help="whisper-cli baseline JSON output")
    parser.add_argument("--stable-json", required=True, help="whisper-cli stable JSON output")
    parser.add_argument(
        "--out-csv",
        default="stable_timestamps_005_silence_summary.csv",
        help="Summary output CSV",
    )
    parser.add_argument(
        "--out-violations-csv",
        default="",
        help="Detailed violations output CSV (default: <out-csv> with _violations suffix)",
    )
    parser.add_argument(
        "--overlap-threshold-ms",
        type=int,
        default=500,
        help="PASS threshold: no segment may overlap silence by more than this many ms",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print top-k segment violations per run",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def load_silence_regions(path: Path) -> list[SilenceRegion]:
    timeline = pd.read_csv(path)
    require_columns(timeline, ["type", "start_s", "end_s"], str(path))

    silence = timeline[timeline["type"].astype(str).str.contains("silence", na=False)].copy()
    silence = silence.reset_index(drop=True)

    regions: list[SilenceRegion] = []
    for i, row in silence.iterrows():
        start_ms = int(round(float(row["start_s"]) * 1000.0))
        end_ms = int(round(float(row["end_s"]) * 1000.0))
        if end_ms <= start_ms:
            continue
        regions.append(
            SilenceRegion(
                region_id=i,
                region_type=str(row["type"]),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )

    regions.sort(key=lambda r: r.start_ms)
    return regions


def parse_offsets(obj: dict[str, Any]) -> tuple[int, int] | None:
    offsets = obj.get("offsets")
    if not isinstance(offsets, dict):
        return None
    if "from" not in offsets or "to" not in offsets:
        return None

    start_ms = int(offsets["from"])
    end_ms = int(offsets["to"])
    if start_ms < 0 or end_ms < 0:
        return None
    return start_ms, end_ms


def load_whisper_units(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    segments_raw = obj.get("transcription", [])
    if not isinstance(segments_raw, list):
        segments_raw = []

    seg_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []

    for seg_idx, seg in enumerate(segments_raw):
        if not isinstance(seg, dict):
            continue

        seg_offsets = parse_offsets(seg)
        if seg_offsets is not None:
            seg_rows.append(
                {
                    "segment_idx": seg_idx,
                    "token_idx": -1,
                    "text": str(seg.get("text", "")),
                    "start_ms": seg_offsets[0],
                    "end_ms": seg_offsets[1],
                    "dur_ms": seg_offsets[1] - seg_offsets[0],
                }
            )

        tokens = seg.get("tokens", [])
        if not isinstance(tokens, list):
            continue

        for tok_idx, tok in enumerate(tokens):
            if not isinstance(tok, dict):
                continue
            tok_offsets = parse_offsets(tok)
            if tok_offsets is None:
                continue
            word_rows.append(
                {
                    "segment_idx": seg_idx,
                    "token_idx": tok_idx,
                    "text": str(tok.get("text", "")),
                    "start_ms": tok_offsets[0],
                    "end_ms": tok_offsets[1],
                    "dur_ms": tok_offsets[1] - tok_offsets[0],
                }
            )

    seg_df = pd.DataFrame(
        seg_rows,
        columns=["segment_idx", "token_idx", "text", "start_ms", "end_ms", "dur_ms"],
    )
    word_df = pd.DataFrame(
        word_rows,
        columns=["segment_idx", "token_idx", "text", "start_ms", "end_ms", "dur_ms"],
    )
    return seg_df, word_df


def point_region(t_ms: int, regions: list[SilenceRegion]) -> SilenceRegion | None:
    for r in regions:
        if r.end_ms <= t_ms:
            continue
        if r.start_ms > t_ms:
            return None
        return r
    return None


def overlap_details(start_ms: int, end_ms: int, regions: list[SilenceRegion]) -> dict[str, Any]:
    if end_ms <= start_ms:
        return {
            "overlap_ms": 0,
            "max_overlap_ms": 0,
            "n_regions_hit": 0,
            "first_region_id": -1,
            "first_region_type": "",
            "first_region_start_ms": -1,
            "first_region_end_ms": -1,
        }

    total_overlap = 0
    max_overlap = 0
    n_regions_hit = 0
    first_region: SilenceRegion | None = None

    for r in regions:
        if r.end_ms <= start_ms:
            continue
        if r.start_ms >= end_ms:
            break

        overlap = min(end_ms, r.end_ms) - max(start_ms, r.start_ms)
        if overlap <= 0:
            continue

        total_overlap += overlap
        max_overlap = max(max_overlap, overlap)
        n_regions_hit += 1
        if first_region is None:
            first_region = r

    if first_region is None:
        return {
            "overlap_ms": 0,
            "max_overlap_ms": 0,
            "n_regions_hit": 0,
            "first_region_id": -1,
            "first_region_type": "",
            "first_region_start_ms": -1,
            "first_region_end_ms": -1,
        }

    return {
        "overlap_ms": total_overlap,
        "max_overlap_ms": max_overlap,
        "n_regions_hit": n_regions_hit,
        "first_region_id": first_region.region_id,
        "first_region_type": first_region.region_type,
        "first_region_start_ms": first_region.start_ms,
        "first_region_end_ms": first_region.end_ms,
    }


def analyze_units(
    run_name: str,
    unit_type: str,
    units: pd.DataFrame,
    regions: list[SilenceRegion],
    threshold_ms: int,
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame(
            columns=[
                "run",
                "unit_type",
                "segment_idx",
                "token_idx",
                "text",
                "start_ms",
                "end_ms",
                "dur_ms",
                "start_in_silence",
                "end_in_silence",
                "overlap_ms",
                "max_overlap_ms",
                "n_regions_hit",
                "first_region_id",
                "first_region_type",
                "first_region_start_ms",
                "first_region_end_ms",
                "overlap_any",
                "overlap_gt_threshold",
            ]
        )

    rows: list[dict[str, Any]] = []
    for row in units.itertuples(index=False):
        start_ms = int(row.start_ms)
        end_ms = int(row.end_ms)
        end_probe = end_ms - 1 if end_ms > start_ms else end_ms

        start_region = point_region(start_ms, regions)
        end_region = point_region(end_probe, regions)
        ov = overlap_details(start_ms, end_ms, regions)
        overlap_any = ov["overlap_ms"] > 0

        rows.append(
            {
                "run": run_name,
                "unit_type": unit_type,
                "segment_idx": int(row.segment_idx),
                "token_idx": int(row.token_idx),
                "text": str(row.text),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "dur_ms": int(row.dur_ms),
                "start_in_silence": start_region is not None,
                "end_in_silence": end_region is not None,
                "overlap_ms": int(ov["overlap_ms"]),
                "max_overlap_ms": int(ov["max_overlap_ms"]),
                "n_regions_hit": int(ov["n_regions_hit"]),
                "first_region_id": int(ov["first_region_id"]),
                "first_region_type": str(ov["first_region_type"]),
                "first_region_start_ms": int(ov["first_region_start_ms"]),
                "first_region_end_ms": int(ov["first_region_end_ms"]),
                "overlap_any": overlap_any,
                "overlap_gt_threshold": int(ov["overlap_ms"]) > threshold_ms,
            }
        )

    return pd.DataFrame(rows)


def summarize_run(run_name: str, all_details: pd.DataFrame, threshold_ms: int) -> dict[str, Any]:
    seg = all_details[all_details["unit_type"] == "segment"]
    wrd = all_details[all_details["unit_type"] == "word"]

    def metrics(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
        n_total = int(df.shape[0])
        n_overlap_any = int(df["overlap_any"].sum()) if n_total else 0
        n_overlap_gt_threshold = int(df["overlap_gt_threshold"].sum()) if n_total else 0
        overlap_values = df.loc[df["overlap_any"], "overlap_ms"] if n_total else pd.Series(dtype="int64")
        return {
            f"n_{prefix}": n_total,
            f"n_{prefix}_overlap_any": n_overlap_any,
            f"n_{prefix}_overlap_gt_threshold": n_overlap_gt_threshold,
            f"pct_{prefix}_overlap_any": (100.0 * n_overlap_any / n_total) if n_total else 0.0,
            f"max_{prefix}_overlap_ms": int(df["overlap_ms"].max()) if n_total else 0,
            f"mean_{prefix}_overlap_ms_overlapping": float(overlap_values.mean()) if n_overlap_any else 0.0,
            f"n_{prefix}_start_in_silence": int(df["start_in_silence"].sum()) if n_total else 0,
            f"n_{prefix}_end_in_silence": int(df["end_in_silence"].sum()) if n_total else 0,
        }

    seg_m = metrics(seg, "segments")
    wrd_m = metrics(wrd, "words")
    passed = seg_m["n_segments_overlap_gt_threshold"] == 0

    return {
        "run": run_name,
        "overlap_threshold_ms": threshold_ms,
        **seg_m,
        **wrd_m,
        "pass_segments_threshold": passed,
    }


def print_top_segment_violations(details: pd.DataFrame, run_name: str, top_k: int) -> None:
    seg = details[(details["run"] == run_name) & (details["unit_type"] == "segment") & (details["overlap_any"])]
    seg = seg.sort_values("overlap_ms", ascending=False).head(top_k)
    if seg.empty:
        print(f"[{run_name}] no segment silence overlaps")
        return

    print(f"[{run_name}] top {len(seg)} segment silence overlaps:")
    for row in seg.itertuples(index=False):
        print(
            "  "
            f"seg={row.segment_idx} "
            f"{row.start_ms/1000.0:.3f}s->{row.end_ms/1000.0:.3f}s "
            f"overlap={row.overlap_ms/1000.0:.3f}s "
            f"silence={row.first_region_start_ms/1000.0:.3f}s->{row.first_region_end_ms/1000.0:.3f}s "
            f"text={row.text.strip()[:90]}"
        )


def main() -> None:
    args = parse_args()

    regions = load_silence_regions(Path(args.timeline_csv))
    if not regions:
        raise RuntimeError("No silence regions found in timeline CSV")

    runs = [
        ("baseline", Path(args.baseline_json)),
        ("stable", Path(args.stable_json)),
    ]

    all_details_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for run_name, json_path in runs:
        seg_df, word_df = load_whisper_units(json_path)
        seg_details = analyze_units(run_name, "segment", seg_df, regions, args.overlap_threshold_ms)
        word_details = analyze_units(run_name, "word", word_df, regions, args.overlap_threshold_ms)
        run_details = pd.concat([seg_details, word_details], ignore_index=True)

        all_details_parts.append(run_details)
        summaries.append(summarize_run(run_name, run_details, args.overlap_threshold_ms))

    details_df = pd.concat(all_details_parts, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("run").reset_index(drop=True)

    violations_df = details_df[details_df["overlap_any"]].copy()
    violations_df = violations_df.sort_values(
        ["run", "unit_type", "overlap_ms"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    out_summary = Path(args.out_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(summary_df.to_csv(index=False), encoding="utf-8")

    if args.out_violations_csv:
        out_violations = Path(args.out_violations_csv)
    else:
        out_violations = out_summary.with_name(f"{out_summary.stem}_violations.csv")
    out_violations.parent.mkdir(parents=True, exist_ok=True)
    out_violations.write_text(violations_df.to_csv(index=False), encoding="utf-8")

    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary CSV: {out_summary}")
    print(f"Saved violations CSV: {out_violations}")
    print()

    for run_name, _ in runs:
        print_top_segment_violations(details_df, run_name, args.top_k)


if __name__ == "__main__":
    main()
