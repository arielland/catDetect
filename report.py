"""
Before/after cat detection report.

Usage:
    python report.py                    # compare 'before' vs 'after' phases
    python report.py --phases before after custom_phase
    python report.py --csv detections.csv
"""

import argparse
from datetime import datetime

import pandas as pd


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


def phase_stats(df: pd.DataFrame, phase: str) -> dict:
    p = df[df["phase"] == phase]
    if p.empty:
        return {"phase": phase, "samples": 0}

    total_samples = len(p)
    cat_samples = (p["cat_count"] > 0).sum()
    occupancy_rate = cat_samples / total_samples if total_samples else 0
    avg_cats_when_present = p[p["cat_count"] > 0]["cat_count"].mean()
    peak = p["cat_count"].max()
    duration_h = (p["timestamp"].max() - p["timestamp"].min()).total_seconds() / 3600

    return {
        "phase": phase,
        "samples": total_samples,
        "duration_hours": round(duration_h, 2),
        "cat_detections": int(cat_samples),
        "occupancy_rate": f"{occupancy_rate:.1%}",
        "avg_cats_when_present": round(avg_cats_when_present, 2) if cat_samples else 0,
        "peak_cats_at_once": int(peak),
        "first_seen": p["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
        "last_seen": p["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
    }


def hourly_heatmap(df: pd.DataFrame, phase: str):
    """Print a simple text heatmap of cat activity by hour of day."""
    p = df[(df["phase"] == phase) & (df["cat_count"] > 0)].copy()
    if p.empty:
        return
    p["hour"] = p["timestamp"].dt.hour
    counts = p.groupby("hour")["cat_count"].sum()

    print(f"\n  Hourly activity heatmap — {phase}")
    print("  " + "-" * 50)
    max_val = counts.max() if not counts.empty else 1
    for h in range(24):
        bar_len = int((counts.get(h, 0) / max_val) * 30)
        bar = "█" * bar_len
        print(f"  {h:02d}:00  {bar:<30}  {int(counts.get(h, 0))}")


def main():
    parser = argparse.ArgumentParser(description="Cat detection report")
    parser.add_argument("--csv", default="detections.csv")
    parser.add_argument("--phases", nargs="+", default=["before", "after"])
    args = parser.parse_args()

    try:
        df = load(args.csv)
    except FileNotFoundError:
        print(f"No log file found at '{args.csv}'. Run detect.py first.")
        return

    print(f"\n{'='*55}")
    print(f"  Cat Detection Report   |   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    print(f"  Total log entries: {len(df)}")
    print(f"  Phases found:      {', '.join(df['phase'].unique())}")
    print()

    all_stats = []
    for phase in args.phases:
        if phase not in df["phase"].values:
            print(f"  ⚠  Phase '{phase}' not found in log.\n")
            continue
        stats = phase_stats(df, phase)
        all_stats.append(stats)

        print(f"  Phase: {stats['phase'].upper()}")
        print(f"    Monitoring duration : {stats['duration_hours']} hours")
        print(f"    Total samples       : {stats['samples']}")
        print(f"    Cat detections      : {stats['cat_detections']}")
        print(f"    Occupancy rate      : {stats['occupancy_rate']}")
        print(f"    Avg cats (present)  : {stats['avg_cats_when_present']}")
        print(f"    Peak cats at once   : {stats['peak_cats_at_once']}")
        print(f"    Window              : {stats['first_seen']} → {stats['last_seen']}")
        print()

    # Delta summary
    if len(all_stats) == 2:
        before_rate = float(all_stats[0]["occupancy_rate"].strip("%")) / 100
        after_rate = float(all_stats[1]["occupancy_rate"].strip("%")) / 100
        delta = after_rate - before_rate
        direction = "▼ reduced" if delta < 0 else "▲ increased"
        print(f"{'='*55}")
        print(f"  Result: cat occupancy {direction} by {abs(delta):.1%}")
        print(f"  ({all_stats[0]['phase']}: {all_stats[0]['occupancy_rate']}  →  "
              f"{all_stats[1]['phase']}: {all_stats[1]['occupancy_rate']})")
        print(f"{'='*55}")

    # Heatmaps
    for phase in args.phases:
        if phase in df["phase"].values:
            hourly_heatmap(df, phase)

    print()


if __name__ == "__main__":
    main()
