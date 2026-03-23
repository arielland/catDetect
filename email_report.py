"""
Sends a daily summary email with the cat detection stats and attached CSV log.

Setup:
    1. Copy .env.example to .env and fill in your credentials
    2. Add to cron: 0 7 * * * /usr/bin/python3 /home/pi/catDetect/email_report.py

Usage:
    python3 email_report.py                  # send report for today
    python3 email_report.py --days 7         # include last 7 days of data
    python3 email_report.py --phases before after
"""

import argparse
import csv
import io
import os
import smtplib
import zipfile
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

# ── Config (loaded from .env file or environment variables) ──────────────────

def load_env(path=".env"):
    """Simple .env loader — no external dependency needed."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    return env

ENV = load_env()

SMTP_HOST     = ENV.get("SMTP_HOST",     os.getenv("SMTP_HOST",     "smtp.gmail.com"))
SMTP_PORT     = int(ENV.get("SMTP_PORT", os.getenv("SMTP_PORT",     "587")))
SMTP_USER     = ENV.get("SMTP_USER",     os.getenv("SMTP_USER",     ""))
SMTP_PASSWORD = ENV.get("SMTP_PASSWORD", os.getenv("SMTP_PASSWORD", ""))
EMAIL_TO      = ENV.get("EMAIL_TO",      os.getenv("EMAIL_TO",      SMTP_USER))
LOG_FILE      = ENV.get("LOG_FILE",      os.getenv("LOG_FILE",      "detections.csv"))

# ── Report building ──────────────────────────────────────────────────────────

def phase_stats(df: pd.DataFrame, phase: str) -> dict | None:
    p = df[df["phase"] == phase]
    if p.empty:
        return None
    total = len(p)
    with_cats = (p["cat_count"] > 0).sum()
    occupancy = with_cats / total if total else 0
    avg = p[p["cat_count"] > 0]["cat_count"].mean() if with_cats else 0
    return {
        "phase": phase,
        "samples": total,
        "cat_detections": int(with_cats),
        "occupancy_rate": occupancy,
        "avg_cats_when_present": round(avg, 1),
        "peak": int(p["cat_count"].max()),
    }


def hourly_table(df: pd.DataFrame, phase: str) -> str:
    p = df[(df["phase"] == phase) & (df["cat_count"] > 0)].copy()
    if p.empty:
        return ""
    p["hour"] = p["timestamp"].dt.hour
    counts = p.groupby("hour")["cat_count"].sum()
    max_val = counts.max() or 1
    rows = []
    for h in range(24):
        v = int(counts.get(h, 0))
        bar = "█" * int(v / max_val * 20)
        rows.append(f"  {h:02d}:00  {bar:<20}  {v}")
    return "\n".join(rows)


def build_report(df: pd.DataFrame, phases: list[str], since: datetime) -> str:
    recent = df[df["timestamp"] >= since]
    lines = []
    lines.append(f"Cat Detection Daily Report — {datetime.now().strftime('%A, %d %b %Y')}")
    lines.append("=" * 56)
    lines.append(f"Log period shown: last {(datetime.now() - since).days + 1} day(s)")
    lines.append(f"Total entries in window: {len(recent)}")
    lines.append("")

    all_stats = []
    for phase in phases:
        s = phase_stats(recent, phase)
        if s is None:
            lines.append(f"  Phase '{phase}': no data yet.\n")
            continue
        all_stats.append(s)
        lines.append(f"Phase: {phase.upper()}")
        lines.append(f"  Samples          : {s['samples']}")
        lines.append(f"  Cat detections   : {s['cat_detections']}")
        lines.append(f"  Occupancy rate   : {s['occupancy_rate']:.1%}")
        lines.append(f"  Avg cats (when present): {s['avg_cats_when_present']}")
        lines.append(f"  Peak at once     : {s['peak']}")
        lines.append("")

        hmap = hourly_table(recent, phase)
        if hmap:
            lines.append(f"  Hourly breakdown ({phase}):")
            lines.append(hmap)
            lines.append("")

    # Delta between first two phases
    if len(all_stats) == 2:
        delta = all_stats[1]["occupancy_rate"] - all_stats[0]["occupancy_rate"]
        direction = "REDUCED" if delta < 0 else "INCREASED"
        lines.append("=" * 56)
        lines.append(f"Result: cat occupancy {direction} by {abs(delta):.1%}")
        lines.append(f"  {all_stats[0]['phase']}: {all_stats[0]['occupancy_rate']:.1%}  →  "
                     f"{all_stats[1]['phase']}: {all_stats[1]['occupancy_rate']:.1%}")
        lines.append("=" * 56)

    lines.append("")
    lines.append("Full CSV log attached.")
    return "\n".join(lines)


# ── Email sending ────────────────────────────────────────────────────────────

def attach_csv(msg: MIMEMultipart, csv_path: str):
    """Attach the CSV, zipped to keep email size small."""
    p = Path(csv_path)
    if not p.exists():
        return

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, p.name)
    buf.seek(0)

    part = MIMEBase("application", "zip")
    part.set_payload(buf.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{p.stem}.zip"',
    )
    msg.attach(part)


def send_email(subject: str, body: str, csv_path: str):
    if not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError(
            "SMTP_USER and SMTP_PASSWORD must be set in .env or environment variables."
        )

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    attach_csv(msg, csv_path)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())
    print(f"Email sent to {EMAIL_TO}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Email daily cat detection report")
    parser.add_argument("--csv", default=LOG_FILE)
    parser.add_argument("--phases", nargs="+", default=["before", "after"])
    parser.add_argument("--days", type=int, default=1,
                        help="Include data from the last N days in the report body")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"Log file '{args.csv}' not found — nothing to send yet.")
        return

    since = datetime.now() - timedelta(days=args.days)
    report = build_report(df, args.phases, since)

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"Cat Detector Report — {today}"

    send_email(subject, report, args.csv)
    print(report)


if __name__ == "__main__":
    main()
