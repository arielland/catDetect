"""
Hourly health + log email — only runs when test mode is active.

Activate:   touch /home/$(whoami)/catDetect/.test_mode
Deactivate: rm   /home/$(whoami)/catDetect/.test_mode

Cron (added by setup.sh):
    0 * * * * cd /home/<user>/catDetect && /home/<user>/catDetect/venv/bin/python test_mode_email.py
"""

import os
import subprocess
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Config (same .env as email_report.py) ────────────────────────────────────

def load_env(path=".env"):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    return env

ENV           = load_env()
SMTP_HOST     = ENV.get("SMTP_HOST",     os.getenv("SMTP_HOST",     "smtp.gmail.com"))
SMTP_PORT     = int(ENV.get("SMTP_PORT", os.getenv("SMTP_PORT",     "587")))
SMTP_USER     = ENV.get("SMTP_USER",     os.getenv("SMTP_USER",     ""))
SMTP_PASSWORD = ENV.get("SMTP_PASSWORD", os.getenv("SMTP_PASSWORD", ""))
EMAIL_TO      = ENV.get("EMAIL_TO",      os.getenv("EMAIL_TO",      SMTP_USER))
LOG_FILE      = ENV.get("LOG_FILE",      os.getenv("LOG_FILE",      "detections.csv"))
FLAG_FILE     = Path(".test_mode")

# ── System stats ─────────────────────────────────────────────────────────────

def cpu_temp() -> str:
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return f"{int(raw) / 1000:.1f}°C"
    except Exception:
        return "unavailable"

def cpu_usage() -> str:
    try:
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "Cpu(s)" in line or "%Cpu" in line:
                # Extract idle % and calculate usage
                parts = line.replace(",", " ").split()
                for i, p in enumerate(parts):
                    if "id" in p and i > 0:
                        idle = float(parts[i - 1].replace("%", ""))
                        return f"{100 - idle:.1f}%"
        return "unavailable"
    except Exception:
        return "unavailable"

def memory_usage() -> str:
    try:
        result = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                return f"{parts[2]} used / {parts[1]} total"
        return "unavailable"
    except Exception:
        return "unavailable"

def program_state() -> str:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "detect.py"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            pids = result.stdout.strip().splitlines()
            return f"✅ Running (PID {', '.join(pids)})"
        return "❌ Not running"
    except Exception:
        return "unavailable"

def last_log_lines(n=20) -> str:
    p = Path(LOG_FILE)
    if not p.exists():
        return "  (no detections logged yet)"
    lines = p.read_text().splitlines()
    if len(lines) <= 1:
        return "  (only header row — no detections yet)"
    # Return header + last n data lines
    header = lines[0]
    data   = lines[1:]
    recent = data[-n:]
    return "\n".join([header] + recent)

# ── Build email ───────────────────────────────────────────────────────────────

def build_body() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"CatDetect — Hourly Health Check  [{now}]",
        "=" * 50,
        "",
        "SYSTEM HEALTH",
        f"  CPU Temperature : {cpu_temp()}",
        f"  CPU Usage       : {cpu_usage()}",
        f"  Memory          : {memory_usage()}",
        f"  detect.py       : {program_state()}",
        "",
        "=" * 50,
        f"LAST 20 LOG ENTRIES  ({LOG_FILE})",
        "",
        last_log_lines(20),
        "",
        "=" * 50,
        "To deactivate test mode, run:",
        "  rm ~/catDetect/.test_mode",
    ]
    return "\n".join(lines)

# ── Send ──────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str):
    msg = MIMEMultipart()
    msg["From"]    = SMTP_USER
    msg["To"]      = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not FLAG_FILE.exists():
        print("Test mode not active — skipping.")
        return

    if not SMTP_USER or not SMTP_PASSWORD:
        print("SMTP credentials missing in .env")
        return

    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"CatDetect Health Check — {now}"
    body    = build_body()

    send_email(subject, body)
    print(f"Health email sent to {EMAIL_TO}")
    print(body)

if __name__ == "__main__":
    main()
