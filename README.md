# catDetect

An automated cat deterrent system using a Raspberry Pi 4, a USB webcam, and a buzzer.
Detects cats on an outdoor sofa using a YOLOv8n ONNX model, logs every event with a timestamped snapshot, and emails daily reports so you can measure the before/after effectiveness of your deterrent.

---

## How It Works

1. The webcam continuously captures frames
2. Every `--interval` seconds, a frame is run through a YOLOv8n ONNX object detection model
3. If one or more cats are detected above the confidence threshold:
   - The event is logged to `detections.csv` with timestamp, phase, count, and confidence
   - An annotated snapshot is saved to `snapshots/`
   - The buzzer beeps 3 times to deter the cat
4. Every morning at 7am, an email is sent with a stats summary and the full CSV log attached
5. An optional **test mode** sends an hourly health-check email including system stats and recent snapshots

---

## Hardware

| Component | Details |
|-----------|---------|
| Raspberry Pi 4 | Main compute unit |
| Microsoft LifeCam VX-800 | USB webcam, mapped to `/dev/catcam` via udev |
| Active buzzer | Connected to GPIO 18 (BCM) via NPN transistor driver circuit |
| NPN transistor | Drives the buzzer (protects RPi GPIO from current draw) |
| Flyback diode | Protection across the buzzer against back-EMF |
| Electrolytic capacitor | Smoothing on the power rail |
| Breadboard + jumper wires | Prototyping |
| Battery pack (optional) | For motor power (future use) |

### Wiring (Buzzer Circuit)

```
RPi Pin 2  (5V)   ──── Buzzer(+), Diode cathode, Capacitor(+)
RPi Pin 6  (GND)  ──── Transistor emitter, Capacitor(-)
RPi Pin 12 (GPIO 18) ── Resistor ── Transistor base
Diode anode        ──── Buzzer(-), Transistor collector
```

### Stable Camera Device

A udev rule maps the LifeCam to a permanent device path `/dev/catcam` regardless of boot order:

```
/etc/udev/rules.d/99-catcam.rules
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="0766", SYMLINK+="catcam"
```

---

## Software Components

| File | Purpose |
|------|---------|
| `detect.py` | Main detection loop — camera → ONNX inference → log + snapshot + buzzer |
| `email_report.py` | Daily summary email with CSV attachment and cat snapshots |
| `test_mode_email.py` | Hourly health-check email (CPU temp, RAM, process state, snapshots) |
| `stream.py` | Live MJPEG video stream served over HTTP (browser viewable) |
| `export_model.py` | One-time script to export `yolov8n.pt` → `yolov8n.onnx` (run on Windows) |
| `setup.sh` | Full RPi setup script — installs dependencies, creates venv, configures cron |
| `requirements.txt` | Python dependencies (onnxruntime, numpy, pandas) |

### Key Dependencies

- **onnxruntime** — lightweight ONNX inference engine (~200MB vs PyTorch's ~2GB)
- **OpenCV** — camera capture and image processing (installed via `apt`)
- **RPi.GPIO** — GPIO control for the buzzer (installed via `apt`)
- **pandas** — CSV log analysis for email reports
- **YOLOv8n ONNX** — nano-sized YOLO model trained on COCO (includes "cat" class 15)

---

## Initial Setup (Fresh RPi)

### 1. Flash the SD card
- Download **Raspberry Pi Imager** from raspberrypi.com/software
- Choose: Raspberry Pi 4 → RPi OS 64-bit Lite
- In settings: set hostname, username/password, WiFi, timezone, enable SSH
- Write and insert into RPi

### 2. First SSH connection
```bash
ssh YOUR_USERNAME@catdetect.local
# Accept the fingerprint when prompted
```

### 3. Install git and clone
```bash
sudo apt install -y git
git clone https://github.com/arielland/catDetect.git
cd catDetect
bash setup.sh
```

### 4. Export the ONNX model (one-time, on the RPi)
```bash
mkdir -p ~/tmp
TMPDIR=~/tmp venv/bin/pip install ultralytics
venv/bin/python export_model.py
TMPDIR=~/tmp venv/bin/pip uninstall -y ultralytics torch torchvision torchaudio
ls -lh yolov8n.onnx   # should show ~12MB
```

### 5. Configure email
```bash
nano .env
```
```ini
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_app_password   # Gmail App Password (not your account password)
EMAIL_TO=your@gmail.com
```
> To create a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords

### 6. Test email
```bash
source venv/bin/activate
python email_report.py
```

### 7. Reboot to auto-start
```bash
sudo reboot
```

---

## Running detect.py

```bash
cd ~/catDetect && source venv/bin/activate

# Basic run (before installing deterrent)
python detect.py --phase before --interval 30 --buzzer

# Full options
python detect.py \
  --camera /dev/catcam   \   # camera device path
  --phase before         \   # 'before' or 'after' (for comparison)
  --interval 30          \   # seconds between checks
  --confidence 0.2       \   # detection confidence threshold (0.0–1.0)
  --buzzer               \   # activate GPIO 18 buzzer on detection
  --model yolov8n.onnx       # path to ONNX model
```

---

## Cron Jobs (auto-configured by setup.sh)

| Schedule | Command |
|----------|---------|
| On every boot | `detect.py --camera /dev/catcam --phase before --interval 30 --buzzer --confidence 0.2` |
| Daily at 07:00 | `email_report.py` — morning summary with CSV + snapshots |
| Every hour | `test_mode_email.py` — health check (only runs if test mode is active) |

---

## Before / After Experiment

The system supports tagging detections with a phase label to measure deterrent effectiveness.

**Phase 1 — Baseline (before deterrent):**
Detection runs with `--phase before`. Logs cat visits over several days.

**Phase 2 — After deterrent:**
Once your deterrent is installed, SSH in and update the cron job:
```bash
crontab -e
# Change: --phase before  →  --phase after
```
Then reboot. The morning email report will automatically compare both phases.

---

## Testing & Diagnostics

### Check system status
```bash
# Via SSH
pgrep -a python                             # check if detect.py is running
tail -f ~/catDetect/detect.log              # live log output
tail -20 ~/catDetect/detections.csv         # recent detections
df -h /                                     # disk space
vcgencmd measure_temp                       # CPU temperature
```

### Take a single snapshot
```bash
cd ~/catDetect && source venv/bin/activate
python detect.py --snapshot
# Saves annotated snapshot.jpg — copy to Windows with:
# scp YOUR_USERNAME@catdetect.local:~/catDetect/snapshot.jpg C:\Users\user\Desktop\
```

### Live video stream (browser)
```bash
# On the RPi (stops detect.py automatically):
cd ~/catDetect && venv/bin/python stream.py

# Then open in browser on Windows:
# http://catdetect.local:8080
# Press Ctrl+C on the RPi to stop. Then restart detect.py:
nohup venv/bin/python -u detect.py --camera /dev/catcam --phase before --interval 30 --buzzer --confidence 0.2 >> detect.log 2>&1 &
```

### Test the buzzer manually
```bash
cd ~/catDetect && source venv/bin/activate
python -c "
import RPi.GPIO as GPIO, time
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
for i in range(3):
    GPIO.output(18, GPIO.HIGH); time.sleep(0.3)
    GPIO.output(18, GPIO.LOW);  time.sleep(0.2)
GPIO.cleanup()
print('Done')
"
```

### Test mode email (hourly health checks)
```bash
# Activate test mode:
touch ~/catDetect/.test_mode

# Deactivate (takes effect next hour, no reboot needed):
rm ~/catDetect/.test_mode

# Trigger immediately:
cd ~/catDetect && venv/bin/python test_mode_email.py
```
Test mode emails include: CPU temperature, CPU load, RAM usage, detect.py running state, and the last 10 cat snapshots attached.

### Send daily report immediately
```bash
cd ~/catDetect && venv/bin/python email_report.py
# Options:
python email_report.py --days 7              # include last 7 days
python email_report.py --phases before after # compare phases explicitly
```

---

## Log Files

| File | Contents |
|------|---------|
| `detections.csv` | timestamp, phase, cat_count, confidences |
| `detect.log` | stdout from detect.py (model load, detections, errors) |
| `email.log` | daily email send log |
| `test_mode.log` | hourly test mode email log |
| `snapshots/cat_YYYY-MM-DDTHH-MM-SS.jpg` | annotated snapshots on detection (max 50 kept) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ssh: Permission denied` | Wrong password — use the one set in Raspberry Pi Imager |
| `catdetect.local` not found | Install Bonjour on Windows, or use the RPi's IP address |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Run `ssh-keygen -R catdetect.local` then reconnect |
| Camera not found | Check `v4l2-ctl --list-devices` and verify `/dev/catcam` symlink exists |
| `No space left on device` | Run `venv/bin/pip cache purge && rm -rf ~/tmp` |
| detect.py not running after reboot | Check `tail -20 ~/catDetect/detect.log` for startup errors |
| Cat visible but not detected | Lower `--confidence` (try 0.15), improve camera angle/lighting |
| Buzzer not beeping | Check GPIO 18 wiring and that `--buzzer` flag is passed |
