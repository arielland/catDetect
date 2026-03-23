#!/bin/bash
# One-shot setup script for catDetect on Raspberry Pi OS (Bookworm)
# Run once after cloning: bash setup.sh

set -e  # stop on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USERNAME="$(whoami)"

echo "==> Installing system packages..."
sudo apt update -qq
sudo apt install -y git python3-opencv python3-full

echo "==> Creating virtual environment (with system-site-packages for opencv)..."
python3 -m venv --system-site-packages "$SCRIPT_DIR/venv"

echo "==> Installing Python dependencies..."
"$SCRIPT_DIR/venv/bin/pip" install --upgrade pip -q
"$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "==> Setting up credentials file..."
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "    Created .env — please edit it with your Gmail credentials:"
    echo "    nano $SCRIPT_DIR/.env"
else
    echo "    .env already exists, skipping."
fi

echo ""
echo "==> Installing cron jobs..."
PYTHON="$SCRIPT_DIR/venv/bin/python"
CRON_DETECT="@reboot cd $SCRIPT_DIR && $PYTHON detect.py --phase before --interval 5 >> $SCRIPT_DIR/detect.log 2>&1"
CRON_EMAIL="0 7 * * * cd $SCRIPT_DIR && $PYTHON email_report.py >> $SCRIPT_DIR/email.log 2>&1"

# Add cron jobs only if not already present
CURRENT_CRON=$(crontab -l 2>/dev/null || true)

if echo "$CURRENT_CRON" | grep -q "detect.py"; then
    echo "    detect.py cron already exists, skipping."
else
    (echo "$CURRENT_CRON"; echo "$CRON_DETECT") | crontab -
    echo "    Added: detect.py on boot"
fi

if echo "$CURRENT_CRON" | grep -q "email_report.py"; then
    echo "    email_report.py cron already exists, skipping."
else
    (crontab -l 2>/dev/null || true; echo "$CRON_EMAIL") | crontab -
    echo "    Added: email_report.py at 7am daily"
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Next steps:"
echo "   1. Edit your credentials:  nano $SCRIPT_DIR/.env"
echo "   2. Test email:             $PYTHON $SCRIPT_DIR/email_report.py"
echo "   3. Test detection:         $PYTHON $SCRIPT_DIR/detect.py --phase before"
echo "   4. Reboot to auto-start:   sudo reboot"
echo ""
echo " To activate the venv manually:"
echo "   source $SCRIPT_DIR/venv/bin/activate"
