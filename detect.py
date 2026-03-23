"""
Cat detection logger using YOLOv8 + webcam.
Logs every cat-detection event to detections.csv with a timestamp.

Usage:
    python detect.py                    # run with default webcam (index 0)
    python detect.py --camera 1         # use a different camera index
    python detect.py --interval 5       # check every 5 seconds (saves CPU)
    python detect.py --show             # show live preview window
    python detect.py --phase before     # tag entries as 'before' phase
    python detect.py --phase after      # tag entries as 'after' phase
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

COCO_CAT_CLASS = 15          # YOLO COCO class index for 'cat'
LOG_FILE = "detections.csv"
MODEL_NAME = "yolov8n.pt"    # nano = fastest; swap for yolov8s.pt for better accuracy


def setup_log(path: str):
    """Create CSV with headers if it doesn't exist yet."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "phase", "cat_count", "confidences"])
        print(f"Created log file: {path}")


def log_event(path: str, phase: str, cat_count: int, confidences: list[float]):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            phase,
            cat_count,
            ";".join(f"{c:.2f}" for c in confidences),
        ])


def detect_cats(model: YOLO, frame) -> tuple[int, list[float]]:
    """Run inference and return (cat_count, [confidence, ...])."""
    results = model(frame, verbose=False)[0]
    cats = [
        box
        for box in results.boxes
        if int(box.cls) == COCO_CAT_CLASS
    ]
    confidences = [float(box.conf) for box in cats]
    return len(cats), confidences


def draw_boxes(frame, results):
    """Draw bounding boxes for cats on the frame."""
    for box in results.boxes:
        if int(box.cls) != COCO_CAT_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame, f"cat {conf:.0%}", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Cat detection logger")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between detection checks")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Minimum confidence threshold (0-1)")
    parser.add_argument("--phase", default="baseline",
                        help="Label for this recording phase, e.g. 'before' or 'after'")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window (requires display)")
    args = parser.parse_args()

    print(f"Loading model {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)

    setup_log(LOG_FILE)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print(f"Recording phase='{args.phase}' | interval={args.interval}s | "
          f"confidence>={args.confidence} | Ctrl+C to stop")

    last_check = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame — retrying ...")
                time.sleep(1)
                continue

            now = time.time()
            if now - last_check >= args.interval:
                last_check = now

                results_raw = model(frame, verbose=False)[0]
                cats = [
                    box for box in results_raw.boxes
                    if int(box.cls) == COCO_CAT_CLASS
                    and float(box.conf) >= args.confidence
                ]
                cat_count = len(cats)
                confidences = [float(b.conf) for b in cats]

                if cat_count > 0:
                    log_event(LOG_FILE, args.phase, cat_count, confidences)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"CAT DETECTED x{cat_count}  conf={confidences}")
                else:
                    # Log zero-count entries too (needed for occupancy rate)
                    log_event(LOG_FILE, args.phase, 0, [])

                if args.show:
                    annotated = draw_boxes(frame.copy(), results_raw)
                    cv2.imshow("catDetect", annotated)

            if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
