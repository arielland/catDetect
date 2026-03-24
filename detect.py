"""
Cat detection logger using YOLOv8n ONNX + webcam.
Logs every cat-detection event to detections.csv with a timestamp.

Requires: yolov8n.onnx in the same directory (run export_model.py on Windows first)

Usage:
    python detect.py                    # run with default webcam (index 0)
    python detect.py --camera 1         # use a different camera index
    python detect.py --interval 5       # check every 5 seconds (saves CPU)
    python detect.py --show             # show live preview window
    python detect.py --phase before     # tag entries as 'before' phase
    python detect.py --phase after      # tag entries as 'after' phase
    python detect.py --snapshot         # save one frame as snapshot.jpg and exit
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

COCO_CAT_CLASS  = 15
LOG_FILE        = "detections.csv"
MODEL_PATH      = "yolov8n.onnx"
INPUT_SIZE      = 640


# ── Pre/post-processing ──────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    """Resize with grey padding to maintain aspect ratio."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (size - new_w) / 2
    pad_h = (size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (pad_w, pad_h)


def preprocess(frame: np.ndarray):
    img, ratio, (pad_w, pad_h) = letterbox(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]  # HWC -> BCHW
    return img, ratio, pad_w, pad_h


def postprocess(outputs, conf_threshold: float, ratio: float,
                pad_w: float, pad_h: float):
    """
    YOLOv8 ONNX output: [1, 84, 8400]
    84 = 4 box coords (cx, cy, w, h) + 80 COCO class scores
    """
    pred = outputs[0][0].T          # [8400, 84]
    boxes_cxywh = pred[:, :4]
    class_scores = pred[:, 4:]      # [8400, 80]

    class_ids   = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    # Keep only cats above threshold
    mask = (class_ids == COCO_CAT_CLASS) & (confidences >= conf_threshold)
    if not mask.any():
        return [], []

    boxes_cxywh = boxes_cxywh[mask]
    confidences = confidences[mask]

    # cx,cy,w,h -> x1,y1,x2,y2 (still in 640-px letterbox space)
    cx, cy, bw, bh = boxes_cxywh.T
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Remove letterbox padding and scale back to original image coords
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio

    # NMS
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(),
                                conf_threshold, iou_threshold=0.45)
    if len(indices) == 0:
        return [], []

    indices = np.array(indices).flatten()
    final_boxes = [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
                   for i in indices]
    final_confs = [float(confidences[i]) for i in indices]
    return final_boxes, final_confs


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_log(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "phase", "cat_count", "confidences"])
        print(f"Created log file: {path}")


def log_event(path: str, phase: str, cat_count: int, confs: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            phase,
            cat_count,
            ";".join(f"{c:.2f}" for c in confs),
        ])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cat detection logger (ONNX)")
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--interval",   type=float, default=2.0,
                        help="Seconds between detection checks")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Minimum confidence threshold (0-1)")
    parser.add_argument("--phase",      default="baseline",
                        help="Label for this recording phase, e.g. 'before' or 'after'")
    parser.add_argument("--show",       action="store_true",
                        help="Show live preview window (requires display)")
    parser.add_argument("--snapshot",   action="store_true",
                        help="Capture one frame, draw detections, save as snapshot.jpg and exit")
    parser.add_argument("--model",      default=MODEL_PATH,
                        help="Path to yolov8n.onnx")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model file '{args.model}' not found.")
        print("Run export_model.py on your Windows PC first, then copy it here:")
        print(f"  scp yolov8n.onnx relland@catdetector.local:~/catDetect/")
        return

    print(f"Loading {args.model} ...")
    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    print("Model loaded.")

    setup_log(LOG_FILE)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    # ── Snapshot mode ────────────────────────────────────────────────────────
    if args.snapshot:
        # Grab a few frames first so the camera has time to adjust exposure
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("ERROR: could not grab frame from camera.")
            return

        blob, ratio, pad_w, pad_h = preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        boxes, confs = postprocess(outputs, args.confidence, ratio, pad_w, pad_h)

        # Draw bounding boxes on the frame
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"cat {conf:.0%}", (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add timestamp and detection count
        label = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  cats={len(boxes)}"
        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_path = "snapshot.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Snapshot saved → {out_path}  (cats detected: {len(boxes)})")
        if boxes:
            print(f"  confidence: {[f'{c:.0%}' for c in confs]}")
        return

    print(f"Recording  phase='{args.phase}'  interval={args.interval}s  "
          f"confidence>={args.confidence}  |  Ctrl+C to stop")

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

                blob, ratio, pad_w, pad_h = preprocess(frame)
                outputs = session.run(None, {input_name: blob})
                boxes, confs = postprocess(
                    outputs, args.confidence, ratio, pad_w, pad_h)

                log_event(LOG_FILE, args.phase, len(boxes), confs)

                if boxes:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"CAT DETECTED x{len(boxes)}  "
                          f"conf={[f'{c:.0%}' for c in confs]}")

                if args.show:
                    for (x1, y1, x2, y2), conf in zip(boxes, confs):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"cat {conf:.0%}",
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2)
                    cv2.imshow("catDetect", frame)

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
