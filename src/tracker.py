import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Simple heatmap + analytics from tracked boxes
def process_video(video_path: str, conf: float = 0.35, max_frames: int | None = None):
    model = YOLO("yolov8n.pt")  # auto-downloads
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    heat = np.zeros((H, W), dtype=np.float32)
    rows = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Track (ByteTrack is built-in via persist=True)
        results = model.track(frame, conf=conf, persist=True, verbose=False)
        r = results[0]

        # Boxes
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            ids = None
            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Heatmap: add a small blob at center
                cv2.circle(heat, (cx, cy), 18, 1.0, -1)

                class_name = model.names.get(int(clss[i]), str(int(clss[i])))
                track_id = int(ids[i]) if ids is not None else -1

                rows.append({
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,
                    "class": class_name,
                    "confidence": float(confs[i]),
                    "track_id": track_id,
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "cx": cx, "cy": cy
                })

        annotated = r.plot()  # nice annotated frame directly

        frame_idx += 1
        yield annotated, heat, rows

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

def analytics_from_rows(rows: list[dict]):
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["class","track_id","confidence"])
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    # Total detections per class
    per_class = df.groupby("class").size().reset_index(name="detections").sort_values("detections", ascending=False)

    # Unique tracks per class (how many distinct objects)
    tracks = df[df["track_id"] != -1].groupby("class")["track_id"].nunique().reset_index(name="unique_objects")
    tracks = tracks.sort_values("unique_objects", ascending=False)

    return df, per_class, tracks