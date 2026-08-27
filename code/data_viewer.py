#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:44:02 2026

@author: christian
"""

import cv2
import json
from pathlib import Path
from dataclasses import dataclass


# ==========================================================
# COMMON DATA STRUCTURES
# ==========================================================

@dataclass
class Instance:
    class_id: int
    bbox: tuple

    keypoints: list | None = None
    polygon: list | None = None
    confidence: float | None = None


# ==========================================================
# BASE READER
# ==========================================================

class DatasetReader:

    def __len__(self):
        raise NotImplementedError

    def get_image(self, idx):
        raise NotImplementedError

    def get_instances(self, idx):
        raise NotImplementedError

    def get_filename(self, idx):
        raise NotImplementedError


# ==========================================================
# COCO READER
# ==========================================================

class CocoPoseReader(DatasetReader):

    def __init__(self, json_path, image_root):

        with open(json_path, "r") as f:
            self.coco = json.load(f)

        self.image_root = Path(image_root)

        self.images = sorted(
            self.coco["images"],
            key=lambda x: x["id"]
        )

        self.anns_by_image = {}

        for ann in self.coco["annotations"]:
            self.anns_by_image.setdefault(
                ann["image_id"], []
            ).append(ann)

    def __len__(self):
        return len(self.images)

    def get_filename(self, idx):
        return self.images[idx]["file_name"]

    def get_image(self, idx):

        img_info = self.images[idx]

        path = self.image_root / img_info["file_name"]

        return cv2.imread(str(path))

    def get_instances(self, idx):

        img_info = self.images[idx]

        anns = self.anns_by_image.get(
            img_info["id"], []
        )

        instances = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            bbox = (
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            )

            kpts = []

            raw = ann["keypoints"]

            for i in range(0, len(raw), 3):

                kx, ky, kv = raw[i:i+3]

                kpts.append((kx, ky, kv))

            instances.append(
                Instance(
                    class_id=ann["category_id"],
                    bbox=bbox,
                    keypoints=kpts
                )
            )

        return instances


# ==========================================================
# YOLO READER
# ==========================================================

class YoloPoseReader(DatasetReader):

    def __init__(self,
                 image_dir,
                 label_dir,
                 n_keypoints=15):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        exts = {".jpg", ".jpeg", ".png", ".bmp"}

        self.images = sorted(
            [
                p for p in self.image_dir.iterdir()
                if p.suffix.lower() in exts
            ]
        )

        self.n_keypoints = n_keypoints

    def __len__(self):
        return len(self.images)

    def get_filename(self, idx):
        return self.images[idx].name

    def get_image(self, idx):

        return cv2.imread(str(self.images[idx]))

    def get_instances(self, idx):

        img = self.get_image(idx)

        H, W = img.shape[:2]

        label_file = (
            self.label_dir /
            f"{self.images[idx].stem}.txt"
        )

        instances = []

        if not label_file.exists():
            return instances

        with open(label_file, "r") as f:

            for line in f:

                vals = line.strip().split()

                vals = list(map(float, vals))

                class_id = int(vals[0])

                xc, yc, bw, bh = vals[1:5]

                x1 = int((xc - bw/2) * W)
                y1 = int((yc - bh/2) * H)
                x2 = int((xc + bw/2) * W)
                y2 = int((yc + bh/2) * H)

                bbox = (x1, y1, x2, y2)

                kpts = []

                offset = 5

                for k in range(self.n_keypoints):

                    kx = vals[offset]
                    ky = vals[offset + 1]
                    kv = int(vals[offset + 2])

                    kpts.append(
                        (
                            kx * W,
                            ky * H,
                            kv
                        )
                    )

                    offset += 3

                instances.append(
                    Instance(
                        class_id=class_id,
                        bbox=bbox,
                        keypoints=kpts
                    )
                )

        return instances

# ==========================================================
# Detector READER
# ==========================================================

class DetectorReader(DatasetReader):

    def __init__(self, image_paths, detector):

        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        self.image_paths = [Path(p) for p in image_paths]
        self.detector = detector

    def __len__(self):
        return len(self.image_paths)

    def get_filename(self, idx):
        return self.image_paths[idx].name

    def get_image(self, idx):

        return cv2.imread(str(self.image_paths[idx]))

    def get_instances(self, idx):

        # Give YOLO the filename so it uses the same native file loader as
        # model.val(), instead of decoding once here and passing an ndarray.
        pred = self.detector.detect(self.image_paths[idx])
    
        instances = []
    
        boxes = pred.get("boxes", [])
        classes = pred.get("classes", [])
        scores = pred.get("scores", None)
        keypoints = pred.get("keypoints", None)
        polygons = pred.get("polygons", None)
    
        for i in range(len(boxes)):
    
            # -----------------------------
            # Bounding box
            # -----------------------------
            x1, y1, x2, y2 = boxes[i]
    
            bbox = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
    
            # -----------------------------
            # Keypoints (pose)
            # -----------------------------
            kpts = None
    
            if keypoints is not None:
    
                kpts = []
    
                for kp in keypoints[i]:
    
                    x = float(kp[0])
                    y = float(kp[1])
    
                    # confidence/visibility if available
                    if len(kp) >= 3:
                        v = float(kp[2])
                    else:
                        v = 2
    
                    kpts.append((x, y, v))
    
            # -----------------------------
            # Polygon (segmentation)
            # -----------------------------
            poly = None
    
            if polygons is not None:
                poly = polygons[i]
    
            # -----------------------------
            # Confidence
            # -----------------------------
            conf = None
    
            if scores is not None:
                conf = float(scores[i])
    
            # -----------------------------
            # Create instance
            # -----------------------------
            instances.append(
                Instance(
                    class_id=int(classes[i]),
                    bbox=bbox,
                    keypoints=kpts,
                    polygon=poly,
                    confidence=conf
                )
            )
    
        return instances





# ==========================================================
# DRAWING
# ==========================================================

CONNECT_LOGIC = [
    [0, 1],
    [0, 3],
    [0, 6],
    [0, 9],
    [0, 12],
    [1, 2],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [12, 13],
    [13, 14],
    [1, 3],
    [3, 6],
    [6, 9],
    [9, 12],
]


def draw_instances(img, instances):

    for inst in instances:

        x1, y1, x2, y2 = inst.bbox

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            str(inst.class_id),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Draw skeleton connections
        # ----------------------------------
        if inst.polygon is not None:
            print('implement me please')
            
        if inst.keypoints is not None:
            for a, b in CONNECT_LOGIC:
    
                if (
                    a < len(inst.keypoints)
                    and b < len(inst.keypoints)
                ):
    
                    xa, ya, va = inst.keypoints[a]
                    xb, yb, vb = inst.keypoints[b]
    
                    if va > 0 and vb > 0:
    
                        cv2.line(
                            img,
                            (int(xa), int(ya)),
                            (int(xb), int(yb)),
                            (255, 0, 0),  # blue
                            2,
                            cv2.LINE_AA
                        )

            # ----------------------------------
            # Draw keypoints
            # ----------------------------------
            for x, y, v in inst.keypoints:
    
                if v > 0:
    
                    cv2.circle(
                        img,
                        (int(x), int(y)),
                        4,
                        (0, 0, 255),  # red
                        -1
                    )

    return img


# ==========================================================
# VIEWER
# ==========================================================

def browse(
    reader,
    start_idx=0,
    window_name="Dataset Browser"
):

    idx = max(0, min(start_idx, len(reader)-1))

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    # Initial window size
    cv2.resizeWindow(
        window_name,
        1400,
        900
    )

    while True:

        img = reader.get_image(idx)

        instances = reader.get_instances(idx)

        vis = draw_instances(
            img.copy(),
            instances
        )

        cv2.putText(
            vis,
            f"{idx+1}/{len(reader)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            "A=prev D=next J=jump Q=quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------
        # Get current window size
        # ----------------------------------------
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(
                window_name
            )
        except:
            win_w, win_h = 1400, 900

        img_h, img_w = vis.shape[:2]

        scale = min(
            win_w / img_w,
            win_h / img_h
        )

        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        display = cv2.resize(
            vis,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKeyEx(0)

        if key in [ord("q"), 27]:
            break

        elif key == ord("d"):
            idx = min(
                idx + 1,
                len(reader)-1
            )

        elif key == ord("a"):
            idx = max(
                idx - 1,
                0
            )

        elif key == ord("j"):

            try:

                target = int(
                    input(
                        f"Jump to image [0-{len(reader)-1}]: "
                    )
                )

                idx = max(
                    0,
                    min(target, len(reader)-1)
                )

            except:
                pass

        elif key == 2555904:  # right arrow
            idx = min(
                idx + 1,
                len(reader)-1
            )

        elif key == 2424832:  # left arrow
            idx = max(
                idx - 1,
                0
            )

    cv2.destroyAllWindows()
    
def browse_predictions(
        image_paths,
        detector,
        window_name="Prediction Browser"
    ):
    """
    Browse predictions from either a YOLO detection or segmentation model.

    Parameters
    ----------
    image_paths : list[str]
        Images to browse.
    detector : Detector
        Detector object.
    window_name : str
        Window title.
    """

    reader = DetectorReader(
        image_paths=image_paths,
        detector=detector
    )

    browse(
        reader,
        start_idx=0,
        window_name=window_name
    )
