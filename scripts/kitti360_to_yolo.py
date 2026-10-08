"""
Build a YOLO facade dataset from one KITTI-360 drive.

Reads the left-camera image zip and the semantic-label zip without unpacking
them fully. Building pixels (semantic id 11) become one box per connected region.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np

BUILDING_ID = 11
SEQUENCE = "2013_05_28_drive_0000_sync"
STRIDE = 40
MIN_AREA = 0.015
MAX_AREA = 0.70
MIN_SIDE = 0.04


def _frame_id(name: str) -> str | None:
    stem = Path(name).stem
    if stem.isdigit():
        return stem
    return None


def boxes_from_mask(mask: np.ndarray) -> list[list[float]]:
    building = (mask == BUILDING_ID).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(building, connectivity=8)
    height, width = mask.shape[:2]
    boxes = []
    for label_id in range(1, n_labels):
        x, y, w, h, area = stats[label_id]
        if area < MIN_AREA * height * width:
            continue
        if area > MAX_AREA * height * width:
            continue
        if w < MIN_SIDE * width or h < MIN_SIDE * height:
            continue
        boxes.append(
            [
                (x + w / 2) / width,
                (y + h / 2) / height,
                w / width,
                h / height,
            ]
        )
    return boxes


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    image_zip_path = root / "kitti360" / f"{SEQUENCE}_image_00.zip"
    label_zip_path = root / "data_2d_semantics.zip"
    out = root / "datasets" / "kitti360_facade"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    sem_prefix = f"data_2d_semantics/train/{SEQUENCE}/image_00/semantic/"
    img_prefix = f"{SEQUENCE}/image_00/data_rect/"

    with zipfile.ZipFile(label_zip_path) as label_zip:
        frames = sorted(
            frame
            for name in label_zip.namelist()
            if name.startswith(sem_prefix) and name.endswith(".png")
            if (frame := _frame_id(name))
        )
        chosen = frames[::STRIDE]
        cut = int(len(chosen) * 0.8)
        split_of = {frame: "train" if i < cut else "val" for i, frame in enumerate(chosen)}

        with zipfile.ZipFile(image_zip_path) as image_zip:
            names = set(image_zip.namelist())
            saved = {"train": 0, "val": 0}
            for frame in chosen:
                image_name = f"{img_prefix}{frame}.png"
                if image_name not in names:
                    continue
                mask = cv2.imdecode(
                    np.frombuffer(label_zip.read(f"{sem_prefix}{frame}.png"), np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                if mask is None:
                    continue
                boxes = boxes_from_mask(mask)
                if not boxes:
                    continue
                split = split_of[frame]
                image = image_zip.read(image_name)
                (out / "images" / split / f"{SEQUENCE}_{frame}.png").write_bytes(image)
                label_path = out / "labels" / split / f"{SEQUENCE}_{frame}.txt"
                label_path.write_text(
                    "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in boxes),
                    encoding="utf-8",
                )
                saved[split] += 1

    yaml_path = root / "config" / "dataset_kitti360_facade.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# KITTI-360 drive 0000, building masks converted to facade boxes",
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: facade",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"train={saved['train']} val={saved['val']} yaml={yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
