"""
Démo : fusion LiDAR + image 2D sur nuScenes mini.

Ce script:
- trouve une paire (LIDAR_TOP + CAM_*) dans nuscmini/
- calcule la calibration réelle nuScenes (K + extrinsèques LiDAR->Cam)
- projette le LiDAR sur l'image et sauvegarde un overlay

Usage:
  python examples/nuscmini_fusion_overlay.py --nusc-root nuscmini --cam CAM_FRONT --out results/nuscmini_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lidar_processing import LiDARProcessor
from src.fusion import DataFusion
from utils.nuscenes_calib import compute_calibration_for_sample_data
from utils.nuscenes_mini import find_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay LiDAR->image (nuScenes mini).")
    parser.add_argument("--nusc-root", required=True, help="Chemin vers nuscmini/ (samples, sweeps, v1.0-mini).")
    parser.add_argument("--cam", default="CAM_FRONT", help="Canal caméra (CAM_FRONT, CAM_FRONT_RIGHT, ...).")
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Choisir une autre paire (0 = première). Utile pour tester une autre image/frame.",
    )
    parser.add_argument("--out", default="results/nuscmini_overlay.png", help="Chemin sortie image overlay.")
    parser.add_argument("--point-size", type=int, default=1, help="Taille des points projetés.")
    args = parser.parse_args()

    nusc_root = Path(args.nusc_root)
    if args.pair_index < 0:
        raise ValueError("--pair-index doit être >= 0")
    pairs = find_pairs(nusc_root, cam_channel=args.cam, limit=args.pair_index + 1)
    if not pairs:
        print("❌ Aucune paire trouvée. Vérifie --nusc-root et --cam.")
        return 2
    if args.pair_index >= len(pairs):
        print(f"❌ pair-index={args.pair_index} hors limite. Paires trouvées: {len(pairs)}")
        return 2
    pair = pairs[args.pair_index]
    if not pair.lidar_sample_data_token or not pair.cam_sample_data_token:
        print("❌ Tokens sample_data manquants (bug).")
        return 3

    calib = compute_calibration_for_sample_data(
        nusc_root=nusc_root,
        lidar_sd_token=pair.lidar_sample_data_token,
        cam_sd_token=pair.cam_sample_data_token,
    )

    # Charger image
    img = cv2.imread(str(pair.cam_path))
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {pair.cam_path}")

    # Charger LiDAR (.pcd.bin supporté)
    proc = LiDARProcessor(voxel_size=0.0, normal_radius=0.2)
    pcd = proc.load_point_cloud(str(pair.lidar_path))
    pts = np.asarray(pcd.points)

    # Setup fusion
    f = DataFusion(calibration_file=None)
    f.camera_matrix = calib.K.astype(np.float32)
    f.distortion_coeffs = np.zeros((5,), dtype=np.float32)
    f.rotation_matrix = calib.R.astype(np.float32)
    f.translation_vector = calib.t.astype(np.float32)

    pts2d, valid = f.project_3d_to_2d(pts, return_mask=True)
    h, w = img.shape[:2]
    in_img = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
    valid = valid & in_img

    overlay = f.overlay_point_cloud_on_image(img, pts2d, valid_mask=valid, point_size=args.point_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

    print("✅ Overlay sauvegardé:", out_path)
    print("LiDAR:", pair.lidar_path)
    print("Image:", pair.cam_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


