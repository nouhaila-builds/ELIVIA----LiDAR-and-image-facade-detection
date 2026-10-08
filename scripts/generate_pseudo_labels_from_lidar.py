"""
Génère des pseudo-labels YOLO (bounding boxes) depuis les détections LiDAR
en projetant les façades détectées dans le nuage de points vers l'image 2D.

Ce script est utile pour générer automatiquement des annotations à partir
des données LiDAR, puis les corriger manuellement si nécessaire.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lidar_processing import LiDARProcessor
from src.fusion import DataFusion
from utils.nuscenes_calib import compute_calibration_for_sample_data
from utils.nuscenes_mini import find_pairs


def facades_to_bbox_2d(facades_3d, fusion_module, image_shape):
    """
    Convertit les façades 3D détectées en bounding boxes 2D dans l'image
    
    Args:
        facades_3d: Liste de nuages de points (façades détectées)
        fusion_module: Module DataFusion avec calibration configurée
        image_shape: (height, width) de l'image
        
    Returns:
        Liste de bboxes au format YOLO: [[x_center, y_center, width, height], ...]
        (valeurs normalisées 0-1)
    """
    bboxes = []
    h, w = image_shape[:2]
    
    for facade in facades_3d:
        points_3d = np.asarray(facade.points)
        if len(points_3d) == 0:
            continue
        
        # Projeter les points 3D vers 2D
        points_2d, valid_mask = fusion_module.project_3d_to_2d(points_3d, return_mask=True)
        valid_points_2d = points_2d[valid_mask]
        
        if len(valid_points_2d) == 0:
            continue
        
        # Filtrer les points dans l'image
        in_image = (
            (valid_points_2d[:, 0] >= 0) & (valid_points_2d[:, 0] < w) &
            (valid_points_2d[:, 1] >= 0) & (valid_points_2d[:, 1] < h)
        )
        valid_points_2d = valid_points_2d[in_image]
        
        if len(valid_points_2d) < 30:
            continue
        
        # Percentiles: quelques points projetés hors façade ne doivent pas
        # étirer la boîte jusqu'aux bords de l'image.
        x_min, x_max = np.percentile(valid_points_2d[:, 0], [2, 98])
        y_min, y_max = np.percentile(valid_points_2d[:, 1], [2, 98])
        
        # Convertir en format YOLO (normalisé, centre + taille)
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        area = width * height
        
        # Trop petit, ou boîte qui recouvre presque toute l'image (mauvaise façade).
        if width < 0.03 or height < 0.05:
            continue
        if width > 0.95 or area > 0.55:
            continue
        bboxes.append([x_center, y_center, width, height])
    
    kept = []
    for bbox in _merge_overlapping_bboxes(bboxes):
        width, height = bbox[2], bbox[3]
        # La fusion de deux murs voisins ne doit pas recréer une boîte plein cadre.
        if width > 0.92 or height > 0.85 or width * height > 0.5:
            continue
        kept.append(bbox)
    return kept


def _yolo_to_xyxy(bbox):
    x, y, w, h = bbox
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _yolo_to_xyxy(a)
    bx1, by1, bx2, by2 = _yolo_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _enclosing(a, b):
    ax1, ay1, ax2, ay2 = _yolo_to_xyxy(a)
    bx1, by1, bx2, by2 = _yolo_to_xyxy(b)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def _merge_overlapping_bboxes(bboxes, iou_threshold: float = 0.4):
    merged = [list(b) for b in bboxes]
    changed = True
    while changed:
        changed = False
        next_boxes = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            current = box
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _iou(current, merged[j]) >= iou_threshold:
                    current = _enclosing(current, merged[j])
                    used[j] = True
                    changed = True
            next_boxes.append(current)
        merged = next_boxes
    return merged


def scene_id_from_filename(path: Path) -> str:
    """n015-2018-...__CAM_FRONT__timestamp.jpg -> n015-2018-..."""
    return path.name.split("__")[0]


def assign_scene_splits(pairs, stride: int = 1, val_fraction: float = 0.2):
    """
    Découpe par scène entière. Les images voisines d'une même scène
    ne se retrouvent pas à la fois en train et en val.

    Un même balayage LiDAR (plusieurs caméras) est gardé ou écarté ensemble.
    """
    from collections import defaultdict

    by_scene = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        by_scene[scene_id_from_filename(pair.cam_path)][str(pair.lidar_path)].append(pair)

    scenes = sorted(by_scene)
    if len(scenes) >= 2:
        n_val = max(1, int(round(len(scenes) * val_fraction)))
        val_scenes = set(scenes[-n_val:])
    else:
        val_scenes = set()

    assigned = []
    for scene in scenes:
        frame_keys = sorted(by_scene[scene])
        if stride > 1:
            frame_keys = frame_keys[::stride]
        split = "val" if scene in val_scenes else "train"
        frames = [by_scene[scene][key] for key in frame_keys]
        if not val_scenes and len(frames) >= 5:
            cut = max(1, int(round(len(frames) * val_fraction)))
            splits = ["train"] * (len(frames) - cut) + ["val"] * cut
        else:
            splits = [split] * len(frames)
        for frame_pairs, frame_split in zip(frames, splits):
            for pair in frame_pairs:
                assigned.append((pair, frame_split))
    return assigned


def _write_dataset_yaml(output_dir: Path) -> None:
    """Met à jour le yaml d'entraînement avec le dossier qui vient d'être écrit."""
    project_root = Path(__file__).resolve().parent.parent
    yaml_path = project_root / "config" / "dataset_facade_pseudo.yaml"
    dataset_path = output_dir.resolve()
    yaml_path.write_text(
        "\n".join(
            [
                "# Pseudo-labels façades projetés depuis les plans LiDAR nuScenes",
                f"path: {dataset_path}",
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
    print(f"  YAML: {yaml_path}")


def save_yolo_label(label_path: Path, bboxes: list, class_id: int = 0):
    """
    Sauvegarde les bounding boxes au format YOLO (.txt)
    
    Format YOLO: class_id x_center y_center width height (toutes valeurs normalisées 0-1)
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            # Format: class_id x_center y_center width height
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    print(f"  ✓ Sauvegardé: {label_path} ({len(bboxes)} bboxes)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Génère des pseudo-labels YOLO depuis les détections LiDAR (nuScenes)"
    )
    parser.add_argument(
        "--nusc-root",
        type=str,
        required=True,
        help="Chemin vers le dossier nuscmini/",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="CAM_FRONT",
        help="Canal caméra si --cams n'est pas fourni (default: CAM_FRONT)",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=None,
        help="Canaux caméra. Défaut avec --by-scene: CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/facades_pseudo",
        help="Dossier de sortie pour images + labels (default: datasets/facades_pseudo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Nombre de paires à traiter (0 = toutes, default: 10)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split du dataset si --by-scene n'est pas utilisé (default: train)",
    )
    parser.add_argument(
        "--by-scene",
        action="store_true",
        help="Répartit des scènes entières entre train et val, et ignore --split",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Avec --by-scene, garde 1 image sur N dans chaque scène (default: 1)",
    )
    
    args = parser.parse_args()
    
    nusc_root = Path(args.nusc_root)
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("Génération de pseudo-labels depuis LiDAR")
    print("=" * 60)
    print(f"Source: {nusc_root}")
    print(f"Sortie: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Limite: {args.limit} paires")
    print()
    
    if args.cams:
        cameras = args.cams
    elif args.by_scene:
        # Les façades LiDAR tombent surtout dans les caméras latérales.
        cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    else:
        cameras = [args.cam]

    pairs = []
    for camera in cameras:
        pairs.extend(find_pairs(nusc_root, cam_channel=camera, limit=args.limit))
    print(f"Caméras: {', '.join(cameras)}")
    
    if not pairs:
        print("❌ Aucune paire trouvée!")
        return 1
    
    if args.by_scene:
        assigned = assign_scene_splits(pairs, stride=max(1, args.stride))
        for split_name in ("train", "val"):
            for kind in ("images", "labels"):
                folder = output_dir / kind / split_name
                folder.mkdir(parents=True, exist_ok=True)
                for old in folder.glob("*"):
                    if old.is_file():
                        old.unlink()
    else:
        assigned = [(pair, args.split) for pair in pairs]
        images_dir = output_dir / "images" / args.split
        labels_dir = output_dir / "labels" / args.split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ {len(assigned)} paires à traiter\n")
    
    # Processeur LiDAR. Un balayage est segmenté une seule fois, puis
    # projeté dans chaque caméra qui le voit.
    lidar_processor = LiDARProcessor(voxel_size=0.15, normal_radius=0.5)
    facades_by_lidar = {}
    
    processed = 0
    skipped = 0
    
    for i, (pair, split_name) in enumerate(assigned):
        print(f"[{i+1}/{len(assigned)}] {split_name} {pair.cam_path.name}")
        
        try:
            # Charger l'image
            img = cv2.imread(str(pair.cam_path))
            if img is None:
                print(f"  ⚠️  Impossible de charger l'image: {pair.cam_path}")
                skipped += 1
                continue
            
            lidar_key = str(pair.lidar_path)
            if lidar_key not in facades_by_lidar:
                pcd = lidar_processor.load_point_cloud(lidar_key)
                processed_pcd = lidar_processor.preprocess(pcd)
                facades_by_lidar[lidar_key] = lidar_processor.detect_facades(
                    processed_pcd,
                    min_points=40,
                    angle_threshold=0.85,
                )
            facades = facades_by_lidar[lidar_key]
            
            if len(facades) == 0:
                print(f"  ⚠️  Aucune façade détectée, skip")
                skipped += 1
                continue
            
            # Calculer la calibration
            calib = compute_calibration_for_sample_data(
                nusc_root=nusc_root,
                lidar_sd_token=pair.lidar_sample_data_token,
                cam_sd_token=pair.cam_sample_data_token,
            )
            
            # Setup fusion
            fusion = DataFusion(calibration_file=None)
            fusion.camera_matrix = calib.K.astype(np.float32)
            fusion.distortion_coeffs = np.zeros((5,), dtype=np.float32)
            fusion.rotation_matrix = calib.R.astype(np.float32)
            fusion.translation_vector = calib.t.astype(np.float32)
            
            # Convertir façades 3D -> bboxes 2D
            bboxes = facades_to_bbox_2d(facades, fusion, img.shape)
            
            if len(bboxes) == 0:
                print(f"  ⚠️  Aucune bbox générée, skip")
                skipped += 1
                continue
            
            # Nom du fichier (basé sur le nom original)
            stem = pair.cam_path.stem
            images_dir = output_dir / "images" / split_name
            labels_dir = output_dir / "labels" / split_name
            
            # Sauvegarder l'image
            img_out = images_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_out), img)
            
            # Sauvegarder le label YOLO
            label_out = labels_dir / f"{stem}.txt"
            save_yolo_label(label_out, bboxes, class_id=0)
            
            processed += 1
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            skipped += 1
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("✅ Terminé!")
    print(f"  Traités: {processed}")
    print(f"  Ignorés: {skipped}")
    print(f"  Dataset: {output_dir}")
    _write_dataset_yaml(output_dir)
    print()
    print("📝 Prochaines étapes:")
    print("  1. Vérifiez les labels générés (éditez si nécessaire)")
    print("  2. Utilisez scripts/prepare_dataset.py pour valider")
    print("  3. Lancez train_yolo_facade.py pour entraîner")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

