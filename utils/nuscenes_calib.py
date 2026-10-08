"""
Calibration nuScenes (v1.0-mini) sans dépendance externe.

Objectif: calculer la transformation LiDAR -> Caméra et récupérer la matrice intrinsèque K.

Références (structure nuScenes):
- v1.0-mini/sample_data.json : filename, ego_pose_token, calibrated_sensor_token
- v1.0-mini/calibrated_sensor.json : translation + rotation (quaternion) sensor->ego, camera_intrinsic (cam)
- v1.0-mini/ego_pose.json : translation + rotation (quaternion) ego->global

Convention:
On calcule T_cam_from_lidar (4x4) pour transformer un point LiDAR (x,y,z,1) en repère caméra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _load_json(nusc_root: Path, rel: str) -> list:
    p = nusc_root / rel
    return json.loads(p.read_text(encoding="utf-8"))


def _index_by_token(rows: list) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        tok = r.get("token")
        if tok:
            out[tok] = r
    return out


def _quat_wxyz_to_rotmat(q) -> np.ndarray:
    """
    Convertit un quaternion [w, x, y, z] en matrice 3x3.
    """
    w, x, y, z = map(float, q)
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_from_rot_trans(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass(frozen=True)
class NuScenesCalibration:
    K: np.ndarray  # (3,3)
    T_cam_from_lidar: np.ndarray  # (4,4)

    @property
    def R(self) -> np.ndarray:
        return self.T_cam_from_lidar[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T_cam_from_lidar[:3, 3].reshape(3, 1)


def compute_calibration_for_sample_data(
    nusc_root: Path,
    lidar_sd_token: str,
    cam_sd_token: str,
) -> NuScenesCalibration:
    """
    Calcule K et T_cam_from_lidar à partir de 2 sample_data tokens (LiDAR et Caméra).
    """
    nusc_root = nusc_root.resolve()

    sample_data = _index_by_token(_load_json(nusc_root, "v1.0-mini/sample_data.json"))
    calibrated_sensor = _index_by_token(_load_json(nusc_root, "v1.0-mini/calibrated_sensor.json"))
    ego_pose = _index_by_token(_load_json(nusc_root, "v1.0-mini/ego_pose.json"))

    lidar_sd = sample_data.get(lidar_sd_token)
    cam_sd = sample_data.get(cam_sd_token)
    if lidar_sd is None or cam_sd is None:
        raise KeyError("Tokens sample_data invalides (lidar ou caméra).")

    # --- LIDAR: sensor->ego, ego->global
    lidar_cal = calibrated_sensor[lidar_sd["calibrated_sensor_token"]]
    lidar_pose = ego_pose[lidar_sd["ego_pose_token"]]

    R_ego_from_lidar = _quat_wxyz_to_rotmat(lidar_cal["rotation"])
    t_ego_from_lidar = np.array(lidar_cal["translation"], dtype=np.float64)
    T_ego_from_lidar = _transform_from_rot_trans(R_ego_from_lidar, t_ego_from_lidar)

    R_global_from_ego_l = _quat_wxyz_to_rotmat(lidar_pose["rotation"])
    t_global_from_ego_l = np.array(lidar_pose["translation"], dtype=np.float64)
    T_global_from_ego_l = _transform_from_rot_trans(R_global_from_ego_l, t_global_from_ego_l)

    T_global_from_lidar = T_global_from_ego_l @ T_ego_from_lidar

    # --- CAM: sensor->ego, ego->global
    cam_cal = calibrated_sensor[cam_sd["calibrated_sensor_token"]]
    cam_pose = ego_pose[cam_sd["ego_pose_token"]]

    R_ego_from_cam = _quat_wxyz_to_rotmat(cam_cal["rotation"])
    t_ego_from_cam = np.array(cam_cal["translation"], dtype=np.float64)
    T_ego_from_cam = _transform_from_rot_trans(R_ego_from_cam, t_ego_from_cam)

    R_global_from_ego_c = _quat_wxyz_to_rotmat(cam_pose["rotation"])
    t_global_from_ego_c = np.array(cam_pose["translation"], dtype=np.float64)
    T_global_from_ego_c = _transform_from_rot_trans(R_global_from_ego_c, t_global_from_ego_c)

    T_global_from_cam = T_global_from_ego_c @ T_ego_from_cam
    T_cam_from_global = _invert_transform(T_global_from_cam)

    # --- Compose
    T_cam_from_lidar = T_cam_from_global @ T_global_from_lidar

    # Intrinsics camera
    K = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)

    return NuScenesCalibration(K=K, T_cam_from_lidar=T_cam_from_lidar)


def find_nusc_root(start: Path) -> Optional[Path]:
    """Remonte les dossiers jusqu'à une racine nuScenes mini (v1.0-mini/sample_data.json)."""
    for candidate in [start, *start.resolve().parents]:
        if (candidate / "v1.0-mini" / "sample_data.json").is_file():
            return candidate
    return None


def calibration_for_files(
    lidar_file: str,
    cam_file: str,
) -> Optional[NuScenesCalibration]:
    """
    Retrouve la calibration nuScenes à partir des chemins image et LiDAR.

    Retourne None si les fichiers ne sont pas dans un extrait v1.0-mini.
    """
    lidar_path = Path(lidar_file).resolve()
    cam_path = Path(cam_file).resolve()
    nusc_root = find_nusc_root(lidar_path.parent) or find_nusc_root(cam_path.parent)
    if nusc_root is None:
        return None

    sample_data = _load_json(nusc_root, "v1.0-mini/sample_data.json")
    lidar_token = _match_sample_data_token(sample_data, nusc_root, lidar_path)
    cam_token = _match_sample_data_token(sample_data, nusc_root, cam_path)
    if lidar_token is None or cam_token is None:
        return None

    return compute_calibration_for_sample_data(nusc_root, lidar_token, cam_token)


def _match_sample_data_token(sample_data: list, nusc_root: Path, file_path: Path) -> Optional[str]:
    """Associe un fichier du disque à son token sample_data via le chemin relatif nuScenes."""
    try:
        relative = file_path.resolve().relative_to(nusc_root.resolve()).as_posix()
    except ValueError:
        relative = None

    target_name = file_path.name
    for row in sample_data:
        filename = row.get("filename")
        token = row.get("token")
        if not filename or not token:
            continue
        if relative is not None and filename == relative:
            return token
        if filename.endswith(target_name):
            return token
    return None


