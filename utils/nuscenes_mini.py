"""
Utilitaires nuScenes mini (v1.0-mini)

Permet:
- d'identifier automatiquement des paires (LIDAR_TOP, CAM_*) à partir des JSON
- de construire des chemins complets vers les fichiers dans le dataset extrait

Ce module évite d'ajouter une dépendance externe (nuscenes-devkit).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class NuScenesPair:
    """Une paire LiDAR + caméra pour un même sample."""

    sample_token: str
    lidar_path: Path
    cam_path: Path
    cam_channel: str
    lidar_sample_data_token: Optional[str] = None
    cam_sample_data_token: Optional[str] = None


def _load_json(nusc_root: Path, rel: str) -> list:
    p = nusc_root / rel
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _index_by_token(rows: list) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        tok = r.get("token")
        if tok:
            out[tok] = r
    return out


def find_pairs(
    nusc_root: Path,
    cam_channel: str = "CAM_FRONT",
    lidar_channel: str = "LIDAR_TOP",
    limit: int = 10,
) -> List[NuScenesPair]:
    """
    Retourne des paires (lidar, caméra) trouvées dans nuScenes mini.

    Args:
        nusc_root: Racine du dataset extrait (contient `samples/`, `sweeps/`, `v1.0-mini/`).
        cam_channel: Ex: CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK...
        lidar_channel: En général LIDAR_TOP.
        limit: Nombre max de paires retournées.
    """
    nusc_root = nusc_root.resolve()
    # Tables nuScenes nécessaires pour retrouver le "channel" (CAM_FRONT, LIDAR_TOP, ...)
    sample_data = _load_json(nusc_root, "v1.0-mini/sample_data.json")
    calibrated_sensor = _load_json(nusc_root, "v1.0-mini/calibrated_sensor.json")
    sensor = _load_json(nusc_root, "v1.0-mini/sensor.json")

    cal_by_token = _index_by_token(calibrated_sensor)
    sensor_by_token = _index_by_token(sensor)

    # Regrouper par sample_token + channel (déduit via calibrated_sensor -> sensor)
    # nuScenes contient des "key frames" (dossier samples/) et des "sweeps" (dossier sweeps/).
    # Pour des résultats reproductibles et cohérents, on préfère les key frames quand elles existent.
    by_sample: Dict[str, Dict[str, dict]] = {}
    for sd in sample_data:
        samp = sd.get("sample_token")
        cal_tok = sd.get("calibrated_sensor_token")
        fname = sd.get("filename")
        if not samp or not cal_tok or not fname:
            continue
        cal = cal_by_token.get(cal_tok)
        if not cal:
            continue
        sensor_tok = cal.get("sensor_token")
        sen = sensor_by_token.get(sensor_tok) if sensor_tok else None
        chan = sen.get("channel") if sen else None
        if not chan:
            continue
        cur = by_sample.setdefault(samp, {}).get(chan)
        if cur is None:
            by_sample[samp][chan] = sd
        else:
            # préférer is_key_frame=True
            cur_key = bool(cur.get("is_key_frame", False))
            sd_key = bool(sd.get("is_key_frame", False))
            if sd_key and not cur_key:
                by_sample[samp][chan] = sd

    pairs: List[NuScenesPair] = []
    for sample_token, channels in by_sample.items():
        if lidar_channel not in channels or cam_channel not in channels:
            continue
        lidar_rel = channels[lidar_channel].get("filename")
        cam_rel = channels[cam_channel].get("filename")
        if not lidar_rel or not cam_rel:
            continue
        lidar_path = nusc_root / lidar_rel
        cam_path = nusc_root / cam_rel
        if not lidar_path.exists() or not cam_path.exists():
            continue
        pairs.append(
            NuScenesPair(
                sample_token=sample_token,
                lidar_path=lidar_path,
                cam_path=cam_path,
                cam_channel=cam_channel,
                lidar_sample_data_token=channels[lidar_channel].get("token"),
                cam_sample_data_token=channels[cam_channel].get("token"),
            )
        )
        if limit > 0 and len(pairs) >= limit:
            break

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Lister des paires LiDAR+caméra dans nuScenes mini.")
    parser.add_argument(
        "--root",
        required=True,
        help=r"Chemin vers le dataset extrait (ex: C:\nuscmini).",
    )
    parser.add_argument("--cam", default="CAM_FRONT", help="Canal caméra (ex: CAM_FRONT).")
    parser.add_argument("--lidar", default="LIDAR_TOP", help="Canal LiDAR (ex: LIDAR_TOP).")
    parser.add_argument("--limit", type=int, default=5, help="Nombre de paires à afficher.")
    args = parser.parse_args()

    pairs = find_pairs(Path(args.root), cam_channel=args.cam, lidar_channel=args.lidar, limit=args.limit)
    if not pairs:
        print("Aucune paire trouvée. Vérifie que le dataset est bien extrait et que --root pointe au bon endroit.")
        return 2

    for i, p in enumerate(pairs, start=1):
        print(f"[{i}] sample_token={p.sample_token}")
        print(f"    lidar: {p.lidar_path}")
        print(f"    cam  : {p.cam_path} ({p.cam_channel})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


