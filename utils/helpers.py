"""
Fonctions utilitaires pour le projet
"""

import numpy as np
from pathlib import Path
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Crée un répertoire s'il n'existe pas
    
    Args:
        path: Chemin du répertoire
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Vérifie qu'un fichier existe
    
    Args:
        file_path: Chemin du fichier
        
    Returns:
        True si le fichier existe, False sinon
    """
    return Path(file_path).exists()


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Retourne l'extension d'un fichier (sans le point)
    
    Args:
        file_path: Chemin du fichier
        
    Returns:
        Extension du fichier
    """
    return Path(file_path).suffix[1:].lower()


def filter_points_by_height(
    points: np.ndarray,
    min_height: float = None,
    max_height: float = None
) -> np.ndarray:
    """
    Filtre les points par hauteur (coordonnée Z)
    
    Args:
        points: Points 3D (N, 3)
        min_height: Hauteur minimale
        max_height: Hauteur maximale
        
    Returns:
        Masque booléen des points valides
    """
    mask = np.ones(len(points), dtype=bool)
    
    if min_height is not None:
        mask &= points[:, 2] >= min_height
    
    if max_height is not None:
        mask &= points[:, 2] <= max_height
    
    return mask


def calculate_iou_2d(box1: List[float], box2: List[float]) -> float:
    """
    Calcule l'Intersection over Union (IoU) de deux bounding boxes 2D
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calcul de l'intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calcul de l'union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


