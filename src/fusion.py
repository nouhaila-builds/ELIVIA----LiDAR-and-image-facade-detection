"""
Module de fusion des données LiDAR et images 2D
Calibration caméra-LiDAR et projection 3D vers 2D
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFusion:
    """Classe pour la fusion de données LiDAR et images 2D"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialise le module de fusion
        
        Args:
            calibration_file: Chemin vers le fichier de calibration caméra-LiDAR (YAML)
        """
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rotation_matrix = None
        self.translation_vector = None
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, calibration_file: str) -> None:
        """
        Charge les paramètres de calibration depuis un fichier YAML
        
        Args:
            calibration_file: Chemin vers le fichier de calibration
        """
        logger.info(f"Chargement de la calibration: {calibration_file}")
        
        with open(calibration_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        camera = calib_data['camera']
        self.camera_matrix = np.array([
            [camera['fx'], 0, camera['cx']],
            [0, camera['fy'], camera['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.distortion_coeffs = np.array(
            calib_data['distortion'][0], dtype=np.float32
        )
        
        # Rotation et translation caméra -> LiDAR
        rot = calib_data['rotation'][0]
        if np.linalg.norm(rot) > 0:
            self.rotation_matrix, _ = cv2.Rodrigues(np.array(rot, dtype=np.float32))
        else:
            self.rotation_matrix = np.eye(3)
        
        self.translation_vector = np.array(
            calib_data['translation'][0], dtype=np.float32
        ).reshape(3, 1)
        
        logger.info("Calibration chargée avec succès")

    def set_calibration_matrices(
        self,
        camera_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        distortion_coeffs: Optional[np.ndarray] = None,
    ) -> None:
        """Applique une calibration déjà calculée (par exemple nuScenes)."""
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
        self.translation_vector = np.asarray(translation_vector, dtype=np.float32).reshape(3, 1)
        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros((5,), dtype=np.float32)
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float32).reshape(-1)
        logger.info("Calibration matricielle appliquée")
    
    def project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        return_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Projette des points 3D (LiDAR) vers des coordonnées 2D (image)
        
        Args:
            points_3d: Points 3D (N, 3)
            return_mask: Si True, retourne aussi un masque des points valides
            
        Returns:
            Points 2D projetés (N, 2) et optionnellement un masque (N,)
        """
        if self.camera_matrix is None:
            raise ValueError("Calibration non chargée. Appelez load_calibration() d'abord.")
        
        # Transformer les points du repère LiDAR vers le repère caméra
        points_cam = self.rotation_matrix @ points_3d.T + self.translation_vector
        points_cam = points_cam.T
        
        # Filtrer les points derrière la caméra (z < 0)
        valid_mask = points_cam[:, 2] > 0
        
        # Projection
        points_2d_homogeneous = self.camera_matrix @ points_cam.T
        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
        points_2d = points_2d.T
        
        if return_mask:
            return points_2d, valid_mask
        return points_2d
    
    def fuse_lidar_images(
        self,
        point_cloud_path: str,
        image_path: str,
        camera_params: Optional[str] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Fusionne les données LiDAR et images 2D
        
        Args:
            point_cloud_path: Chemin vers le nuage de points
            image_path: Chemin vers l'image
            camera_params: Chemin vers le fichier de calibration (si pas chargé)
            image_shape: Taille de l'image (height, width). Si None, chargée depuis l'image
            
        Returns:
            Dictionnaire contenant:
            - image: Image chargée
            - point_cloud: Nuage de points
            - projected_points: Points 3D projetés en 2D
            - valid_mask: Masque des points valides
        """
        # Charger la calibration si nécessaire
        if camera_params and self.camera_matrix is None:
            self.load_calibration(camera_params)
        
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        if image_shape is None:
            image_shape = (image.shape[0], image.shape[1])
        
        # Charger le nuage de points
        from .lidar_processing import LiDARProcessor
        processor = LiDARProcessor()
        point_cloud = processor.load_point_cloud(point_cloud_path)
        points_3d = np.asarray(point_cloud.points)
        
        # Projeter les points 3D vers 2D
        points_2d, valid_mask = self.project_3d_to_2d(points_3d, return_mask=True)
        
        # Filtrer les points en dehors de l'image
        in_image_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0])
        )
        valid_mask = valid_mask & in_image_mask
        
        logger.info(f"Points valides après projection: {np.sum(valid_mask)}/{len(points_3d)}")
        
        return {
            'image': image,
            'point_cloud': point_cloud,
            'projected_points': points_2d,
            'valid_mask': valid_mask,
            'points_3d': points_3d
        }
    
    def overlay_point_cloud_on_image(
        self,
        image: np.ndarray,
        points_2d: np.ndarray,
        valid_mask: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: int = 1
    ) -> np.ndarray:
        """
        Superpose les points LiDAR sur l'image
        
        Args:
            image: Image sur laquelle superposer
            points_2d: Points 2D projetés (N, 2)
            valid_mask: Masque des points valides (N,)
            colors: Couleurs des points (N, 3) en BGR. Si None, utilise une couleur par défaut
            point_size: Taille des points
            
        Returns:
            Image avec les points superposés
        """
        overlay_image = image.copy()
        valid_points_2d = points_2d[valid_mask].astype(int)
        
        if colors is not None:
            valid_colors = colors[valid_mask]
            # Convertir les couleurs de [0, 1] à [0, 255] si nécessaire
            if valid_colors.max() <= 1.0:
                valid_colors = (valid_colors * 255).astype(np.uint8)
        else:
            # Couleur par défaut: rouge
            valid_colors = np.array([(0, 0, 255)] * len(valid_points_2d), dtype=np.uint8)
        
        # Dessiner les points
        for i, point in enumerate(valid_points_2d):
            x, y = point
            color = tuple(int(c) for c in valid_colors[i])
            cv2.circle(overlay_image, (x, y), point_size, color, -1)
        
        return overlay_image
    
    def fuse_detections(
        self,
        lidar_detections: List[o3d.geometry.PointCloud],
        yolo_detections: List[Dict],
        fusion_method: str = "weighted_average"
    ) -> List[Dict]:
        """
        Fusionne les détections LiDAR et YOLO
        
        Args:
            lidar_detections: Liste de nuages de points représentant les façades détectées par LiDAR
            yolo_detections: Liste de détections YOLO (bbox, confidence, etc.)
            fusion_method: Méthode de fusion ('weighted_average', 'confidence_based')
            
        Returns:
            Liste de détections fusionnées
        """
        logger.info(f"Fusion de {len(lidar_detections)} détections LiDAR et {len(yolo_detections)} détections YOLO")
        
        fused_detections = []
        
        if fusion_method == "confidence_based":
            # Utiliser la détection avec la plus haute confiance
            for yolo_det in yolo_detections:
                fused_detections.append({
                    'source': 'yolo',
                    'bbox': yolo_det['bbox'],
                    'confidence': yolo_det['confidence'],
                    'class_name': yolo_det['class_name']
                })
        
        elif fusion_method == "weighted_average":
            # Combiner les détections avec des poids basés sur la confiance
            # Pour l'instant, on retourne les détections YOLO enrichies
            for yolo_det in yolo_detections:
                fused_detections.append({
                    'source': 'fused',
                    'bbox': yolo_det['bbox'],
                    'confidence': yolo_det['confidence'],
                    'class_name': yolo_det['class_name'],
                    'lidar_support': len(lidar_detections)  # Nombre de façades LiDAR supportant la détection
                })
        
        logger.info(f"{len(fused_detections)} détections fusionnées")
        return fused_detections
    
    def validate_detections(
        self,
        lidar_facades: List[o3d.geometry.PointCloud],
        yolo_detections: List[Dict],
        reprojection_error_threshold: float = 2.0
    ) -> Dict:
        """
        Valide les détections YOLO en les comparant avec les façades LiDAR
        
        Args:
            lidar_facades: Façades détectées par LiDAR
            yolo_detections: Détections YOLO
            reprojection_error_threshold: Seuil d'erreur de reprojection (pixels)
            
        Returns:
            Dictionnaire de validation avec métriques
        """
        logger.info("Validation croisée des détections...")
        
        # Calculer les centres des façades LiDAR projetés en 2D
        lidar_centers_2d = []
        for facade in lidar_facades:
            points_3d = np.asarray(facade.points)
            if len(points_3d) > 0:
                center_3d = np.mean(points_3d, axis=0)
                center_2d = self.project_3d_to_2d(center_3d.reshape(1, -1))
                lidar_centers_2d.append(center_2d[0])
        
        # Pour chaque détection YOLO, vérifier si elle est proche d'une façade LiDAR
        validated_detections = []
        for yolo_det in yolo_detections:
            bbox = yolo_det['bbox']
            yolo_center = np.array([
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ])
            
            # Le centre LiDAR projeté doit tomber dans la boîte YOLO (marge en pixels).
            x1, y1, x2, y2 = bbox
            margin = reprojection_error_threshold
            min_distance = float('inf')
            is_valid = False
            for lidar_center in lidar_centers_2d:
                distance = float(np.linalg.norm(yolo_center - lidar_center))
                min_distance = min(min_distance, distance)
                cx, cy = lidar_center
                if (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin):
                    is_valid = True
                    break
            
            validated_detections.append({
                **yolo_det,
                'validated': is_valid,
                'distance_to_lidar': min_distance
            })
        
        validation_metrics = {
            'total_detections': len(validated_detections),
            'validated_detections': sum(1 for d in validated_detections if d['validated']),
            'validation_rate': sum(1 for d in validated_detections if d['validated']) / len(validated_detections) if validated_detections else 0
        }
        
        logger.info(f"Taux de validation: {validation_metrics['validation_rate']:.2%}")
        
        return {
            'detections': validated_detections,
            'metrics': validation_metrics
        }


