"""
Module principal de détection de façades
Pipeline complet combinant LiDAR, YOLO et fusion
"""

import logging
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

from .lidar_processing import LiDARProcessor
from .yolo_detection import YOLODetector
from .fusion import DataFusion
from .visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacadeDetectionPipeline:
    """Pipeline complet pour la détection de façades"""
    
    def __init__(self, config: Optional[str] = None):
        """
        Initialise le pipeline de détection
        
        Args:
            config: Chemin vers le fichier de configuration YAML
        """
        self._config_file: Optional[Path] = None

        # Charger la configuration
        if config and Path(config).exists():
            self._config_file = Path(config).resolve()
            with open(self._config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Configuration par défaut
            self.config = self._default_config()
        
        # Initialiser les composants
        lidar_config = self.config.get('lidar', {})
        self.lidar_processor = LiDARProcessor(
            voxel_size=lidar_config.get('voxel_size', 0.05),
            normal_radius=lidar_config.get('normal_radius', 0.1)
        )
        
        yolo_config = self.config.get('yolo', {})
        model_path = self._resolve_model_path(yolo_config.get('model_path'))
        self.yolo_detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=yolo_config.get('confidence_threshold', 0.25),
            iou_threshold=yolo_config.get('iou_threshold', 0.45),
            image_size=yolo_config.get('image_size', 640),
            device=yolo_config.get('device', 'cpu')
        )
        
        fusion_config = self.config.get('fusion', {})
        self._fusion_enabled = fusion_config.get('enable_fusion', True)
        self._calibration_file_raw = fusion_config.get('calibration_file')
        calibration_file = self._resolve_existing_path(self._calibration_file_raw)

        # Important: dans un notebook, le CWD peut être `notebooks/`, donc on évite de
        # charger un fichier relatif à l'init. On chargera la calibration "lazy" plus tard,
        # seulement si la fusion est activée et que le fichier existe.
        if self._fusion_enabled and calibration_file is not None:
            self.data_fusion = DataFusion(calibration_file=str(calibration_file))
        else:
            if self._fusion_enabled and self._calibration_file_raw:
                logger.warning(
                    f"Calibration introuvable au démarrage: {self._calibration_file_raw}. "
                    "La fusion pourra être désactivée ou la calibration fournie via un chemin valide."
                )
            self.data_fusion = DataFusion(calibration_file=None)
        
        self.visualizer = Visualizer()
        
        logger.info("Pipeline de détection de façades initialisé")

    def _resolve_existing_path(self, maybe_path: Optional[str]) -> Optional[Path]:
        """
        Résout un chemin potentiellement relatif vers un fichier existant.

        Stratégie:
        - si chemin absolu et existe -> ok
        - sinon, tester:
          - CWD / path
          - dossier du fichier de config / path
          - parent du dossier de config / path (utile si config est dans ./config/)
        """
        if not maybe_path:
            return None
        p = Path(maybe_path)
        if p.is_absolute() and p.exists():
            return p

        candidates = [Path.cwd() / p]
        if self._config_file is not None:
            cfg_dir = self._config_file.parent
            candidates.extend([cfg_dir / p, cfg_dir.parent / p])

        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def _resolve_model_path(self, configured: Optional[str]) -> Optional[str]:
        """Résout le .pt configuré, sinon le dernier entraînement façade disponible."""
        resolved = self._resolve_existing_path(configured)
        if resolved is not None:
            return str(resolved)

        for fallback in (
            "models/yolo_facade_v2/weights/best.pt",
            "models/yolo_facade/weights/best.pt",
        ):
            candidate = self._resolve_existing_path(fallback)
            if candidate is not None:
                logger.info(
                    f"Modèle configuré introuvable ({configured}). Utilisation de {candidate}"
                )
                return str(candidate)
        return None
    
    def _default_config(self) -> Dict:
        """Retourne la configuration par défaut"""
        return {
            'lidar': {
                'voxel_size': 0.05,
                'normal_radius': 0.1,
                'facade_threshold': 0.85,
                'min_points_per_facade': 80
            },
            'yolo': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'image_size': 640,
                'device': 'cpu'
            },
            'fusion': {
                'enable_fusion': True,
                'reprojection_error_threshold': 2.0,
                'fusion_method': 'weighted_average'
            }
        }
    
    def process(
        self,
        lidar_file: str,
        image_file: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Traite une paire LiDAR/Image pour détecter les façades
        
        Args:
            lidar_file: Chemin vers le fichier LiDAR
            image_file: Chemin vers l'image
            output_dir: Dossier de sortie pour sauvegarder les résultats
            
        Returns:
            Dictionnaire contenant tous les résultats:
            - lidar_facades: Façades détectées par LiDAR
            - yolo_detections: Détections YOLO
            - fused_result: Résultat de la fusion
            - validation: Résultat de la validation croisée
        """
        logger.info("Démarrage du pipeline de détection...")
        
        results = {}
        
        # 1. Traitement LiDAR
        logger.info("Étape 1: Traitement LiDAR...")
        point_cloud = self.lidar_processor.load_point_cloud(lidar_file)
        processed_cloud = self.lidar_processor.preprocess(point_cloud)
        
        lidar_config = self.config.get('lidar', {})
        lidar_facades = self.lidar_processor.detect_facades(
            processed_cloud,
            min_points=lidar_config.get('min_points_per_facade', 80),
            angle_threshold=lidar_config.get('facade_threshold', 0.85)
        )
        results['lidar_facades'] = lidar_facades
        results['point_cloud'] = processed_cloud
        logger.info(f"✓ {len(lidar_facades)} façades détectées par LiDAR")
        
        # 2. Détection YOLO
        logger.info("Étape 2: Détection YOLO...")
        yolo_detections = self.yolo_detector.detect(image_file)
        results['yolo_detections'] = yolo_detections
        logger.info(f"✓ {len(yolo_detections)} détections YOLO")
        
        # 3. Fusion des données
        fusion_config = self.config.get('fusion', {})
        if fusion_config.get('enable_fusion', True):
            logger.info("Étape 3: Fusion des données...")
            
            # Calibration réelle nuScenes si les fichiers en font partie, sinon le YAML.
            from utils.nuscenes_calib import calibration_for_files

            nusc_calib = calibration_for_files(lidar_file, image_file)
            if nusc_calib is not None:
                self.data_fusion.set_calibration_matrices(
                    nusc_calib.K, nusc_calib.R, nusc_calib.t
                )
                logger.info("Calibration nuScenes appliquée pour cette paire")
            else:
                calibration_file = self._resolve_existing_path(fusion_config.get('calibration_file'))
                if calibration_file and self.data_fusion.camera_matrix is None:
                    self.data_fusion.load_calibration(str(calibration_file))
            
            # Fusionner les détections
            fused_detections = self.data_fusion.fuse_detections(
                lidar_facades,
                yolo_detections,
                fusion_method=fusion_config.get('fusion_method', 'weighted_average')
            )
            results['fused_detections'] = fused_detections
            
            # Validation croisée
            validation = self.data_fusion.validate_detections(
                lidar_facades,
                yolo_detections,
                reprojection_error_threshold=fusion_config.get('reprojection_error_threshold', 2.0)
            )
            results['validation'] = validation
            
            logger.info("✓ Fusion et validation terminées")
        else:
            results['fused_detections'] = []
            results['validation'] = {'metrics': {}}
        
        # 4. Sauvegarder les résultats si un dossier de sortie est spécifié
        if output_dir:
            self._save_results(results, image_file, output_dir)
        
        logger.info("Pipeline terminé avec succès")
        return results
    
    def _save_results(self, results: Dict, image_file: str, output_dir: str) -> None:
        """Sauvegarde les résultats dans le dossier de sortie"""
        from pathlib import Path
        import cv2
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder l'image annotée
        image = cv2.imread(image_file)
        if image is not None:
            annotated_image = self.yolo_detector.draw_detections(
                image,
                results['yolo_detections']
            )
            cv2.imwrite(str(output_path / "yolo_detections.png"), annotated_image)
        
        logger.info(f"Résultats sauvegardés dans: {output_dir}")
    
    def visualize(
        self,
        results: Dict,
        output_path: Optional[str] = None,
        show_3d: bool = True,
        show_2d: bool = True
    ) -> None:
        """
        Visualise les résultats de détection
        
        Args:
            results: Résultats retournés par process()
            output_path: Chemin pour sauvegarder la visualisation
            show_3d: Afficher la visualisation 3D
            show_2d: Afficher la visualisation 2D
        """
        if show_3d and 'point_cloud' in results:
            self.visualizer.visualize_point_cloud_3d(
                results['point_cloud'],
                results.get('lidar_facades', []),
                output_path=output_path
            )
        
        if show_2d:
            logger.info("Visualisation 2D disponible dans les résultats sauvegardés")


