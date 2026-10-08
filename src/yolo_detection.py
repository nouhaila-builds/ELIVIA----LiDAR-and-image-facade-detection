"""
Module de détection YOLO pour la détection de façades dans les images 2D
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLODetector:
    """Classe pour la détection de façades avec YOLO"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "cuda"
    ):
        """
        Initialise le détecteur YOLO
        
        Args:
            model_path: Chemin vers le modèle YOLO pré-entraîné (.pt)
                      Si None, utilise le modèle YOLO8 par défaut
            confidence_threshold: Seuil de confiance pour les détections
            iou_threshold: Seuil IoU pour la suppression non-maximale
            image_size: Taille d'image pour l'inférence
            device: Device à utiliser ('cuda' ou 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        if str(device).startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA indisponible, le détecteur YOLO utilise le CPU")
                device = "cpu"
        self.device = device
        
        # Charger le modèle YOLO
        if model_path and Path(model_path).exists():
            logger.info(f"Chargement du modèle YOLO: {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info("Chargement du modèle YOLO par défaut (YOLOv8)")
            # Utiliser YOLOv8n par défaut (vous pouvez changer pour YOLOv8s, YOLOv8m, etc.)
            self.model = YOLO('yolov8n.pt')  # Téléchargera automatiquement si nécessaire
        
        logger.info(f"Détecteur YOLO initialisé (device: {device})")
    
    def detect(self, image_path: str) -> List[Dict]:
        """
        Détecte les façades dans une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Liste de dictionnaires contenant les détections:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': int,
                'class_name': str
            }, ...]
        """
        logger.info(f"Détection dans l'image: {image_path}")
        
        # Exécuter la détection
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device
        )
        
        # Parser les résultats
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class': int(box.cls[0].cpu().numpy()),
                    'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                }
                detections.append(detection)
        
        logger.info(f"{len(detections)} détections trouvées")
        return detections
    
    def detect_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        Détecte les façades dans plusieurs images
        
        Args:
            image_paths: Liste de chemins vers les images
            
        Returns:
            Liste de listes de détections (une liste par image)
        """
        logger.info(f"Traitement par lot de {len(image_paths)} images")
        all_detections = []
        
        for image_path in image_paths:
            detections = self.detect(image_path)
            all_detections.append(detections)
        
        return all_detections
    
    def detect_from_array(self, image: np.ndarray) -> List[Dict]:
        """
        Détecte les façades depuis un tableau numpy (image)
        
        Args:
            image: Image au format numpy array (BGR ou RGB)
            
        Returns:
            Liste de dictionnaires contenant les détections
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class': int(box.cls[0].cpu().numpy()),
                    'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Dessine les détections sur l'image
        
        Args:
            image: Image originale
            detections: Liste de détections
            show_labels: Afficher les labels
            show_confidence: Afficher les scores de confiance
            
        Returns:
            Image avec les détections dessinées
        """
        annotated_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Dessiner le rectangle
            color = (0, 255, 0)  # Vert
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Ajouter le label et la confiance
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(det['class_name'])
                if show_confidence:
                    label_parts.append(f"{det['confidence']:.2f}")
                
                label = " ".join(label_parts)
                
                # Calculer la taille du texte
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Dessiner le fond du texte
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Dessiner le texte
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return annotated_image
    
    def save_results(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: str,
        show_labels: bool = True
    ) -> None:
        """
        Sauvegarde l'image annotée avec les détections
        
        Args:
            image_path: Chemin vers l'image originale
            detections: Liste de détections
            output_path: Chemin de sortie
            show_labels: Afficher les labels
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        annotated_image = self.draw_detections(image, detections, show_labels=show_labels)
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Résultats sauvegardés: {output_path}")
    
    def train(
        self,
        data_config: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cuda"
    ) -> None:
        """
        Entraîne ou fine-tune le modèle YOLO
        
        Args:
            data_config: Chemin vers le fichier de configuration du dataset (YAML)
            epochs: Nombre d'époques
            imgsz: Taille des images
            batch: Taille du batch
            device: Device à utiliser
        """
        logger.info(f"Démarrage de l'entraînement (epochs: {epochs})")
        
        self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="models",
            name="yolo_facade"
        )
        
        logger.info("Entraînement terminé")


