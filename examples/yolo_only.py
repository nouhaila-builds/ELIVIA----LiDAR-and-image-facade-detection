"""
Exemple d'utilisation de la détection YOLO seule
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yolo_detection import YOLODetector
import cv2


def main():
    """Exemple d'utilisation du détecteur YOLO"""
    
    print("Exemple: Détection YOLO seule")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Détection YOLO seule sur une image.")
    parser.add_argument(
        "--image",
        default="data/images/building.jpg",
        help="Chemin vers l'image 2D (.jpg/.png).",
    )
    parser.add_argument(
        "--output",
        default="results/yolo/detections.jpg",
        help="Chemin de sortie pour l'image annotée.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device: 'cuda' ou 'cpu'.",
    )
    args = parser.parse_args()

    # Initialiser le détecteur
    detector = YOLODetector(
        model_path=None,  # Utilise YOLOv8 par défaut
        confidence_threshold=0.5,
        iou_threshold=0.45,
        image_size=640,
        device=args.device  # ou "cpu"
    )
    
    # Image à traiter
    image_file = args.image
    
    if not Path(image_file).exists():
        print(f"⚠️  Fichier non trouvé: {image_file}")
        print("   Placez votre image dans data/images/")
        return
    
    print(f"📁 Traitement de l'image: {image_file}")
    
    # Détection
    detections = detector.detect(image_file)
    print(f"   Détections trouvées: {len(detections)}")
    
    # Afficher les détections
    for i, det in enumerate(detections):
        print(f"\n   Détection {i+1}:")
        print(f"      Classe: {det['class_name']}")
        print(f"      Confiance: {det['confidence']:.2%}")
        print(f"      BBox: {det['bbox']}")
    
    # Visualiser les résultats
    image = cv2.imread(image_file)
    annotated_image = detector.draw_detections(
        image,
        detections,
        show_labels=True,
        show_confidence=True
    )
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), annotated_image)
    print(f"\n✅ Image annotée sauvegardée: {output_path}")
    
    # Afficher l'image (optionnel)
    # cv2.imshow("Détections", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


