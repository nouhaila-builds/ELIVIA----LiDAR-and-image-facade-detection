"""
Script d'entraînement/fine-tuning YOLO pour la détection de façades
Basé sur Ultralytics YOLOv8

Usage:
    python train_yolo_facade.py --data config/dataset_facade.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.yolo_detection import YOLODetector


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement/fine-tuning YOLO pour détection de façades"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Chemin vers le fichier dataset.yaml (format YOLO)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'époques (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille des images (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Taille du batch (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda ou cpu, default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Modèle de base (default: yolov8n.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="models",
        help="Dossier du projet (default: models)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_facade",
        help="Nom de l'expérience (default: yolo_facade)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre l'entraînement depuis le dernier checkpoint",
    )
    
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Entraînement YOLO pour détection de façades")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Modèle de base: {args.model}")
    print(f"Époques: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print()
    
    # Vérifier que le fichier dataset existe
    if not Path(args.data).exists():
        print(f"❌ Erreur: Le fichier dataset '{args.data}' n'existe pas.")
        print("\n📝 Créez un fichier dataset.yaml avec la structure suivante:")
        print("""
# dataset.yaml
path: datasets/facades  # Chemin vers le dossier du dataset
train: images/train     # Chemin relatif vers les images d'entraînement
val: images/val         # Chemin relatif vers les images de validation
test: images/test       # (optionnel) Chemin vers les images de test

# Classes
names:
  0: facade

nc: 1  # Nombre de classes
""")
        return 1
    
    # Initialiser le détecteur
    detector = YOLODetector(model_path=None, device=args.device)
    
    # Charger le modèle de base
    from ultralytics import YOLO
    model = YOLO(args.model)
    
    print(f"✅ Modèle de base chargé: {args.model}")
    print()
    
    # Lancer l'entraînement
    print("🚀 Démarrage de l'entraînement...")
    print()
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            exist_ok=True,
            workers=0,
        )
        
        print()
        print("=" * 60)
        print("✅ Entraînement terminé!")
        print("=" * 60)
        print(f"Meilleur modèle: {args.project}/{args.name}/weights/best.pt")
        print(f"Dernier checkpoint: {args.project}/{args.name}/weights/last.pt")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

