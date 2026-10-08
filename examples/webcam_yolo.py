"""
Détection YOLO en temps réel depuis la caméra (webcam) de l'ordinateur.

Usage (PowerShell):
  C:\venvs\fd\Scripts\python examples\webcam_yolo.py --device cpu

Notes:
- Par défaut, YOLODetector charge yolov8n.pt (peut nécessiter un téléchargement au premier lancement).
- Pour utiliser ton propre modèle entraîné (façades), passe --model models/yolo_facade.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.yolo_detection import YOLODetector


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO temps réel depuis webcam (OpenCV).")
    parser.add_argument("--camera-index", type=int, default=0, help="Index de la caméra (0, 1, ...).")
    parser.add_argument(
        "--model",
        default=None,
        help="Chemin vers un modèle YOLO .pt (ex: models/yolo_facade.pt). Si absent, utilise yolov8n.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance.")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image pour l'inférence.")
    parser.add_argument("--device", default="cpu", help="Device: 'cuda' ou 'cpu'.")
    parser.add_argument(
        "--save",
        default=None,
        help="Optionnel: chemin de sortie .mp4 pour enregistrer la vidéo annotée.",
    )
    args = parser.parse_args()

    # Initialiser YOLO
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra index={args.camera_index}. Essaie --camera-index 1")
        return 2

    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"⚠️  Impossible d'ouvrir l'encodeur vidéo pour: {out_path} (enregistrement désactivé)")
            writer = None

    print("Appuie sur 'q' pour quitter.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("⚠️  Frame non lue (fin ou erreur caméra).")
                break

            detections = detector.detect_from_array(frame)
            annotated = detector.draw_detections(frame, detections, show_labels=True, show_confidence=True)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("YOLO Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


