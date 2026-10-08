# Modèles

Les poids ne sont pas versionnés (fichiers `.pt`).

Après entraînement, Ultralytics écrit par exemple :

```
models/yolo_facade_kitti/weights/best.pt
```

Le pipeline lit ce chemin dans `config/pipeline_config.yaml`. S'il est absent, `FacadeDetectionPipeline` retombe sur `models/yolo_facade_v2/weights/best.pt`, puis `models/yolo_facade/weights/best.pt`, puis télécharge `yolov8n.pt` (COCO, pas un détecteur de façades).

YOLOv8n de base se télécharge tout seul au premier lancement. Il ne détecte pas la classe `facade`.
