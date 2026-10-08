# Journal des modifications

Tous les changements notables de ce projet seront documentés dans ce fichier.

## [1.1.0] - 2026

### Ajouté
- Construction d'un jeu YOLO à partir de KITTI-360 (`scripts/kitti360_to_yolo.py`)
- Calibration réelle nuScenes et overlay LiDAR sur l'image
- Pseudo-labels obtenus en projetant les façades LiDAR dans l'image

### Modifié
- Chemins de datasets relatifs à la racine du projet
- Archives, jeux de données, poids et résultats exclus du dépôt (limite GitHub de 100 Mo)
- Python minimum : 3.10

## [1.0.0] - 2025 (Version initiale)

### Ajouté
- Module de traitement LiDAR (`src/lidar_processing.py`)
  - Chargement de fichiers LAS/LAZ/PLY/PCD
  - Préprocessing des nuages de points (filtrage, downsampling, calcul des normales)
  - Détection de façades basée sur les normales
  - Segmentation des surfaces verticales

- Module de détection YOLO (`src/yolo_detection.py`)
  - Support de YOLOv8 via Ultralytics
  - Détection de façades dans les images 2D
  - Visualisation des détections
  - Support pour l'entraînement et le fine-tuning

- Module de fusion (`src/fusion.py`)
  - Calibration caméra-LiDAR
  - Projection 3D vers 2D
  - Fusion des détections LiDAR et YOLO
  - Validation croisée des détections

- Module principal (`src/facade_detection.py`)
  - Pipeline complet de détection
  - Intégration de tous les composants
  - Configuration via YAML

- Module de visualisation (`src/visualization.py`)
  - Visualisation 3D des nuages de points avec Open3D
  - Visualisation 2D avec superposition des détections
  - Graphiques de métriques de validation

- Configuration
  - Fichier de configuration du pipeline (`config/pipeline_config.yaml`)
  - Fichier de calibration caméra (`config/camera_calibration.yaml`)

- Documentation
  - README.md en français
  - Guide de démarrage rapide (QUICKSTART.md)
  - Exemples d'utilisation (`examples/`)
  - Notebook Jupyter de démonstration (`notebooks/`)

- Tests
  - Tests unitaires pour le traitement LiDAR (`tests/`)

- Utilitaires
  - Fonctions utilitaires (`utils/helpers.py`)

### Technologies utilisées
- Python 3.10+
- PyTorch / Torchvision
- Ultralytics YOLO
- OpenCV
- Open3D
- NumPy, SciPy
- Matplotlib

---

Projet personnel de portfolio. Version initiale en 2025, améliorée en 2026.


