# Système de Détection 3D des Façades de Bâtiments

## Description du Projet

Projet de portfolio en vision par ordinateur. Il détecte les façades de bâtiments en fusionnant un nuage de points LiDAR et une image 2D. YOLO détecte les façades dans l'image. Le LiDAR extrait les plans verticaux. La fusion recale les deux quand une calibration caméra est disponible.

Première version en 2025. Améliorations en 2026 : jeu KITTI-360, calibration nuScenes, pseudo-labels projetés depuis le LiDAR, et dépôt allégé pour GitHub.

## Technologies Utilisées

- **Python** : Langage de programmation principal
- **LiDAR** : Acquisition et traitement de nuages de points 3D
- **YOLO** : Modèle de détection d'objets pour la détection de façades
- **OpenCV** : Traitement d'images et vision par ordinateur
- **Traitement de nuages de points** : Analyse et traitement des données 3D

## Structure du Projet

```
.
├── src/                         # Pipeline : LiDAR, YOLO, fusion, visualisation
├── scripts/                     # Préparation des datasets et pseudo-labels
├── examples/                    # Exemples en ligne de commande
├── config/                      # YAML du pipeline et des datasets YOLO
├── utils/                       # nuScenes mini, calibration, helpers
├── notebooks/                   # Démonstration
├── tests/                       # Tests du traitement LiDAR
├── data/                        # Données locales (non versionnées)
├── datasets/                    # Jeux YOLO générés localement (non versionnés)
├── models/                      # Poids entraînés (non versionnés)
└── results/                     # Sorties de visualisation (non versionnées)
```

Les archives KITTI-360, nuScenes mini (`v1.0-mini.tgz`, `nuscmini/`) et les poids `.pt` restent sur la machine. GitHub refuse les fichiers de plus de 100 Mo, et ce dossier local dépasse 20 Go.

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Pour un GPU NVIDIA, installer PyTorch avec CUDA avant le reste des dépendances : https://pytorch.org/get-started/locally/

## Données

Lancer les commandes depuis la racine du projet.

**nuScenes mini** (paires LiDAR + caméra, calibration réelle) : télécharger `v1.0-mini.tgz` sur https://www.nuscenes.org/nuscenes puis :

```bash
python utils/extract_v1_mini.py --archive v1.0-mini.tgz --out nuscmini
python examples/basic_usage.py --nusc-root nuscmini --nusc-cam CAM_FRONT
```

**KITTI-360** (détection de façades entraînée sur la classe sémantique *building*, id 11) : placer `kitti360/2013_05_28_drive_0000_sync_image_00.zip` et `data_2d_semantics.zip` à la racine, puis :

```bash
python scripts/kitti360_to_yolo.py
python train_yolo_facade.py --data config/dataset_kitti360_facade.yaml --epochs 30 --name yolo_facade_kitti
```

Les fichiers `config/dataset_*.yaml` utilisent des chemins relatifs à la racine du projet.

## Utilisation

### Traitement des données LiDAR

```python
from src.lidar_processing import LiDARProcessor

processor = LiDARProcessor()
point_cloud = processor.load_point_cloud("data/raw/building.las")
processed_cloud = processor.preprocess(point_cloud)
facades = processor.detect_facades(processed_cloud)
```

### Détection YOLO

```python
from src.yolo_detection import YOLODetector

detector = YOLODetector(model_path="models/yolo_facade.pt")
detections = detector.detect("data/images/building.jpg")
```

### Fusion LiDAR + Images 2D

```python
from src.fusion import DataFusion

fusion = DataFusion()
result = fusion.fuse_lidar_images(
    point_cloud="data/processed/building.pcd",
    image="data/images/building.jpg",
    camera_params="config/camera.yaml"
)
```

### Pipeline complet de détection

```python
from src.facade_detection import FacadeDetectionPipeline

pipeline = FacadeDetectionPipeline(config="config/pipeline_config.yaml")
results = pipeline.process(
    lidar_file="data/raw/building.las",
    image_file="data/images/building.jpg"
)
pipeline.visualize(results, output_path="results/detection_result.png")
```

## Fonctionnalités Principales

1. **Traitement LiDAR** :
   - Chargement de fichiers LiDAR (.las, .laz, .ply, .pcd)
   - Préprocessing des nuages de points
   - Segmentation et extraction de façades
   - Filtrage du bruit et normalisation

2. **Détection YOLO** :
   - Chargement de modèles YOLO pré-entraînés
   - Détection de façades dans les images 2D
   - Fine-tuning sur données personnalisées
   - Export des résultats en différents formats

3. **Fusion de données** :
   - Calibration caméra-LiDAR
   - Projection 3D vers 2D
   - Fusion des détections 2D et 3D
   - Validation croisée des résultats

4. **Visualisation** :
   - Visualisation 3D des nuages de points
   - Superposition des détections sur les images
   - Export des visualisations 2D et 3D

## Configuration

Les fichiers de configuration se trouvent dans le dossier `config/`. Vous pouvez modifier les paramètres de traitement, les chemins des modèles, et les hyperparamètres selon vos besoins.

## Entraînement du Modèle YOLO

Pour entraîner ou fine-tuner un modèle YOLO sur vos propres données :

```bash
python train_yolo_facade.py --data config/dataset_facade.yaml --epochs 100 --imgsz 640
```

## Tests

Exécuter les tests unitaires :

```bash
pytest tests/
```

## Auteur

Nouhaila Berghane. Projet personnel de portfolio, commencé en 2025 et amélioré en 2026.

## Licence

MIT. Voir le fichier `LICENSE`.


