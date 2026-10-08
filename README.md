# Elivia

LiDAR and image facade detection.

*English version below: [English](#english).*

## Description du projet

Projet de portfolio en vision par ordinateur. Elivia détecte les façades de bâtiments en fusionnant un nuage de points LiDAR et une image 2D. YOLO détecte les façades dans l'image. Le LiDAR extrait les plans verticaux. La fusion recale les deux quand une calibration caméra est disponible.

Première version en 2025. Améliorations en 2026 : jeu KITTI-360, calibration nuScenes, pseudo-labels projetés depuis le LiDAR, et dépôt allégé pour GitHub.

## Le nom

**Elivia** vient d’*élévation* : en architecture, l’élévation est le dessin qui montre une façade de face. Le projet reprend cette idée. À partir d’un nuage de points LiDAR et d’une image, il retrouve les façades d’un bâtiment.

## Technologies

- **Python**
- **LiDAR** et **Open3D** pour les nuages de points
- **YOLOv8** (Ultralytics) pour la détection 2D
- **OpenCV** pour l'image
- **PyTorch**, NumPy, SciPy

## Structure

```
.
├── src/          # Pipeline : LiDAR, YOLO, fusion, visualisation
├── scripts/      # Préparation des datasets et pseudo-labels
├── examples/     # Exemples en ligne de commande
├── config/       # YAML du pipeline et des datasets YOLO
├── utils/        # nuScenes mini, calibration, helpers
├── notebooks/    # Démonstration
├── tests/        # Tests du traitement LiDAR
├── data/         # Données locales, non versionnées
├── datasets/     # Jeux YOLO générés localement, non versionnés
├── models/       # Poids entraînés, non versionnés
└── results/      # Sorties de visualisation, non versionnées
```

Les archives KITTI-360, nuScenes mini (`v1.0-mini.tgz`, `nuscmini/`) et les poids `.pt` restent sur la machine. GitHub refuse les fichiers de plus de 100 Mo.

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

Lancer les commandes depuis la racine du projet. Les fichiers `config/dataset_*.yaml` utilisent des chemins relatifs à cette racine.

**nuScenes mini**, paires LiDAR et caméra avec calibration réelle. Télécharger `v1.0-mini.tgz` sur https://www.nuscenes.org/nuscenes puis :

```bash
python utils/extract_v1_mini.py --archive v1.0-mini.tgz --out nuscmini
python examples/basic_usage.py --nusc-root nuscmini --nusc-cam CAM_FRONT
```

**KITTI-360**, façades dérivées de la classe sémantique *building* (id 11). Placer `kitti360/2013_05_28_drive_0000_sync_image_00.zip` et `data_2d_semantics.zip` à la racine, puis :

```bash
python scripts/kitti360_to_yolo.py
python train_yolo_facade.py --data config/dataset_kitti360_facade.yaml --epochs 30 --name yolo_facade_kitti
```

## Utilisation

LiDAR :

```python
from src.lidar_processing import LiDARProcessor

processor = LiDARProcessor()
point_cloud = processor.load_point_cloud("data/raw/building.las")
processed_cloud = processor.preprocess(point_cloud)
facades = processor.detect_facades(processed_cloud)
```

YOLO :

```python
from src.yolo_detection import YOLODetector

detector = YOLODetector(model_path="models/yolo_facade.pt")
detections = detector.detect("data/images/building.jpg")
```

Fusion :

```python
from src.fusion import DataFusion

fusion = DataFusion()
result = fusion.fuse_lidar_images(
    point_cloud="data/processed/building.pcd",
    image="data/images/building.jpg",
    camera_params="config/camera.yaml"
)
```

Pipeline complet :

```python
from src.facade_detection import FacadeDetectionPipeline

pipeline = FacadeDetectionPipeline(config="config/pipeline_config.yaml")
results = pipeline.process(
    lidar_file="data/raw/building.las",
    image_file="data/images/building.jpg"
)
pipeline.visualize(results, output_path="results/detection_result.png")
```

## Fonctionnalités

1. **LiDAR** : lecture `.las`, `.laz`, `.ply`, `.pcd`, prétraitement, extraction des plans verticaux.
2. **YOLO** : détection des façades dans l'image, fine-tuning, export des résultats.
3. **Fusion** : calibration caméra-LiDAR, projection 3D vers 2D, rapprochement des détections.
4. **Visualisation** : nuage 3D, détections dessinées sur l'image, export 2D et 3D.

La configuration est dans `config/`.

Entraînement :

```bash
python train_yolo_facade.py --data config/dataset_facade.yaml --epochs 100 --imgsz 640
```

Tests :

```bash
pytest tests/
```

## Auteur

Nouhaila Berghane. Projet personnel de portfolio, commencé en 2025 et amélioré en 2026.

## Licence

MIT. Voir `LICENSE`.

---

# English

## Project

Portfolio project in computer vision. Elivia detects building facades by fusing a LiDAR point cloud with a 2D image. YOLO finds facades in the image. LiDAR extracts vertical planes. Fusion aligns the two when a camera calibration is available.

First version in 2025. Improvements in 2026: a KITTI-360 dataset, nuScenes calibration, pseudo-labels projected from LiDAR, and a repository small enough for GitHub.

## The name

**Elivia** comes from *élévation*. In architecture, an elevation is the drawing that shows a facade head-on. The project does the same thing from a LiDAR point cloud and an image: it recovers the facades of a building.

## Stack

- **Python**
- **LiDAR** and **Open3D** for point clouds
- **YOLOv8** (Ultralytics) for 2D detection
- **OpenCV** for images
- **PyTorch**, NumPy, SciPy

## Layout

```
.
├── src/          # Pipeline: LiDAR, YOLO, fusion, visualization
├── scripts/      # Dataset preparation and pseudo-labels
├── examples/     # Command-line examples
├── config/       # Pipeline and YOLO dataset YAML
├── utils/        # nuScenes mini, calibration, helpers
├── notebooks/    # Demo
├── tests/        # LiDAR processing tests
├── data/         # Local data, not versioned
├── datasets/     # Generated YOLO sets, not versioned
├── models/       # Trained weights, not versioned
└── results/      # Visualization outputs, not versioned
```

KITTI-360 archives, nuScenes mini (`v1.0-mini.tgz`, `nuscmini/`), and `.pt` weights stay on your machine. GitHub rejects files larger than 100 MB.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For an NVIDIA GPU, install PyTorch with CUDA before the other dependencies: https://pytorch.org/get-started/locally/

## Data

Run the commands from the project root. Paths in `config/dataset_*.yaml` are relative to that root.

**nuScenes mini**, LiDAR and camera pairs with real calibration. Download `v1.0-mini.tgz` from https://www.nuscenes.org/nuscenes then:

```bash
python utils/extract_v1_mini.py --archive v1.0-mini.tgz --out nuscmini
python examples/basic_usage.py --nusc-root nuscmini --nusc-cam CAM_FRONT
```

**KITTI-360**, facades taken from the semantic class *building* (id 11). Place `kitti360/2013_05_28_drive_0000_sync_image_00.zip` and `data_2d_semantics.zip` at the project root, then:

```bash
python scripts/kitti360_to_yolo.py
python train_yolo_facade.py --data config/dataset_kitti360_facade.yaml --epochs 30 --name yolo_facade_kitti
```

## Use

The code samples in the French section above are the ones to run. Same entry points:

- `LiDARProcessor` in `src/lidar_processing.py`
- `YOLODetector` in `src/yolo_detection.py`
- `DataFusion` in `src/fusion.py`
- `FacadeDetectionPipeline` in `src/facade_detection.py`

## What it does

1. **LiDAR**: read `.las`, `.laz`, `.ply`, `.pcd`, preprocess, extract vertical planes.
2. **YOLO**: detect facades in the image, fine-tune, export results.
3. **Fusion**: camera-LiDAR calibration, 3D-to-2D projection, match the two detections.
4. **Visualization**: 3D cloud, detections drawn on the image, 2D and 3D export.

Configuration lives in `config/`.

Training:

```bash
python train_yolo_facade.py --data config/dataset_facade.yaml --epochs 100 --imgsz 640
```

Tests:

```bash
pytest tests/
```

## Author

Nouhaila Berghane. Personal portfolio project, started in 2025 and improved in 2026.

## License

MIT. See `LICENSE`.
