# Guide de Démarrage Rapide

Ce guide vous aidera à démarrer rapidement avec le système de détection de façades.

## Installation

1. **Installer Python** (version 3.10 ou supérieure)

Les jeux de données, archives et poids YOLO ne sont pas dans le dépôt Git. Voir la section Données du `README.md`.

2. **Créer un environnement virtuel** :
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
```

3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

**Note** : Si vous utilisez CUDA pour l'accélération GPU, assurez-vous d'installer PyTorch avec support CUDA :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Préparation des données

1. **Placez vos fichiers LiDAR** dans `data/raw/` :
   - Formats supportés : `.las`, `.laz`, `.ply`, `.pcd`

2. **Placez vos images** dans `data/images/` :
   - Formats supportés : `.jpg`, `.png`

3. **Configurez la calibration** (optionnel) :
   - Modifiez `config/camera_calibration.yaml` avec vos paramètres de calibration caméra-LiDAR

## Utilisation

### Option 1 : Pipeline complet (recommandé)

```python
from src.facade_detection import FacadeDetectionPipeline

pipeline = FacadeDetectionPipeline(config="config/pipeline_config.yaml")
results = pipeline.process(
    lidar_file="data/raw/building.las",
    image_file="data/images/building.jpg",
    output_dir="results"
)
```

### Option 2 : Utilisation séparée des composants

#### Traitement LiDAR seul
```python
from src.lidar_processing import LiDARProcessor

processor = LiDARProcessor()
point_cloud = processor.load_point_cloud("data/raw/building.las")
processed = processor.preprocess(point_cloud)
facades = processor.detect_facades(processed)
```

#### Détection YOLO seule
```python
from src.yolo_detection import YOLODetector

detector = YOLODetector()
detections = detector.detect("data/images/building.jpg")
```

### Option 3 : Exemples fournis

Exécutez les exemples fournis dans le dossier `examples/` :

```bash
python examples/basic_usage.py
python examples/lidar_only.py
python examples/yolo_only.py
```

## Configuration

Modifiez `config/pipeline_config.yaml` pour ajuster :
- Paramètres de traitement LiDAR (voxel_size, normal_radius, etc.)
- Seuils de confiance YOLO
- Paramètres de fusion

## Résultats

Les résultats seront sauvegardés dans le dossier `results/` :
- Images annotées avec les détections
- Visualisations 3D des nuages de points
- Métriques de validation

## Dépannage

### Erreur "CUDA out of memory"
- Réduisez la taille du batch ou utilisez CPU (`device="cpu"` dans la config)
- Réduisez la taille des images (`image_size` dans la config)

### Aucune détection trouvée
- Vérifiez les seuils de confiance dans la configuration
- Vérifiez que vos données sont bien formatées
- Consultez les logs pour plus d'informations

### Erreur de calibration
- Vérifiez que `config/camera_calibration.yaml` contient les bons paramètres
- Si vous n'avez pas de calibration, désactivez la fusion dans la config

## Prochaines étapes

- Consultez le `README.md` pour plus de détails
- Explorez le notebook `notebooks/demo_detection_facades.ipynb`
- Personnalisez les modèles YOLO pour vos données spécifiques


