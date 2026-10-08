# Datasets YOLO

Ce dossier n'est pas versionné. Il est produit en local.

| Dossier | Origine | Config |
| --- | --- | --- |
| `kitti360_facade/` | `python scripts/kitti360_to_yolo.py` | `config/dataset_kitti360_facade.yaml` |
| `facades_pseudo/` | `python scripts/generate_pseudo_labels_from_lidar.py` | `config/dataset_facade_pseudo.yaml` |
| `facades/` | annotations manuelles, structure créée par `python scripts/prepare_dataset.py` | `config/dataset_facade.yaml` |

Lancer l'entraînement depuis la racine du projet pour que les chemins relatifs des YAML restent valides.
