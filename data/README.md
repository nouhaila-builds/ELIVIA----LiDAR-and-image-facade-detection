# Données

Ce dossier contient les données utilisées pour le projet.

## Structure

```
data/
├── raw/              # Données brutes (non versionnées)
│   ├── *.las        # Fichiers LiDAR
│   ├── *.laz
│   ├── *.ply
│   └── *.pcd
├── processed/        # Données traitées (non versionnées)
│   └── *.pcd
└── images/           # Images 2D (non versionnées)
    ├── *.jpg
    └── *.png
```

## Formats supportés

### LiDAR
- `.las` / `.laz` : Formats LAS standard
- `.ply` : Format Polygon File Format
- `.pcd` : Point Cloud Data (Open3D)

### Images
- `.jpg` / `.jpeg` : JPEG
- `.png` : PNG

## Instructions

1. Placez vos fichiers LiDAR dans `raw/`
2. Placez vos images dans `images/`
3. Les fichiers traités seront sauvegardés dans `processed/` (si nécessaire)

**Note** : Les fichiers volumineux ne sont pas versionnés (voir `.gitignore`)


