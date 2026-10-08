"""
Script pour préparer un dataset YOLO pour l'entraînement de détection de façades
- Crée la structure de dossiers
- Vérifie la cohérence images/labels
- Génère le fichier dataset.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_dataset_structure(base_path: Path):
    """Crée la structure de dossiers pour un dataset YOLO"""
    dirs = [
        base_path / "images" / "train",
        base_path / "images" / "val",
        base_path / "images" / "test",
        base_path / "labels" / "train",
        base_path / "labels" / "val",
        base_path / "labels" / "test",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")
    
    print(f"\n✅ Structure créée dans: {base_path}")
    return base_path


def validate_dataset(dataset_path: Path):
    """Valide qu'un dataset YOLO est correctement structuré"""
    errors = []
    warnings = []
    
    # Vérifier les dossiers
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val",
    ]
    
    for rel_dir in required_dirs:
        full_dir = dataset_path / rel_dir
        if not full_dir.exists():
            errors.append(f"Dossier manquant: {rel_dir}")
    
    if errors:
        print("❌ Erreurs trouvées:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    # Vérifier les images et labels
    for split in ["train", "val"]:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        
        images = set(f.stem for f in img_dir.glob("*.jpg")) | set(f.stem for f in img_dir.glob("*.png"))
        labels = set(f.stem for f in label_dir.glob("*.txt"))
        
        missing_labels = images - labels
        missing_images = labels - images
        
        if missing_labels:
            warnings.append(f"{split}: {len(missing_labels)} images sans labels")
        if missing_images:
            warnings.append(f"{split}: {len(missing_images)} labels sans images")
    
    if warnings:
        print("⚠️  Avertissements:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("✅ Dataset valide!")
    
    # Afficher les statistiques
    print("\n📊 Statistiques:")
    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        if img_dir.exists():
            n_img = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
            n_lbl = len(list(label_dir.glob("*.txt")))
            if n_img > 0:
                print(f"  {split}: {n_img} images, {n_lbl} labels")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Préparer/valider un dataset YOLO pour détection de façades"
    )
    parser.add_argument(
        "--create",
        type=str,
        help="Créer la structure de dossiers dans le chemin spécifié",
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Valider un dataset existant",
    )
    
    args = parser.parse_args()
    
    if args.create:
        base = Path(args.create)
        create_dataset_structure(base)
        print("\n📝 Prochaines étapes:")
        print("  1. Placez vos images dans images/train/ et images/val/")
        print("  2. Annotez avec un outil (LabelImg, Roboflow, etc.)")
        print("  3. Placez les labels (.txt) dans labels/train/ et labels/val/")
        print("  4. Exécutez --validate pour vérifier")
    
    elif args.validate:
        dataset_path = Path(args.validate)
        if not dataset_path.exists():
            print(f"❌ Chemin inexistant: {dataset_path}")
            return 1
        
        validate_dataset(dataset_path)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

