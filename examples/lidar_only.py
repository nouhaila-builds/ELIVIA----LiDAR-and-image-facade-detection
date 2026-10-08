"""
Exemple d'utilisation du traitement LiDAR seul
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lidar_processing import LiDARProcessor


def main():
    """Exemple d'utilisation du processeur LiDAR"""
    
    print("Exemple: Traitement LiDAR seul")
    print("=" * 50)
    
    # Initialiser le processeur
    processor = LiDARProcessor(
        voxel_size=0.05,  # 5 cm
        normal_radius=0.1  # 10 cm
    )
    
    parser = argparse.ArgumentParser(description="Traitement LiDAR seul (préprocessing + détection de façades).")
    parser.add_argument(
        "--lidar",
        default="data/raw/building.las",
        help="Chemin vers le fichier LiDAR (.las/.laz/.ply/.pcd).",
    )
    parser.add_argument(
        "--output",
        default="results/lidar",
        help="Dossier de sortie.",
    )
    args = parser.parse_args()

    # Charger le nuage de points
    lidar_file = args.lidar
    
    if not Path(lidar_file).exists():
        print(f"⚠️  Fichier non trouvé: {lidar_file}")
        print("   Placez votre fichier LiDAR dans data/raw/")
        return
    
    print(f"📁 Chargement: {lidar_file}")
    point_cloud = processor.load_point_cloud(lidar_file)
    print(f"   Points chargés: {len(point_cloud.points)}")
    
    # Préprocessing
    print("\n🔧 Préprocessing...")
    processed_cloud = processor.preprocess(point_cloud)
    print(f"   Points après traitement: {len(processed_cloud.points)}")
    
    # Détection des façades
    print("\n🏢 Détection des façades...")
    facades = processor.detect_facades(
        processed_cloud,
        min_points=1000,
        angle_threshold=0.85
    )
    print(f"   Façades détectées: {len(facades)}")
    
    # Afficher les informations sur chaque façade
    for i, facade in enumerate(facades):
        bbox_min, bbox_max = processor.get_bounding_box(facade)
        print(f"\n   Façade {i+1}:")
        print(f"      Points: {len(facade.points)}")
        print(f"      Bounding box min: {bbox_min}")
        print(f"      Bounding box max: {bbox_max}")
    
    # Sauvegarder les résultats
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor.save_point_cloud(
        processed_cloud,
        str(output_dir / "processed_cloud.pcd")
    )
    
    print(f"\n✅ Résultats sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()


