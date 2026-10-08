"""
Exemple d'utilisation basique du système de détection de façades
"""

import sys
from pathlib import Path
import argparse

# Ajouter le répertoire src au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.facade_detection import FacadeDetectionPipeline
from utils.nuscenes_mini import find_pairs


def main():
    """Exemple d'utilisation du pipeline complet"""
    
    print("=" * 60)
    print("Système de Détection 3D des Façades")
    print("Fusion LiDAR + Images 2D avec YOLO")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Pipeline complet de détection de façades (LiDAR + image 2D + YOLO)."
    )
    parser.add_argument(
        "--lidar",
        default="data/raw/building.las",
        help="Chemin vers le fichier LiDAR (.las/.laz/.ply/.pcd/.pcd.bin).",
    )
    parser.add_argument(
        "--image",
        default="data/images/building.jpg",
        help="Chemin vers l'image 2D (.jpg/.png).",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Chemin vers le fichier YAML de configuration.",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Dossier de sortie pour les résultats.",
    )
    parser.add_argument(
        "--nusc-root",
        default=None,
        help=r"Optionnel: racine nuScenes mini extraite (contient samples/, sweeps/, v1.0-mini/). "
             r"Ex: C:\nuscmini. Si fourni, ignore --lidar/--image et prend une paire auto.",
    )
    parser.add_argument("--nusc-cam", default="CAM_FRONT", help="Canal caméra nuScenes (ex: CAM_FRONT).")
    args = parser.parse_args()

    # Initialiser le pipeline avec la configuration
    pipeline = FacadeDetectionPipeline(config=args.config)

    lidar_file = args.lidar
    image_file = args.image
    output_dir = args.output

    # Mode nuScenes: trouver automatiquement une paire (LIDAR_TOP + CAM_*)
    if args.nusc_root:
        pairs = find_pairs(Path(args.nusc_root), cam_channel=args.nusc_cam, limit=1)
        if not pairs:
            print("❌ Aucune paire (LiDAR+caméra) trouvée dans nuScenes mini.")
            print("   Vérifie --nusc-root (doit contenir samples/, sweeps/, v1.0-mini/)")
            return
        pair = pairs[0]
        lidar_file = str(pair.lidar_path)
        image_file = str(pair.cam_path)
        print(f"✅ nuScenes mini: paire auto (cam={pair.cam_channel})")
        print(f"   lidar: {lidar_file}")
        print(f"   image: {image_file}")
    
    # Vérifier que les fichiers existent (optionnel pour l'exemple)
    if not Path(lidar_file).exists() or not Path(image_file).exists():
        print("\n⚠️  ATTENTION: Les fichiers de données n'existent pas encore.")
        print("   Veuillez placer vos fichiers LiDAR et images dans:")
        print(f"   - LiDAR: {lidar_file}")
        print(f"   - Image: {image_file}")
        print("\n   Structure attendue:")
        print("   data/")
        print("   ├── raw/")
        print("   │   └── building.las  (ou .laz, .ply, .pcd)")
        print("   └── images/")
        print("       └── building.jpg  (ou .png)")
        print("\n   OU lancez avec vos chemins:")
        print("   python examples/basic_usage.py --lidar \"C:\\chemin\\fichier.las\" --image \"C:\\chemin\\image.jpg\"")
        print("\n   OU si vos données sont dans nuScenes mini (v1.0-mini.tgz):")
        print("   python utils/extract_v1_mini.py --archive v1.0-mini.tgz --out C:\\nuscmini")
        print("   python examples/basic_usage.py --nusc-root C:\\nuscmini --nusc-cam CAM_FRONT")
        return
    
    # Traiter les données
    print(f"\n📁 Traitement des fichiers:")
    print(f"   LiDAR: {lidar_file}")
    print(f"   Image: {image_file}")
    print()
    
    try:
        results = pipeline.process(
            lidar_file=lidar_file,
            image_file=image_file,
            output_dir=output_dir
        )
        
        # Afficher les résultats
        print("\n" + "=" * 60)
        print("RÉSULTATS")
        print("=" * 60)
        print(f"✓ Façades détectées par LiDAR: {len(results['lidar_facades'])}")
        print(f"✓ Détections YOLO: {len(results['yolo_detections'])}")
        
        if 'validation' in results and 'metrics' in results['validation']:
            metrics = results['validation']['metrics']
            print(f"✓ Taux de validation: {metrics.get('validation_rate', 0):.1%}")
        
        # Visualiser les résultats
        print("\n📊 Génération des visualisations...")
        pipeline.visualize(
            results,
            output_path=f"{output_dir}/3d_visualization.png",
            show_3d=True,
            show_2d=True
        )
        
        print(f"\n✅ Traitement terminé! Résultats dans: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


