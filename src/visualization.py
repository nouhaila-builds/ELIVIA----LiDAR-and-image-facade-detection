"""
Module de visualisation des résultats de détection
Visualisation 3D des nuages de points et superposition des détections
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """Classe pour la visualisation des résultats de détection"""
    
    def __init__(self):
        """Initialise le visualiseur"""
        pass
    
    def visualize_point_cloud_3d(
        self,
        point_cloud: o3d.geometry.PointCloud,
        facades: Optional[List[o3d.geometry.PointCloud]] = None,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualise le nuage de points 3D avec les façades détectées
        
        Args:
            point_cloud: Nuage de points principal
            facades: Liste de nuages de points représentant les façades
            output_path: Chemin pour sauvegarder la visualisation
            show: Afficher la visualisation interactive
        """
        logger.info("Création de la visualisation 3D...")
        
        # Créer la visualisation Open3D
        geometries = [point_cloud]
        
        # Ajouter les façades avec des couleurs différentes
        if facades:
            colors = self._generate_colors(len(facades))
            for i, facade in enumerate(facades):
                facade.paint_uniform_color(colors[i])
                geometries.append(facade)
        
        # Visualiser
        if show:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Détection de façades 3D",
                width=1024,
                height=768
            )
        
        # Sauvegarder si nécessaire
        if output_path:
            # Pour sauvegarder, on peut créer une image avec matplotlib
            self._save_3d_visualization(point_cloud, facades, output_path)
    
    def _generate_colors(self, n: int) -> List[List[float]]:
        """
        Génère n couleurs distinctes
        
        Args:
            n: Nombre de couleurs à générer
            
        Returns:
            Liste de couleurs RGB [0, 1]
        """
        import matplotlib.cm as cm
        
        colormap = cm.get_cmap('tab20')
        colors = [colormap(i / max(n, 1))[:3] for i in range(n)]
        return colors
    
    def _save_3d_visualization(
        self,
        point_cloud: o3d.geometry.PointCloud,
        facades: Optional[List[o3d.geometry.PointCloud]],
        output_path: str
    ) -> None:
        """Sauvegarde une visualisation 3D comme image"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Afficher le nuage de points principal
        points = np.asarray(point_cloud.points)
        # Sous-échantillonnage pour la visualisation si trop de points
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
        
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c='gray',
            s=1,
            alpha=0.3,
            label='Nuage de points'
        )
        
        # Afficher les façades avec des couleurs différentes
        if facades:
            colors = self._generate_colors(len(facades))
            for i, facade in enumerate(facades):
                facade_points = np.asarray(facade.points)
                if len(facade_points) > 5000:
                    indices = np.random.choice(len(facade_points), 5000, replace=False)
                    facade_points = facade_points[indices]
                
                ax.scatter(
                    facade_points[:, 0],
                    facade_points[:, 1],
                    facade_points[:, 2],
                    c=[colors[i]],
                    s=2,
                    alpha=0.8,
                    label=f'Façade {i+1}'
                )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Détection de façades 3D')
        ax.legend()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualisation 3D sauvegardée: {output_path}")
    
    def visualize_detections_2d(
        self,
        image: np.ndarray,
        detections: List[dict],
        lidar_points_2d: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualise les détections 2D sur une image
        
        Args:
            image: Image originale
            detections: Liste de détections YOLO
            lidar_points_2d: Points LiDAR projetés en 2D (optionnel)
            output_path: Chemin pour sauvegarder
            
        Returns:
            Image annotée
        """
        import cv2
        
        # Créer une copie de l'image
        annotated = image.copy()
        
        # Dessiner les détections YOLO
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Couleur selon la validation si disponible
            if det.get('validated', False):
                color = (0, 255, 0)  # Vert = validé
            else:
                color = (0, 165, 255)  # Orange = non validé
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.get('class_name', 'facade')}: {det['confidence']:.2f}"
            if 'distance_to_lidar' in det:
                label += f" (d={det['distance_to_lidar']:.1f}px)"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Superposer les points LiDAR si disponibles
        if lidar_points_2d is not None:
            for point in lidar_points_2d:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < annotated.shape[1] and 0 <= y < annotated.shape[0]:
                    cv2.circle(annotated, (x, y), 1, (255, 0, 0), -1)  # Rouge pour LiDAR
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            logger.info(f"Visualisation 2D sauvegardée: {output_path}")
        
        return annotated
    
    def plot_validation_metrics(
        self,
        validation_results: dict,
        output_path: Optional[str] = None
    ) -> None:
        """
        Trace les métriques de validation
        
        Args:
            validation_results: Résultats de validation depuis DataFusion.validate_detections()
            output_path: Chemin pour sauvegarder le graphique
        """
        metrics = validation_results.get('metrics', {})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Total', 'Validées', 'Non validées']
        values = [
            metrics.get('total_detections', 0),
            metrics.get('validated_detections', 0),
            metrics.get('total_detections', 0) - metrics.get('validated_detections', 0)
        ]
        colors = ['gray', 'green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Nombre de détections')
        ax.set_title('Résultats de la validation croisée LiDAR/YOLO')
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        # Ajouter le taux de validation
        validation_rate = metrics.get('validation_rate', 0)
        ax.text(
            0.5, 0.95,
            f"Taux de validation: {validation_rate:.1%}",
            transform=ax.transAxes,
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Métriques sauvegardées: {output_path}")
        else:
            plt.show()
        
        plt.close()


