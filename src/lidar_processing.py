"""
Module de traitement des nuages de points LiDAR
Gère le chargement, le préprocessing et l'extraction de façades depuis les données LiDAR
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# `laspy` est requis uniquement pour lire les fichiers .las/.laz.
# On le rend optionnel pour permettre l'utilisation avec .ply/.pcd sans dépendance LAS.
try:
    import laspy  # type: ignore
except ImportError:  # pragma: no cover
    laspy = None


class LiDARProcessor:
    """Classe pour le traitement des nuages de points LiDAR"""
    
    def __init__(self, voxel_size: float = 0.05, normal_radius: float = 0.1):
        """
        Initialise le processeur LiDAR
        
        Args:
            voxel_size: Taille des voxels pour le downsampling (mètres)
            normal_radius: Rayon pour le calcul des normales (mètres)
        """
        self.voxel_size = voxel_size
        self.normal_radius = normal_radius
        
    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        Charge un nuage de points depuis un fichier
        
        Args:
            file_path: Chemin vers le fichier (.las, .laz, .ply, .pcd, .pcd.bin)
            
        Returns:
            Nuage de points Open3D
        """
        logger.info(f"Chargement du nuage de points: {file_path}")
        
        path_lower = file_path.lower()
        # NuScenes LiDAR: *.pcd.bin (binaire float32)
        if path_lower.endswith(".pcd.bin"):
            return self._load_nuscenes_pcd_bin(file_path)

        file_extension = path_lower.split('.')[-1]
        
        if file_extension in ['las', 'laz']:
            return self._load_las_file(file_path)
        elif file_extension in ['ply', 'pcd']:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                raise ValueError(f"Impossible de charger le fichier: {file_path}")
            return pcd
        else:
            raise ValueError(f"Format de fichier non supporté: {file_extension}")

    def _load_nuscenes_pcd_bin(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        Charge un fichier NuScenes LiDAR *.pcd.bin.

        Format NuScenes (LIDAR_TOP): float32, shape (N, 5) avec (x, y, z, intensity, ring_index).
        On conserve (x, y, z). L'intensité peut être utilisée comme couleur optionnelle.
        """
        try:
            raw = np.fromfile(file_path, dtype=np.float32)
            if raw.size % 5 != 0:
                raise ValueError(f"Fichier inattendu (taille non multiple de 5 float32): {file_path}")

            pts = raw.reshape((-1, 5))
            xyz = pts[:, :3]
            intensity = pts[:, 3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Couleur optionnelle: intensité normalisée (grayscale)
            if intensity.size:
                imin, imax = float(intensity.min()), float(intensity.max())
                if imax > imin:
                    g = (intensity - imin) / (imax - imin)
                else:
                    g = np.zeros_like(intensity)
                colors = np.stack([g, g, g], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            logger.info(f"Nuage NuScenes chargé: {len(pcd.points)} points")
            return pcd
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier NuScenes .pcd.bin: {e}")
            raise
    
    def _load_las_file(self, file_path: str) -> o3d.geometry.PointCloud:
        """Charge un fichier LAS/LAZ"""
        if laspy is None:
            raise ModuleNotFoundError(
                "Le module 'laspy' n'est pas installé. "
                "Installe-le pour lire des fichiers .las/.laz: `pip install laspy` "
                "(ou `pip install -r requirements.txt`)."
            )
        try:
            las = laspy.read(file_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Ajouter les couleurs si disponibles
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((
                    las.red / 65535.0,
                    las.green / 65535.0,
                    las.blue / 65535.0
                )).transpose()
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            logger.info(f"Nuage de points chargé: {len(pcd.points)} points")
            return pcd
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier LAS: {e}")
            raise
    
    def preprocess(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Prétraite le nuage de points: filtrage du bruit, downsampling, calcul des normales
        
        Args:
            point_cloud: Nuage de points à traiter
            
        Returns:
            Nuage de points préprocessé
        """
        logger.info("Préprocessing du nuage de points...")
        processed = point_cloud
        
        # 1. Suppression des points aberrants (outliers)
        processed, _ = processed.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        logger.info(f"Points après suppression des outliers: {len(processed.points)}")
        
        # 2. Downsampling avec voxel grid
        if self.voxel_size > 0:
            processed = processed.voxel_down_sample(voxel_size=self.voxel_size)
            logger.info(f"Points après downsampling: {len(processed.points)}")
        
        # 3. Calcul des normales
        processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30
            )
        )
        processed.orient_normals_consistent_tangent_plane(100)
        
        logger.info("Préprocessing terminé")
        return processed
    
    def detect_facades(
        self, 
        point_cloud: o3d.geometry.PointCloud,
        min_points: int = 80,
        angle_threshold: float = 0.85,
        max_facades: int = 12,
    ) -> List[o3d.geometry.PointCloud]:
        """
        Détecte les façades comme des plans verticaux distincts.

        Un mur est un plan (RANSAC) dont la normale est quasi horizontale, puis
        découpé spatialement (DBSCAN) pour ne pas fusionner deux bâtiments
        coplanaires. Le sol, le véhicule porteur et les petits plans (voitures)
        sont écartés.

        Args:
            point_cloud: Nuage de points préprocessé
            min_points: Nombre minimum de points pour considérer une façade
            angle_threshold: Plus la valeur est haute, plus le plan doit être vertical.
                |n_z| max = 1 - angle_threshold, borné à [0.2, 0.45].
            max_facades: Nombre maximum de façades renvoyées
            
        Returns:
            Liste de nuages de points représentant les façades détectées
        """
        logger.info("Détection des façades...")

        if len(point_cloud.points) == 0:
            return []

        cloud = o3d.geometry.PointCloud(point_cloud)
        points = np.asarray(cloud.points)
        # Repère nuScenes: x avant, y gauche, z haut. On retire le véhicule
        # porteur et les points trop loin pour former un mur fiable.
        dist_xy = np.linalg.norm(points[:, :2], axis=1)
        keep = (dist_xy > 3.0) & (dist_xy < 55.0) & (points[:, 2] > -3.0) & (points[:, 2] < 15.0)
        cloud = cloud.select_by_index(np.flatnonzero(keep))
        if len(cloud.points) < min_points:
            logger.info("0 façades détectées")
            return []

        # Les normales par point sont peu fiables sur un LiDAR en anneaux
        # (le sol ressemble localement à des lignes). On classe les plans
        # avec la normale RANSAC, pas avec les normales des points.
        vertical_nz = float(np.clip(1.0 - angle_threshold, 0.2, 0.45))
        plane_distance = float(np.clip(max(self.voxel_size * 1.5, 0.15), 0.15, 0.22))
        remaining = self._remove_horizontal_planes(cloud, plane_distance, min_points)
        logger.info(f"Points après retrait du sol: {len(remaining.points)}")

        facades: List[o3d.geometry.PointCloud] = []
        while len(remaining.points) >= min_points and len(facades) < max_facades:
            try:
                plane_model, inliers = remaining.segment_plane(
                    distance_threshold=plane_distance,
                    ransac_n=3,
                    num_iterations=1000,
                )
            except RuntimeError:
                break

            if len(inliers) < min_points:
                break

            inlier_cloud = remaining.select_by_index(inliers)
            remaining = remaining.select_by_index(inliers, invert=True)

            normal = self._unit_normal(plane_model[:3])
            if normal is None or abs(float(normal[2])) > vertical_nz:
                continue

            clusters = self._split_plane_clusters(inlier_cloud, min_points)
            for cluster in clusters:
                if self._is_facade_extent(cluster, normal):
                    facades.append(cluster)
                    if len(facades) >= max_facades:
                        break

        logger.info(f"{len(facades)} façades détectées")
        return facades

    def _remove_horizontal_planes(
        self,
        cloud: o3d.geometry.PointCloud,
        plane_distance: float,
        min_points: int,
    ) -> o3d.geometry.PointCloud:
        """Retire les grands plans horizontaux (sol, toits bas) du nuage."""
        remaining = cloud
        for _ in range(6):
            if len(remaining.points) < min_points:
                break
            try:
                plane_model, inliers = remaining.segment_plane(
                    distance_threshold=plane_distance,
                    ransac_n=3,
                    num_iterations=800,
                )
            except RuntimeError:
                break
            if len(inliers) < max(min_points, int(0.04 * len(cloud.points))):
                break
            normal = self._unit_normal(plane_model[:3])
            if normal is None or abs(float(normal[2])) < 0.85:
                break
            remaining = remaining.select_by_index(inliers, invert=True)
        return remaining

    @staticmethod
    def _unit_normal(normal) -> Optional[np.ndarray]:
        vector = np.asarray(normal, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm

    def _split_plane_clusters(
        self,
        plane_cloud: o3d.geometry.PointCloud,
        min_points: int,
    ) -> List[o3d.geometry.PointCloud]:
        """Découpe un plan RANSAC en murs spatialement séparés."""
        if len(plane_cloud.points) < min_points:
            return []

        labels = np.asarray(
            plane_cloud.cluster_dbscan(eps=2.5, min_points=max(10, min_points // 4))
        )
        clusters: List[o3d.geometry.PointCloud] = []
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            indices = np.flatnonzero(labels == label)
            if len(indices) >= min_points:
                clusters.append(plane_cloud.select_by_index(indices.tolist()))

        if not clusters and len(plane_cloud.points) >= min_points:
            clusters.append(plane_cloud)
        return clusters

    def _is_facade_extent(
        self,
        facade: o3d.geometry.PointCloud,
        normal: np.ndarray,
    ) -> bool:
        """Garde les plans de taille bâtiment et écarte voitures et plans énormes."""
        pts = np.asarray(facade.points)
        if len(pts) == 0:
            return False

        up = np.array([0.0, 0.0, 1.0])
        width_axis = np.cross(up, normal)
        width_norm = np.linalg.norm(width_axis)
        if width_norm < 1e-6:
            return False
        width_axis = width_axis / width_norm

        width = float(np.ptp(pts @ width_axis))
        height = float(np.ptp(pts[:, 2]))
        distance = float(np.linalg.norm(pts.mean(axis=0)[:2]))

        if distance < 4.0 or distance > 50.0:
            return False
        if width < 2.5 or width > 35.0:
            return False
        if height < 1.8 or height > 25.0:
            return False
        # Côté de véhicule: bas et court. Une façade est plus haute ou bien plus longue.
        if height < 2.3 and width < 7.0:
            return False
        return True
    
    def segment_vertical_surfaces(
        self,
        point_cloud: o3d.geometry.PointCloud,
        vertical_threshold: float = 0.7
    ) -> o3d.geometry.PointCloud:
        """
        Ségmente les surfaces verticales (façades) du nuage de points
        
        Args:
            point_cloud: Nuage de points
            vertical_threshold: Seuil pour considérer une surface comme verticale
            
        Returns:
            Nuage de points contenant uniquement les surfaces verticales
        """
        if len(point_cloud.normals) == 0:
            point_cloud.estimate_normals()
        
        normals = np.asarray(point_cloud.normals)
        # Les surfaces verticales ont des normales horizontales (z proche de 0)
        vertical_mask = np.abs(normals[:, 2]) < vertical_threshold
        
        vertical_pcd = o3d.geometry.PointCloud()
        vertical_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(point_cloud.points)[vertical_mask]
        )
        if len(point_cloud.colors) > 0:
            vertical_pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(point_cloud.colors)[vertical_mask]
            )
        
        return vertical_pcd
    
    def get_bounding_box(
        self, 
        point_cloud: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la bounding box du nuage de points
        
        Returns:
            Tuple (min_bound, max_bound)
        """
        bbox = point_cloud.get_axis_aligned_bounding_box()
        return np.array(bbox.min_bound), np.array(bbox.max_bound)
    
    def save_point_cloud(
        self, 
        point_cloud: o3d.geometry.PointCloud, 
        file_path: str
    ) -> None:
        """Sauvegarde un nuage de points"""
        o3d.io.write_point_cloud(file_path, point_cloud)
        logger.info(f"Nuage de points sauvegardé: {file_path}")


