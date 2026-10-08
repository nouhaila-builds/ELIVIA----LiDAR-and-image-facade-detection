"""
Tests unitaires pour le module lidar_processing
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lidar_processing import LiDARProcessor


class TestLiDARProcessor:
    """Tests pour la classe LiDARProcessor"""
    
    def test_initialization(self):
        """Test de l'initialisation"""
        processor = LiDARProcessor(voxel_size=0.05, normal_radius=0.1)
        assert processor.voxel_size == 0.05
        assert processor.normal_radius == 0.1
    
    def test_create_sample_point_cloud(self):
        """Crée un nuage de points de test"""
        # Créer un nuage de points simple
        points = np.random.rand(1000, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def test_preprocess(self):
        """Test du préprocessing"""
        processor = LiDARProcessor(voxel_size=0.1, normal_radius=0.2)
        pcd = self.test_create_sample_point_cloud()
        
        processed = processor.preprocess(pcd)
        
        assert len(processed.points) > 0
        assert len(processed.normals) > 0
    
    def test_get_bounding_box(self):
        """Test du calcul de bounding box"""
        processor = LiDARProcessor()
        pcd = self.test_create_sample_point_cloud()
        
        min_bound, max_bound = processor.get_bounding_box(pcd)
        
        assert len(min_bound) == 3
        assert len(max_bound) == 3
        assert np.all(min_bound <= max_bound)

    def test_detect_facades_keeps_vertical_wall(self):
        """Un mur vertical est gardé, le sol ne devient pas une façade."""
        ys = np.linspace(-4, 4, 50)
        zs = np.linspace(0, 6, 40)
        yy, zz = np.meshgrid(ys, zs)
        wall = np.stack([np.full(yy.size, 8.0), yy.ravel(), zz.ravel()], axis=1)

        xs = np.linspace(-10, 10, 30)
        ys_ground = np.linspace(-10, 10, 30)
        xx, yy_g = np.meshgrid(xs, ys_ground)
        ground = np.stack([xx.ravel(), yy_g.ravel(), np.zeros(xx.size)], axis=1)

        rng = np.random.default_rng(0)
        points = np.vstack([wall, ground])
        points += rng.normal(0, 0.01, points.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        processor = LiDARProcessor(voxel_size=0.05, normal_radius=0.8)
        facades = processor.detect_facades(pcd, min_points=30, angle_threshold=0.85)

        assert len(facades) >= 1
        facade_x = np.concatenate([np.asarray(f.points)[:, 0] for f in facades])
        assert np.median(np.abs(facade_x - 8.0)) < 0.5


