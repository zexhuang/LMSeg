import os
from typing import Tuple

import CSF
import numpy as np

   
def csf(points: np.ndarray, 
        cloth_resolution: float = 0.5, 
        rigidness: int = 1,
        class_threshold: float = 0.5, 
        iterations: int = 500, 
        slope_smooth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.
    Source: https://github.com/Yarroudh/segment-lidar/blob/4a565e9df6221f5771ed7140a9ef47a363aa3400/segment_lidar/samlidar.py

    :param points: The input point cloud as a NumPy array, where each row represents a point with x, y, z coordinates.
    :type points: np.ndarray
    :param class_threshold: The threshold value for classifying points as ground/non-ground, defaults to 0.5.
    :type class_threshold: float, optional
    :param cloth_resolution: The resolution value for cloth simulation, defaults to 0.5.
    :type cloth_resolution: float, optional
    :param iterations: The number of iterations for the CSF algorithm, defaults to 500.
    :type iterations: int, optional
    :param slope_smooth: A boolean indicating whether to enable slope smoothing, defaults to True.
    :type slope_smooth: bool, optional
    :return: A tuple containing two arrays: non-ground (filtered) points indinces and ground points indices.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    csf = CSF.CSF()
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness 
    csf.params.interations = iterations
    csf.params.class_threshold = class_threshold
    csf.params.bSloopSmooth = slope_smooth
    
    csf.setPointCloud(points[:, :3])
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    os.remove('cloth_nodes.txt')

    return np.asarray(non_ground, dtype=np.int32),\
           np.asarray(ground, dtype=np.int32)