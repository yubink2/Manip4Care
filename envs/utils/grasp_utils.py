import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from utils.point_cloud_utils import get_point_cloud_from_collision_shapes_specific_link


def rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90):
    rotation = R.from_euler(axis, degrees, degrees=True).as_quat()
    new_quaternion = R.from_quat(quaternion) * R.from_quat(rotation)
    
    return new_quaternion.as_quat()

def filter_transform_matrices_by_position(matrices, x_range, y_range, z_range):
    """
    Filters transformation matrices based on the user-defined range of positions and returns the indices.

    x_range (tuple): The range (min, max) for the x coordinate.
    y_range (tuple): The range (min, max) for the y coordinate.
    z_range (tuple): The range (min, max) for the z coordinate.
    """
    filtered_matrices = []
    indices = []
    for idx, matrix in enumerate(matrices):
        position = matrix[:3, 3]  # Extract the position (x, y, z)
        if (x_range[0] <= position[0] <= x_range[1] and
            y_range[0] <= position[1] <= y_range[1] and
            z_range[0] <= position[2] <= z_range[1]):
            filtered_matrices.append(matrix)
            indices.append(idx)
    return np.array(filtered_matrices), indices

def get_human_arm_pcd_for_grasp_sampler(env, scale=None, scale_radius=None, scale_height=None, client_id=0):
    right_elbow_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_elbow, resolution=40, 
                                                                          scale_radius=scale_radius, scale_height=scale_height,
                                                                          skip_hemispherical=True, client_id=client_id)

    return right_elbow_pcd
