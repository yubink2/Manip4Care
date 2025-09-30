# MIT License

# Copyright (c) 2019 Healthcare Robotics Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified by: Yubin Koh (koh22@purdue.edu)

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

class Util:
    def __init__(self, pid):
        self.id = pid

    def rotate_quaternion_by_axis(self, quaternion, axis='y', degrees=-90):
        rotation = R.from_euler(axis, degrees, degrees=True).as_quat()
        new_quaternion = R.from_quat(quaternion) * R.from_quat(rotation)
        
        return new_quaternion.as_quat()

    def points_in_cylinder(self, pt1, pt2, r, q):
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const

    def capsule_points(self, p1, p2, radius, distance_between_points=0.05):
        points = []

        p1, p2 = np.array(p1), np.array(p2)
        axis_vector = p2 - p1
        # Normalize axis vector to unit length
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        ortho_vector = self.orthogonal_vector(axis_vector)
        # Normalize orthogonal vector to unit length
        ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)
        # Determine normal vector through cross product (this will be of unit length)
        normal_vector = np.cross(axis_vector, ortho_vector)

        # Determine the section positions along the frustum at which we will create point around in a circular fashion
        sections = int(np.linalg.norm(p2 - p1) / distance_between_points)
        section_positions = [(p2 - p1) / (sections + 1) * (i + 1) for i in range(sections)]
        for i, section_pos in enumerate(section_positions):
            # Determine radius and circumference of this section
            circumference = 2*np.pi*radius
            # Determine the angle difference (in radians) between points
            theta_dist = distance_between_points / radius
            for j in range(int(circumference / distance_between_points)):
                theta = theta_dist * j
                # Determine cartesian coordinates for the point along the circular section of the frustum
                point_on_circle = p1 + section_pos + radius*np.cos(theta)*ortho_vector + radius*np.sin(theta)*normal_vector
                points.append(point_on_circle)

        return points

    def orthogonal_vector(self, v):
        '''
        Two Euclidean vectors are orthogonal if and only if their dot product is zero.
        '''
        # Find first element in v that is nonzero
        m = np.argmax(np.abs(v))
        y = np.zeros(len(v))
        y[(m+1) % len(v)] = 1
        return np.cross(v, y)

    def wrap_to_pi(self, angles: np.ndarray) -> np.ndarray:
        """
        Wrap angles to [-pi, pi].
        angles: shape (...), in radians.
        """
        # First, wrap to [0, 2*pi)
        wrapped = np.remainder(angles, 2.0 * np.pi)
        # Then subtract 2*pi for any angles > pi
        mask = wrapped > np.pi
        wrapped[mask] = wrapped[mask] - 2.0 * np.pi
        return wrapped

    def clamp_to_limits(self,
                        angles: np.ndarray,
                        lower_limits: np.ndarray,
                        upper_limits: np.ndarray) -> np.ndarray:
        """
        Clamp each joint angle to [lower, upper].
        angles: shape (batch_size, num_joints) or (num_joints,)
        lower_limits, upper_limits: shape (num_joints,)
        """
        angles = np.asarray(angles, dtype=float)
        lower_limits = np.asarray(lower_limits, dtype=float)
        upper_limits = np.asarray(upper_limits, dtype=float)

        # Expand dimensions if necessary for broadcasting
        while angles.ndim > lower_limits.ndim:
            lower_limits = np.expand_dims(lower_limits, axis=0)
            upper_limits = np.expand_dims(upper_limits, axis=0)

        # Clamp
        clamped = np.maximum(angles, lower_limits)
        clamped = np.minimum(clamped, upper_limits)
        return clamped
