import pybullet as p
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# grasp generation
from utils.grasp_utils import get_human_arm_pcd_for_grasp_sampler
from grasp_sampler.antipodal_grasp_sampler import AntipodalGraspSampler

# utils
from utils.collision_utils import get_collision_fn
from utils.transform_utils import quaternion_from_matrix

# environment
from envs.base_env import BaseEnv


class GraspEnv(BaseEnv):
    def __init__(self, gui=False, seated=False):
        super().__init__(gui=gui, seated=seated)

    def reset(self):
        self.create_world()

    def create_world(self):
        super().create_world()

        # initialize robot parameters
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_eef_grasp = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14],
                              world_to_eef[1]]
        eef_grasp_to_world = self.bc.invertTransform(world_to_eef_grasp[0], world_to_eef_grasp[1])
        eef_grasp_to_eef = self.bc.multiplyTransforms(eef_grasp_to_world[0], eef_grasp_to_world[1],
                                                      world_to_eef[0], world_to_eef[1])
        self.eef_grasp_to_eef = eef_grasp_to_eef

        # initialize collision checker        
        robot_obstacles = [self.bed_id, self.robot_2.id, self.cube_2_id, self.humanoid._humanoid]
        self.robot_in_collision = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=robot_obstacles,
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)

    def generate_grasps(self, q_H):
        def quaternion_dot(q1, q2):
            """ Compute the dot product of two quaternions """
            return np.dot(q1, q2)

        def check_perpendicularity(qA, qB):
            """ Check if the orientations are perpendicular """
            # Normalize quaternions to ensure correct dot product
            qA = qA / np.linalg.norm(qA)
            qB = qB / np.linalg.norm(qB)
            
            # Compute the dot product & compare deviation from zero
            dot_product = quaternion_dot(qA, qB)
            deviation = np.abs(dot_product)
            
            return deviation
        
        # initialize human arm and get its point cloud
        self.reset_human_arm(q_H)
        right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        right_wrist = self.bc.getLinkState(self.humanoid._humanoid, self.right_wrist)[:2]
        m = R.from_quat(right_elbow[1]).as_matrix()
        inward_vec = m[:, 1]  # inward vec is the green axis (rgb axis)

        # generate object point cloud
        point_cloud = get_human_arm_pcd_for_grasp_sampler(self, client_id=self.bc._client)
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
        pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=15))
        pc_ply.orient_normals_consistent_tangent_plane(50)

        # generate antipodal grasp samples
        sampler = AntipodalGraspSampler(obj_inward_vector=inward_vec, max_num_surface_points=100, num_samples=10)
        prev_time = time.time()
        grasp_matrices = sampler.generate_grasps(pc_ply, vis=False)
        print(f"Generated {len(grasp_matrices)} grasps. Time: {time.time()-prev_time}.")

        # test each grasp sample
        q_R_grasp_samples = []
        grasp_pose_samples = []
        world_to_eef_goals = []
        best_q_R_grasp = None
        best_world_to_grasp = None
        best_world_to_eef_goal = None
        best_combined_score = float('inf')  # Initialize with a high value for comparison

        deviations = []
        distances = []

        for grasp in grasp_matrices:
            world_to_eef = self.bc.multiplyTransforms(grasp[:3, 3], quaternion_from_matrix(grasp),
                                                    self.eef_grasp_to_eef[0], self.eef_grasp_to_eef[1])
            q_R_grasp = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, 
                                                        world_to_eef[0], world_to_eef[1],
                                                        self.robot.arm_lower_limits, self.robot.arm_upper_limits, 
                                                        self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                        maxNumIterations=50)
            q_R_grasp = [q_R_grasp[i] for i in range(len(self.robot.arm_controllable_joints))]

            self.reset_robot(self.robot, q_R_grasp)
            eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
            dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))

            if not self.robot_in_collision(q_R_grasp) and dist <= 0.01:
                q_R_grasp_samples.append(q_R_grasp)
                grasp_pose_samples.append([grasp[:3, 3], quaternion_from_matrix(grasp)])
                world_to_eef_goals.append(world_to_eef)

                # Calculate deviation from right elbow quaternion (Criteria 1)
                grasp_quaternion = quaternion_from_matrix(grasp)
                deviation = check_perpendicularity(right_elbow[1], grasp_quaternion)
                deviations.append(deviation)

                # Calculate distance from right wrist (Criteria 2)
                distance = np.linalg.norm(np.array(grasp[:3, 3]) - np.array(right_wrist[0]))
                distances.append(distance)

        # Normalize both the deviations and distances for scoring
        if deviations:
            deviations = np.array(deviations)
            distances = np.array(distances)

            deviations_norm = (deviations - deviations.min()) / (deviations.max() - deviations.min() + 1e-8)
            distances_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

            # Calculate weighted scores and select the best grasp
            for i in range(len(q_R_grasp_samples)):
                weighted_score = 0.4 * deviations_norm[i] + 0.6 * distances_norm[i]
                if weighted_score < best_combined_score:
                    best_combined_score = weighted_score
                    best_q_R_grasp = q_R_grasp_samples[i]
                    best_world_to_grasp = grasp_pose_samples[i]
                    best_world_to_eef_goal = world_to_eef_goals[i]

        print(f'No collision grasps: {len(q_R_grasp_samples)}')
        
        if len(q_R_grasp_samples) == 0:
            raise ValueError('No grasp available')

        return q_R_grasp_samples, grasp_pose_samples, world_to_eef_goals, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal