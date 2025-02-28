import pybullet as p
import time
import numpy as np

# mppi planner (ramp)
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower
from mppi_planning.mppi_human_handshake import MPPI_H_Handshake

# utils
from envs.utils.collision_utils import get_collision_fn
from envs.wiping_task.targets_util import TargetsUtil
from envs.utils.transform_utils import compute_matrix, inverse_matrix
from envs.utils.point_cloud_utils import *

# environment
from envs.base_env import BaseEnv


# urdf paths
robot_urdf_location = 'envs/agents/pybullet_ur5/urdf/ur5_robotiq_85.urdf'
scene_urdf_location = 'resources/environment/environment.urdf'
control_points_location = 'resources/ur5_control_points/T_control_points.json'
control_points_number = 166

# UR5 parameters
LINK_FIXED = 'base_link'
LINK_EE = 'ee_link'
LINK_SKELETON = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
]


class ManipulationEnv(BaseEnv):
    def __init__(self, gui=True, seated=False):
        super().__init__(gui=gui, seated=seated)
        self.bc.setPhysicsEngineParameter(numSolverIterations=200)
        self.targets_util = TargetsUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()

        # get 'static' obstacle point cloud
        self.static_obstacles = [self.cube_id, self.cube_2_id]
        self.static_obs_pcd = self.get_obstacle_point_cloud(self.static_obstacles)

        ### wiping robot parameters
        # initialize collision checker
        robot_2_obstacles = [self.bed_id, self.humanoid._humanoid, self.cube_id]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=robot_2_obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.189], target_orn]
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)

        self.target_orn = target_orn
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)

        # generate targets
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_right_arm,
                                            self.robot_2, self.tool,
                                            self.target_closer_to_eef, self.robot_2_in_collision)
        self.targets_util.generate_new_targets_pose()
        self.targets_util.generate_targets()
        self.targets_util.initialize_deleted_targets_list()

    def create_world(self):
        super().create_world()
        self.human_cid = None
        self.tool_cid = None

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

    def init_tool(self):
        # close gripper
        for _ in range(50):
            self.robot_2.move_gripper(0.04)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

        # initialize tool for wiping task
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)        
        world_to_tool = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14], [0,0,0,1]]
        
        eef_to_tool = self.bc.multiplyTransforms(positionA=eef_to_world[0], orientationA=eef_to_world[1], 
                                                 positionB=world_to_tool[0], orientationB=world_to_tool[1], physicsClientId=self.bc._client)
        self.eef_to_tool = eef_to_tool
        self.tool = self.bc.loadURDF("./envs/urdf/wiper.urdf", basePosition=world_to_tool[0], baseOrientation=world_to_tool[1], physicsClientId=self.bc._client)

        # disable collisions between the tool and robot
        for j in self.robot_2.arm_controllable_joints:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.robot_2.id, self.tool, j, tj, False, physicsClientId=self.bc._client)

        # disable collisions between the tool and human arm
        for j in self.human_right_arm:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.humanoid._humanoid, self.tool, j, tj, False, physicsClientId=self.bc._client)

        # disable collisions between the second robot (wiping) and human arm
        for j in self.human_right_arm:
            for tj in list(range(self.bc.getNumJoints(self.robot_2.id, physicsClientId=self.bc._client))):
                self.bc.setCollisionFilterPair(self.humanoid._humanoid, self.robot_2.id, j, tj, False, physicsClientId=self.bc._client)

        # disable collisions between robot (manip) and second robot (wiping)
        for j in list(range(self.bc.getNumJoints(self.robot.id, physicsClientId=self.bc._client))):
            for tj in list(range(self.bc.getNumJoints(self.robot_2.id, physicsClientId=self.bc._client))):
                self.bc.setCollisionFilterPair(self.robot.id, self.robot_2.id, j, tj, False, physicsClientId=self.bc._client)

    def set_arm_joint_range(self, shoulder_reduction_group):
        if shoulder_reduction_group == 'A':
            self.human_arm_lower_limits = [self.human_arm_lower_limits[0]+0.2617,
                                           self.human_arm_lower_limits[1]+0.3086,
                                           self.human_arm_lower_limits[2]+0.4278,
                                           self.human_arm_lower_limits[3]]
            self.human_arm_upper_limits = [self.human_arm_upper_limits[0]-0.2617,
                                           self.human_arm_upper_limits[1]-0.3086,
                                           self.human_arm_upper_limits[2]-0.4278,
                                           self.human_arm_upper_limits[3]]
        elif shoulder_reduction_group == 'B':
            self.human_arm_lower_limits = [self.human_arm_lower_limits[0]+0.3544,
                                           self.human_arm_lower_limits[1]+0.5720,
                                           self.human_arm_lower_limits[2]+0.5149,
                                           self.human_arm_lower_limits[3]]
            self.human_arm_upper_limits = [self.human_arm_upper_limits[0]-0.3544,
                                           self.human_arm_upper_limits[1]-0.5720,
                                           self.human_arm_upper_limits[2]-0.5149,
                                           self.human_arm_upper_limits[3]]
        elif shoulder_reduction_group == 'C':
            self.human_arm_lower_limits = [self.human_arm_lower_limits[0]+0.5734,
                                           self.human_arm_lower_limits[1]+0.7114,
                                           self.human_arm_lower_limits[2]+0.7617,
                                           self.human_arm_lower_limits[3]]
            self.human_arm_upper_limits = [self.human_arm_upper_limits[0]-0.5734,
                                           self.human_arm_upper_limits[1]-0.7114,
                                           self.human_arm_upper_limits[2]-0.7617,
                                           self.human_arm_upper_limits[3]]
        elif shoulder_reduction_group == 'D':
            self.human_arm_lower_limits = [self.human_arm_lower_limits[0]+0.8175,
                                           self.human_arm_lower_limits[1]+0.7240,
                                           self.human_arm_lower_limits[2]+1.0662,
                                           self.human_arm_lower_limits[3]]
            self.human_arm_upper_limits = [self.human_arm_upper_limits[0]-0.8175,
                                           self.human_arm_upper_limits[1]-0.7240,
                                           self.human_arm_upper_limits[2]-1.0662,
                                           self.human_arm_upper_limits[3]]
        
    ###### POINT CLOUD ######
    def compute_env_pcd(self, robot, resolution=8, add_bed_padding=False, add_chest_padding=False, exclude_robot_fingers=False):
        if exclude_robot_fingers:
            num_joints = 10
        else:
            num_joints = None

        link_to_separate = [self.right_elbow, self.right_wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate, client_id=self.bc._client, resolution=resolution,
                                                           extra_length=0.03, extra_radius=0.0)
        robot_pcd = self.get_robot_point_cloud(robot, num_joints)
        bed_pcd = self.get_bed_point_cloud(self.bed_id, add_bed_padding)

        if add_chest_padding:
            chest = self.bc.getLinkState(self.humanoid._humanoid, self.chest)[4:6]
            half_extents = [0.09, 0.5, 0.23]
            chest_padding_pcd = generate_box_vertices(half_extents, chest[0], chest[1], resolution=20)
            human_pcd = np.vstack((human_pcd, chest_padding_pcd))

        env_pcd = np.vstack((self.static_obs_pcd, bed_pcd, robot_pcd, human_pcd))
        right_arm_pcd = np.array(separate_pcd)
        right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.right_shoulder, 
                                                                                 resolution=resolution, client_id=self.bc._client,
                                                                                 scale_radius=1.0, scale_height=1.7)

        return env_pcd, right_arm_pcd, right_shoulder_pcd
    ###### POINT CLOUD ######

    ###### TRAJ PLANNER & FOLLOWER ######
    def init_traj_planner(self, world_to_robot_base, clamp_by_human, q_H_init=None, q_R_init=None):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Instantiate mppi H clamp
        if clamp_by_human:
            assert q_H_init is not None
            assert q_R_init is not None
            mppi_H_handshake = MPPI_H_Handshake(self.eef_to_cp, self.right_elbow_joint_to_cp, 
                                                self.robot_base_pose, self.human_base_pose,
                                                self.human_arm_lower_limits, self.human_arm_upper_limits, self.human_controllable_joints,
                                                human_rest_poses=q_H_init, robot_rest_poses=q_R_init, 
                                                seated=self.seated)
        else:
            mppi_H_handshake = None

        # Instantiate trajectory planner
        trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = control_points_number,
            mppi_H_handshake = mppi_H_handshake,
            world_to_robot_base = world_to_robot_base,
        )
        print("Instantiated trajectory planner")

        return trajectory_planner

    def init_mppi_planner(self, trajectory_planner, current_joint_angles, target_joint_angles, clamp_by_human, init_traj=[]):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        if clamp_by_human:
            mppi_control_limits = [
                -0.05 * np.ones(N_JOINTS),
                0.05 * np.ones(N_JOINTS)
            ]
            mppi_covariance = 0.005
            mppi_nsamples = 500
            mppi_lambda = 0.5
            mppi_cost_weight_convergence = 1000.0
            mppi_cost_weight_path_length = 600.0
            mppi_cost_weight_collision = 300.0
            collision_threshold = 0.01
        else:
            mppi_control_limits = [
                -0.5 * np.ones(N_JOINTS),
                0.5 * np.ones(N_JOINTS)
            ]
            mppi_covariance = 0.05
            mppi_nsamples = 500
            mppi_lambda = 1.0
            mppi_cost_weight_convergence = 1000.0
            mppi_cost_weight_path_length = 200.0
            mppi_cost_weight_collision = 800.0
            collision_threshold = 0.03

        # Update whether to clamp_by_human
        trajectory_planner.update_clamp_by_human(clamp_by_human)

        # Instantiate MPPI object
        trajectory_planner.instantiate_mppi_ja_to_ja(
            current_joint_angles,
            target_joint_angles,
            init_traj=init_traj,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
            waypoint_density = 5,
            mppi_cost_weight_convergence = mppi_cost_weight_convergence,
            mppi_cost_weight_path_length = mppi_cost_weight_path_length,
            mppi_cost_weight_collision = mppi_cost_weight_collision,
            collision_threshold = collision_threshold
        )
        print('Instantiated MPPI object')

        return trajectory_planner

    def get_mppi_trajectory(self, trajectory_planner, current_joint_angles):
        # Plan trajectory
        start_time = time.time()
        trajectory = trajectory_planner.get_mppi_rollout(current_joint_angles)
        print("planning time : ", time.time()-start_time)
        # print(np.array(trajectory))
        return trajectory
    
    def init_traj_follower(self, world_to_robot_base):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Trajectory Follower initialization
        trajectory_follower = TrajectoryFollower(
            joint_limits = JOINT_LIMITS,
            robot_urdf_location = robot_urdf_location,
            control_points_json = control_points_location,
            link_fixed = LINK_FIXED,
            link_ee = LINK_EE,
            link_skeleton = LINK_SKELETON,
            control_points_number = control_points_number,
            world_to_robot_base = world_to_robot_base,
        )
        print('trajectory follower instantiated')

        return trajectory_follower
    ###### TRAJ PLANNER & FOLLOWER ######

    def attach_human_arm_to_eef(self, attach_to_gripper=False, right_arm_pcd=None, trajectory_planner=None):
        # attach human arm (obj) to eef (body)
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_to_right_elbow = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                                        world_to_right_elbow[0], world_to_right_elbow[1])

        if self.right_elbow_to_cp is None or self.eef_to_cp is None:
            raise ValueError('right_elbow_to_cp or eef_to_cp not initialized.')
        
        if self.human_cid is None:
            self.human_cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                                                      parentLinkIndex=self.robot.eef_id,
                                                      childBodyUniqueId=self.humanoid._humanoid,
                                                      childLinkIndex=self.right_elbow,
                                                      jointType=p.JOINT_FIXED,
                                                      jointAxis=(0, 0, 0),
                                                      parentFramePosition=eef_to_right_elbow[0],
                                                      parentFrameOrientation=eef_to_right_elbow[1],
                                                      childFramePosition=(0, 0, 0),
                                                      childFrameOrientation=(0, 0, 0))

        # attach human arm pcd as part of robot control point
        if attach_to_gripper:
            assert trajectory_planner is not None
            assert right_arm_pcd is not None

            # compute transform matrix from robot's gripper to object frame
            world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
            world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                     self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
            T_eef_to_object = compute_matrix(translation=self.eef_to_cp[0], rotation=self.eef_to_cp[1], rotation_type='quaternion')

            # compute transform matrix for inverse of object pose in world frame
            T_world_to_object = compute_matrix(translation=world_to_cp[0], rotation=world_to_cp[1], rotation_type='quaternion')
            T_object_to_world = inverse_matrix(T_world_to_object)

            trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=right_arm_pcd,
                                                 T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)

            return T_eef_to_object, T_object_to_world
    
    def detach_human_arm_from_eef(self, detach_from_gripper=False, trajectory_planner=None):
        if self.human_cid is not None:
            self.bc.removeConstraint(self.human_cid)
        self.human_cid = None

        if detach_from_gripper:
            assert trajectory_planner is not None
            trajectory_planner.detach_from_gripper()

    def attach_tool(self, attach_to_gripper=False, tool_pcd=None, trajectory_planner=None):
        # reset tool and attach it to eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                   self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
        self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)

        # create constraint that keeps the tool in the gripper
        self.tool_cid = self.bc.createConstraint(parentBodyUniqueId=self.robot_2.id,
                                                 parentLinkIndex=self.robot_2.eef_id,
                                                 childBodyUniqueId=self.tool,
                                                 childLinkIndex=-1,
                                                 jointType=p.JOINT_FIXED,
                                                 jointAxis=(0, 0, 0),
                                                 parentFramePosition=self.eef_to_tool[0],
                                                 parentFrameOrientation=self.eef_to_tool[1],
                                                 childFramePosition=(0, 0, 0),
                                                 childFrameOrientation=(0, 0, 0),
                                                 physicsClientId=self.bc._client)
        
        # attach tool pcd as part of robot control point
        if attach_to_gripper:
            assert trajectory_planner is not None
            assert tool_pcd is not None

            # compute transform matrix from robot's gripper to object frame
            T_eef_to_object = compute_matrix(translation=self.eef_to_tool[0], rotation=self.eef_to_tool[1], rotation_type='quaternion')

            # compute transform matrix for inverse of object pose in world frame
            T_world_to_object = compute_matrix(translation=world_to_tool[0], rotation=world_to_tool[1], rotation_type='quaternion')
            T_object_to_world = inverse_matrix(T_world_to_object)

            trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=tool_pcd,
                                                 T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)

            return T_eef_to_object, T_object_to_world
        
    def detach_tool(self, detach_from_gripper=False, trajectory_planner=None):
        if self.tool_cid is not None:
            self.bc.removeConstraint(self.tool_cid)
        self.tool_cid = None
        self.bc.resetBasePositionAndOrientation(self.tool, [-100,-100,-100], [0,0,0,1], physicsClientId=self.bc._client)

        if detach_from_gripper:
            assert trajectory_planner is not None
            trajectory_planner.detach_from_gripper()
        
    def is_near_goal_W_space(self, world_to_eef, world_to_eef_goal, threshold=0.03):
        dist = np.linalg.norm(np.array(world_to_eef_goal[0]) - np.array(world_to_eef[0]))
        if dist <= threshold:
            return True
        else:
            return False
    
    def interpolate_trajectory(self, robot_traj, alpha=0.5):
        new_traj = []
        for i in range(len(robot_traj) - 1):
            q_R_i = np.array(robot_traj[i])
            q_R_next = np.array(robot_traj[i + 1])
            
            interpolated_point = (1 - alpha) * q_R_i + alpha * q_R_next
            new_traj.append(robot_traj[i])  # Append the current point
            new_traj.append(interpolated_point.tolist())  # Append the interpolated point

        new_traj.append(robot_traj[-1])  # Append the last point to complete the trajectory

        return new_traj
    
    def get_init_traj_from_q_H(self, q_H_init, q_H_goal, q_R_init):
        q_H_traj = []
        q_H_traj.append(q_H_init)
        q_H_traj.append(q_H_goal)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        
        q_R_traj = []
        q_R_traj.append(q_R_init)
        prev_q_R = q_R_init
        
        for q_H in q_H_traj[1:]:
            self.reset_human_arm(q_H)
            q_R = self.get_q_R_from_elbow_pose(prev_q_R)
            q_R_traj.append(q_R)
            prev_q_R = q_R
    
        return q_H_traj, q_R_traj

    def get_q_R_from_elbow_pose(self, prev_q_R):
        world_to_right_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1],
                                                 self.right_elbow_joint_to_cp[0], self.right_elbow_joint_to_cp[1])
        world_to_eef = self.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                  self.cp_to_eef[0], self.cp_to_eef[1])
        q_R_goal = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                      self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, 
                                                      restPoses=prev_q_R,
                                                      maxNumIterations=50)
        q_R_goal = [q_R_goal[i] for i in range(len(self.robot.arm_controllable_joints))]
        return q_R_goal
    
    def get_obj_base_pose(self, obj):
        if obj == "bed":
            return self.bc.getBasePositionAndOrientation(self.bed_id)
        elif obj == "robot":
            return self.bc.getBasePositionAndOrientation(self.robot.id)
        elif obj == "robot_2":
            return self.bc.getBasePositionAndOrientation(self.robot_2.id)
        elif obj == "humanoid":
            return self.bc.getBasePositionAndOrientation(self.humanoid._humanoid)
        elif obj == "cube":
            return self.bc.getBasePositionAndOrientation(self.cube_id)
        elif obj == "cube_2":
            return self.bc.getBasePositionAndOrientation(self.cube_2_id)
        else:
            raise ValueError("invalid obj name!")
        
    
if __name__ == '__main__':
    manip_env = ManipulationEnv()
    manip_env.reset()
    
    