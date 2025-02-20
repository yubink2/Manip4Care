import os, inspect
import numpy as np
import time
import sys
import math

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

# humanoid
from agents.humanoid_supine import Humanoid
from agents.humanoid_sitting import HumanoidSeated

# robot
from pybullet_ur5.robot import UR5Robotiq85
from utils.collision_utils import get_collision_fn

# utils
from wiping_task.util import Util
from wiping_task.targets_util import TargetsUtil
from wiping_task.score_util import ScoreUtil
from utils.transform_utils import *

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *

class WipingDemo():
    def __init__(self, gui=False, seated=False):
        # Start the bullet physics server
        self.gui = gui
        self.seated = seated

        if self.gui:
            self.bc = BulletClient(connection_mode=p.GUI)
        else:
            self.bc = BulletClient(connection_mode=p.DIRECT)

        self.util = Util(self.bc._client)
        self.targets_util = TargetsUtil(self.bc._client, self.util)
        self.score_util = ScoreUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_right_arm,
                                            self.robot_2, self.tool,
                                            self.target_closer_to_eef, self.robot_2_in_collision)
        self.score_util.init_score_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_controllable_joints,
                                        self.robot, self.robot_2, self.tool,
                                        self.target_closer_to_eef, self.robot_2_in_collision, self.robot_in_collision)

        self.targets_util.generate_new_targets_pose()
        self.targets_util.generate_targets()
        self.targets_util.initialize_deleted_targets_list()
        
    def reset_wiping_setup(self, q_H, targeted_arm, reorder_feasible_targets=True):
        self.reset_human_arm(q_H)
        self.lock_human_joints(q_H)
        self.targets_util.update_targets()

        # feasible targets
        feasible_targets_found = self.targets_util.get_feasible_targets_pos(targeted_arm=targeted_arm)
        if not feasible_targets_found:
            return feasible_targets_found
        
        if reorder_feasible_targets:
            self.targets_util.reorder_feasible_targets(targeted_arm=targeted_arm)
        self.targets_util.mark_feasible_targets()
        return feasible_targets_found

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, 0, physicsClientId=self.bc._client) 

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04), physicsClientId=self.bc._client)
        if self.seated:
            self.bed_id = self.bc.loadURDF("./urdf/wheelchair.urdf", globalScaling=0.8, useFixedBase=True)
        else:
            self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True)

        # load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        self.human_base_pose = (human_base_pos, human_base_orn)
        
        if self.seated:
            human_base_pos = (0, 0, 0.4)
            human_base_orn = self.bc.getQuaternionFromEuler((1.57, 0, 1.57))
            self.human_base_pose = (human_base_pos, human_base_orn)
            self.humanoid = HumanoidSeated(self.bc, baseShift=self.human_base_pose[0], ornShift=self.human_base_pose[1])

            self.chest = 2
            self.right_shoulder = 6
            self.right_elbow = 7
            self.right_wrist = 8
            self.human_controllable_joints = [3, 4, 5, 7]
            self.human_right_arm = [3, 4, 5, 6, 7, 8]
            self.human_rest_poses = [2.0, -0.5, -1.8, 1.0]

            # initialize human parameters
            shoulder_min = [-3.141560999234642, -1.0003133072549704, -3.1410543732632656]  # order: [yaw, pitch, roll]
            shoulder_max = [3.14156809556302, 1.469721523320065, 3.140911127761456]        # order: [yaw, pitch, roll]
        
        else:
            self.humanoid = Humanoid(self.bc, baseShift=self.human_base_pose[0], ornShift=self.human_base_pose[1])
        
            self.chest = 2
            self.right_shoulder = 6
            self.right_elbow = 7
            self.right_wrist = 8
            self.human_controllable_joints = [3, 4, 5, 7]
            self.human_right_arm = [3, 4, 5, 6, 7, 8]
            self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

            # initialize human parameters
            shoulder_min = [-3.141560999234642, -1.0003133072549704, -3.1410543732632656]  # order: [yaw, pitch, roll]
            shoulder_max = [3.14156809556302, 1.469721523320065, 3.140911127761456]        # order: [yaw, pitch, roll]
        
        elbow_min = [0.0]
        elbow_max = [2.85735]
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max
        self.human_arm_joint_ranges =  list(np.array(self.human_arm_upper_limits) - np.array(self.human_arm_lower_limits))

        # load first robot (manipulation)
        self.robot_base_pose = ((0.5, 0.8, 0.25), (0, 0, 0))
        self.cube_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
                                   (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True)
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        for _ in range(50):
            self.robot.reset()
            self.robot.open_gripper()

        # load second robot (wiping)
        self.robot_2_base_pose = ((0.65, 0, 0.25), (0, 0, 1.57))
        self.cube_2_id = p.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        self.targets_pos_on_upperarm = None
        self.targets_pos_on_forearm = None

        # initialize collision checker (robot 2)
        obstacles = [self.bed_id, self.humanoid._humanoid, self.robot.id, self.cube_id]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # initialize collision checker (robot)
        robot_obstacles = [self.bed_id, self.robot_2.id, self.cube_2_id, self.humanoid._humanoid]
        self.robot_in_collision = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=robot_obstacles,
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target_3 = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.26], target_orn]
        world_to_target_2 = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.23], target_orn]
        world_to_target_1 = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.21], target_orn]
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.189], target_orn]

        self.target_orn = target_orn
        target_3_to_world = self.bc.invertTransform(world_to_target_3[0], world_to_target_3[1], physicsClientId=self.bc._client)
        target_2_to_world = self.bc.invertTransform(world_to_target_2[0], world_to_target_2[1], physicsClientId=self.bc._client)
        target_1_to_world = self.bc.invertTransform(world_to_target_1[0], world_to_target_1[1], physicsClientId=self.bc._client)
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)
        self.target_3_to_eef = self.bc.multiplyTransforms(target_3_to_world[0], target_3_to_world[1],
                                                          world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_2_to_eef = self.bc.multiplyTransforms(target_2_to_world[0], target_2_to_world[1],
                                                          world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_1_to_eef = self.bc.multiplyTransforms(target_1_to_world[0], target_1_to_world[1],
                                                          world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
    
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
        self.tool = self.bc.loadURDF("./urdf/wiper.urdf", basePosition=world_to_tool[0], baseOrientation=world_to_tool[1], physicsClientId=self.bc._client)

        # disable collisions between the tool and robot
        for j in self.robot_2.arm_controllable_joints:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.robot_2.id, self.tool, j, tj, False, physicsClientId=self.bc._client)

        if not self.gui:
            self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)
        
    def attach_tool(self):
        # reset tool and attach it to eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
        self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)

        # create constraint that keeps the tool in the gripper
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot_2.id,
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
        
    def detach_tool(self):
        if self.cid is not None:
            self.bc.removeConstraint(self.cid)
        self.cid = None
        self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)

    def compute_feasible_targets_robot_traj(self):
        def is_discontinuous(q_old, q_new, angle_threshold = math.pi/2):
            """
            Returns True if *any* joint difference between q_old and q_new
            is larger than 'angle_threshold'. Differences are wrapped to [-pi, pi].
            """
            if q_old is None or q_new is None:
                return None
            
            q_old = np.asarray(q_old)
            q_new = np.asarray(q_new)

            # Compute raw difference
            diff = q_new - q_old

            # # Wrap each difference into [-pi, pi] range
            # # (diff + pi) mod 2pi - pi
            # diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi

            # Check if any joint difference exceeds the angle_threshold
            if np.any(np.abs(diff) < angle_threshold):
                return q_new
            return None
        
        def compute_and_validate_ik_sol(target_pos_world, target_orn_world, target_alpha_to_eef, 
                                        pos_threshold, orn_threshold, q_R_prev):
            """Helper function to compute IK for robot_2 and return valid wiping joint states."""
            # compute desired world_to_eef (check if it can get closer to the target point)
            world_to_eef = self.bc.multiplyTransforms(target_pos_world, target_orn_world,
                                                      target_alpha_to_eef[0], target_alpha_to_eef[1], physicsClientId=self.bc._client)

            # set robot initial joint state
            q_robot_2 = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, world_to_eef[0], world_to_eef[1],
                                                           lowerLimits=self.robot_2.arm_lower_limits, upperLimits=self.robot_2.arm_upper_limits, 
                                                           jointRanges=self.robot_2.arm_joint_ranges, restPoses=q_R_prev,
                                                           maxNumIterations=50, physicsClientId=self.bc._client)
            q_robot_2 = [q_robot_2[i] for i in range(len(self.robot_2.arm_controllable_joints))]
            # ###
            # q_robot_2 = self.util.wrap_to_pi(q_robot_2)
            # q_robot_2 = np.clip(q_robot_2, self.robot_2.arm_lower_limits, self.robot_2.arm_upper_limits)
            # ###

            for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                self.bc.resetJointState(self.robot_2.id, joint_id, q_robot_2[i], physicsClientId=self.bc._client)

            # check if config is valid
            eef_pose = p.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
            pos_dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
            dot_product = np.abs(np.dot(world_to_eef[1], eef_pose[1]))
            orn_dist = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            if not self.robot_2_in_collision(q_robot_2) and pos_dist <= pos_threshold and orn_dist < orn_threshold:
                return q_robot_2
            else:
                return None
        
        robot_traj = []
        q_robot_previous = self.robot_2.arm_rest_poses

        prev_target_pos_world = None
        prev_target_orn_world = None
        count = 0
        last_paddings_count = 0

        pos_threshold = 0.02
        orn_threshold = np.deg2rad(15)

        for target_pos_world, target_orn_world in zip(self.targets_util.feasible_targets_pos_world, 
                                                      self.targets_util.feasible_targets_orn_world):
            # compute and check valid wiping joint states
            q_robot_2_closer = compute_and_validate_ik_sol(target_pos_world, target_orn_world, self.target_closer_to_eef,
                                                           pos_threshold=pos_threshold, orn_threshold=orn_threshold,
                                                           q_R_prev=q_robot_previous)
            # ###
            # if count > 0:
            #     q_robot_2_closer = is_discontinuous(q_old=q_robot_previous, q_new=q_robot_2_closer)
            # ###

            if q_robot_2_closer is not None:
                # compute IK for paddings
                q_robot_2_step_3 = compute_and_validate_ik_sol(target_pos_world, target_orn_world, self.target_3_to_eef,
                                                            pos_threshold=pos_threshold, orn_threshold=orn_threshold,
                                                            q_R_prev=q_robot_2_closer)
                q_robot_2_step_2 = compute_and_validate_ik_sol(target_pos_world, target_orn_world, self.target_2_to_eef,
                                                            pos_threshold=pos_threshold, orn_threshold=orn_threshold,
                                                            q_R_prev=q_robot_2_closer)
                q_robot_2_step_1 = compute_and_validate_ik_sol(target_pos_world, target_orn_world, self.target_1_to_eef,
                                                            pos_threshold=pos_threshold, orn_threshold=orn_threshold,
                                                            q_R_prev=q_robot_2_closer)
                # ###
                # if count > 0:
                #     q_robot_2_step_3 = is_discontinuous(q_old=q_robot_2_closer, q_new=q_robot_2_step_3)
                #     q_robot_2_step_2 = is_discontinuous(q_old=q_robot_2_step_3, q_new=q_robot_2_step_2)
                #     q_robot_2_step_1 = is_discontinuous(q_old=q_robot_2_step_2, q_new=q_robot_2_step_1)
                # ###

                # if none of the paddings are reachable, skip
                if q_robot_2_step_3 is None and q_robot_2_step_2 is None and q_robot_2_step_1 is None:
                    continue
                
                count += 1

                # 1. first one -- add before paddings
                if count == 1:
                    if q_robot_2_step_3 is not None:
                        robot_traj.append(q_robot_2_step_3)
                    if q_robot_2_step_2 is not None:
                        robot_traj.append(q_robot_2_step_2)
                    if q_robot_2_step_1 is not None:
                        robot_traj.append(q_robot_2_step_1)

                # 1. check for break in near by target sequence (check if target orn is different)
                if prev_target_pos_world is not None and prev_target_orn_world is not None:
                    dot_product = np.abs(np.dot(prev_target_orn_world, target_orn_world))
                    orn_dist = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
                    if orn_dist > 1e-2:
                        # if break, keep after paddings & add before paddings
                        if q_robot_2_step_3 is not None:
                            robot_traj.append(q_robot_2_step_3)
                        if q_robot_2_step_2 is not None:
                            robot_traj.append(q_robot_2_step_2)
                        if q_robot_2_step_1 is not None:
                            robot_traj.append(q_robot_2_step_1)
                    else:
                        # if no break, remove after paddings
                        robot_traj = robot_traj[:len(robot_traj)-last_paddings_count]

                # 2. add wiping q_R to trajectory
                robot_traj.append(q_robot_2_closer)

                # 3. add after paddings
                last_paddings_count = 0
                if q_robot_2_step_1 is not None:
                    robot_traj.append(q_robot_2_step_1)
                    last_paddings_count += 1
                if q_robot_2_step_2 is not None:
                    robot_traj.append(q_robot_2_step_2)
                    last_paddings_count += 1
                if q_robot_2_step_3 is not None:
                    robot_traj.append(q_robot_2_step_3)
                    last_paddings_count += 1

                prev_target_pos_world = target_pos_world
                prev_target_orn_world = target_orn_world
                q_robot_previous = q_robot_2_closer

        self.robot_2.reset()

        return robot_traj
    
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
    
    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i], physicsClientId=self.bc._client)

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i], physicsClientId=self.bc._client)

    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def generate_random_q_H(self):
        q_H_0 = np.random.uniform(self.human_arm_lower_limits[0], self.human_arm_upper_limits[0])
        q_H_1 = np.random.uniform(self.human_arm_lower_limits[1], self.human_arm_upper_limits[1])
        q_H_2 = np.random.uniform(self.human_arm_lower_limits[2], self.human_arm_upper_limits[2])
        q_H_3 = np.random.uniform(self.human_arm_lower_limits[3], self.human_arm_upper_limits[3])

        return [q_H_0, q_H_1, q_H_2, q_H_3] 
    
    def human_in_collision(self):
        """Check if any part of the human arm collides with other objects."""
        contact_points = self.bc.getContactPoints(bodyA=self.humanoid._humanoid, physicsClientId=self.bc._client)
        for point in contact_points:
            if (point[2] in [self.bed_id]):
                return True
        return False

    def get_valid_q_H(self):
        """Reset the human arm and check for collisions until no collision is detected."""
        for _ in range(5000):
            q_H = self.generate_random_q_H()
            self.reset_human_arm(q_H)
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            if not self.human_in_collision():
                self.lock_human_joints(q_H)
                world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
                return q_H, world_to_right_elbow
        raise ValueError('valid human config not found!')

    def lock_human_joints(self, q_human):
        # Save original mass of each joint to restore later
        self.human_joint_masses = []
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            # Get the current dynamics info to save mass
            dynamics_info = self.bc.getDynamicsInfo(self.humanoid._humanoid, j, physicsClientId=self.bc._client)
            self.human_joint_masses.append(dynamics_info[0])  # Save mass (first item in tuple is mass)
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], targetVelocity=0, physicsClientId=self.bc._client)

    def lock_robot_arm_joints(self, robot, q_robot):
        # Save original mass of each joint to restore later
        self.robot_joint_masses = []
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            dynamics_info = self.bc.getDynamicsInfo(robot.id, j, physicsClientId=self.bc._client)
            self.robot_joint_masses.append(dynamics_info[0])  # Save mass
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(robot.id, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(self.robot.id, jointIndex=joint_id, targetValue=q_robot[i], targetVelocity=0, physicsClientId=self.bc._client)

    def unlock_human_joints(self, q_human):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            original_mass = self.human_joint_masses[j]
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], physicsClientId=self.bc._client)

    def unlock_robot_arm_joints(self, robot, q_robot):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            original_mass = self.robot_joint_masses[j]
            self.bc.changeDynamics(robot.id, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(self.robot.id, jointIndex=joint_id, targetValue=q_robot[i], physicsClientId=self.bc._client)

    def get_robot_joint_angles(self, robot):
        current_joint_angles = []
        for joint_id in robot.arm_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(robot.id, joint_id)[0])
        return current_joint_angles

    def get_human_joint_angles(self):
        current_joint_angles = []
        for joint_id in self.human_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(self.humanoid._humanoid, joint_id)[0])
        return current_joint_angles
    
    def get_eef_pose(self, robot, current_joint_angles, target_joint_angles):
        self.reset_robot(robot, target_joint_angles)
        eef_pose = self.bc.getLinkState(robot.id, robot.eef_id)[:2]
        self.reset_robot(robot, current_joint_angles)
        return eef_pose
    
    def step(self):
        self.bc.stepSimulation(physicsClientId=self.bc._client)
        self.targets_util.update_targets()

    ########## SCORE ##########
    def get_score(self, q_H_init, q_H_goal, q_robot, w_feasibility=0.9, w_closeness=0.1):
        self.targets_util.update_targets()
        self.score_util.reset(targets_pos_upperarm_world=self.targets_util.targets_pos_upperarm_world, 
                              targets_orn_upperarm_world=self.targets_util.targets_orn_upperarm_world, 
                              targets_pos_forearm_world=self.targets_util.targets_pos_forearm_world, 
                              targets_orn_forearm_world=self.targets_util.targets_orn_forearm_world, 
                              q_H=q_H_goal, q_robot=q_robot)

        feasibility_score = self.score_util.compute_score_by_feasibility()
        closeness_score = self.score_util.compute_score_by_closeness(q_H_init, q_H_goal)
        total_score = w_feasibility*feasibility_score + w_closeness*closeness_score

        return total_score
    ########## SCORE ##########
    
    def set_grasp_parameters(self, right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                             right_elbow_joint_to_cp, cp_to_right_elbow_joint, right_wrist_joint_to_cp, cp_to_right_wrist_joint):
        self.right_elbow_to_cp = right_elbow_to_cp
        self.cp_to_right_elbow = cp_to_right_elbow
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef
        self.right_elbow_joint_to_cp = right_elbow_joint_to_cp 
        self.cp_to_right_elbow_joint = cp_to_right_elbow_joint
        self.right_wrist_joint_to_cp = right_wrist_joint_to_cp
        self.cp_to_right_wrist_joint = cp_to_right_wrist_joint
    
    def compute_grasp_parameters(self, q_H, q_R_grasp, grasp):
        # compute right_elbow_to_cp
        self.reset_human_arm(q_H)
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_cp = (grasp[0], world_to_right_elbow[1])
        right_elbow_to_world = self.bc.invertTransform(world_to_right_elbow[0], world_to_right_elbow[1])
        right_elbow_to_cp = self.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
                                                       world_to_cp[0], world_to_cp[1])
        cp_to_right_elbow = self.bc.invertTransform(right_elbow_to_cp[0], right_elbow_to_cp[1])

        # compute right_elbow_joint_to_cp
        world_to_right_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        right_elbow_joint_to_world = self.bc.invertTransform(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1])
        right_elbow_joint_to_cp = self.bc.multiplyTransforms(right_elbow_joint_to_world[0], right_elbow_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        T_right_elbow_joint_to_cp = compute_matrix(translation=right_elbow_joint_to_cp[0], rotation=right_elbow_joint_to_cp[1])

        # compute right_wrist_joint_to_cp
        world_to_right_wrist_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_wrist)[4:6]
        right_wrist_joint_to_world = self.bc.invertTransform(world_to_right_wrist_joint[0], world_to_right_wrist_joint[1])
        right_wrist_joint_to_cp = self.bc.multiplyTransforms(right_wrist_joint_to_world[0], right_wrist_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        
        # compute eef_to_cp
        self.reset_robot(self.robot, q_R_grasp)
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                               world_to_cp[0], world_to_cp[1])
        cp_to_eef = self.bc.invertTransform(eef_to_cp[0], eef_to_cp[1])
        
        self.right_elbow_to_cp = right_elbow_to_cp
        self.cp_to_right_elbow = cp_to_right_elbow
        self.T_right_elbow_joint_to_cp = T_right_elbow_joint_to_cp
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef

        self.right_elbow_joint_to_cp = right_elbow_joint_to_cp
        self.cp_to_right_elbow_joint = self.bc.invertTransform(right_elbow_joint_to_cp[0], right_elbow_joint_to_cp[1])

        self.right_wrist_joint_to_cp = right_wrist_joint_to_cp
        self.cp_to_right_wrist_joint = self.bc.invertTransform(right_wrist_joint_to_cp[0], right_wrist_joint_to_cp[1])
    
    ###### ROBOT CONFIG VALIDATION ######
    def get_human_to_robot_dist(self, robot, q_H, q_robot):
        self.reset_human_arm(q_H)
        self.reset_robot(robot, q_robot)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=robot.id, bodyB=self.humanoid._humanoid, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if robot == self.robot_2:
                if linkA >= 4 and linkB in self.human_right_arm:
                    continue
            elif linkB in self.human_right_arm:
                continue

            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance

        return min_dist
    
    def get_bed_to_robot_dist(self, robot, q_robot):
        self.reset_robot(robot, q_robot)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=robot.id, bodyB=self.bed_id, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if robot == self.robot:
                if linkA >= 9:  # skip gripper fingers
                    continue

            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance

        return min_dist
    
    def get_cube_to_robot_dist(self, robot, q_robot, cube_id):
        self.reset_robot(robot, q_robot)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=robot.id, bodyB=cube_id, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if robot == self.robot:
                if linkA >= 9:  # skip gripper fingers
                    continue

            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance

        return min_dist
    
    def get_robot_to_robot_2_dist(self, q_robot, q_robot_2):
        self.reset_robot(self.robot, q_robot)
        self.reset_robot(self.robot_2, q_robot_2)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=self.robot_2.id, bodyB=self.robot.id, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if linkB >= 9:  # skip gripper fingers
                continue
        
            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance

        return min_dist
    
    def validate_q_R(self, q_H, q_R, check_goal=False):
        # feasibility check
        if min(q_R) < min(self.robot.arm_lower_limits) or max(q_R) > max(self.robot.arm_upper_limits):
            return False
        
        self.reset_robot(self.robot, q_R)
        self.robot_2.reset()
        self.reset_human_arm(q_H)
        
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                 self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
        cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
        eef_to_world = self.bc.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                cp_to_world[0], cp_to_world[1])
        world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])
        
        # collision check
        if self.robot_in_collision(q_R):
            return False

        # reachability check
        eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
        if dist > 0.03:
            return False

        if check_goal:
            # distance to human check
            dist = self.get_human_to_robot_dist(robot=self.robot, q_H=q_H, q_robot=q_R)
            if dist <= 0.05:
                return False

            # distance check from robot to bed (exclude gripper fingers)
            dist = self.get_bed_to_robot_dist(robot=self.robot, q_robot=q_R)
            if dist <= 0.05:
                return False
        
        return True
    
    def validate_q_robot_2(self, q_H, q_robot, q_robot_2):
        # feasibility check
        if min(q_robot_2) < min(self.robot_2.arm_lower_limits) or max(q_robot_2) > max(self.robot_2.arm_upper_limits):
            return False
        
        self.reset_robot(self.robot_2, q_robot_2)
        self.reset_robot(self.robot, q_robot)
        self.reset_human_arm(q_H)

        # distance check from human
        dist = self.get_human_to_robot_dist(robot=self.robot_2, q_H=q_H, q_robot=q_robot_2)
        if dist <= 0.05:
            return False

        # distance check from bed (exclude gripper fingers)
        dist = self.get_bed_to_robot_dist(robot=self.robot_2, q_robot=q_robot_2)
        if dist <= 0.05:
            return False
        
        # distance check from cube (exclude gripper fingers)
        dist = self.get_cube_to_robot_dist(robot=self.robot_2, q_robot=q_robot_2, cube_id=self.cube_id)
        if dist <= 0.05:
            return False
        
        # distance check from manip robot
        dist = self.get_robot_to_robot_2_dist(q_robot=q_robot, q_robot_2=q_robot_2)
        if dist <= 0.05:
            return False
        
        return True
    
    def is_not_discontinuous(self, q_old, q_new, angle_threshold = math.pi/2):
        q_old = np.asarray(q_old)
        q_new = np.asarray(q_new)

        # Compute raw difference
        diff = q_new - q_old

        # Check if any joint difference exceeds the angle_threshold
        if np.any(np.abs(diff) > angle_threshold):
            return False
        return True
    ###### ROBOT CONFIG VALIDATION ######

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
        # compute IK
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

        # # fix and validate
        # q_R_goal = self.util.wrap_to_pi(q_R_goal)
        # q_R_goal = self.util.clamp_to_limits(q_R_goal, self.robot.arm_lower_limits, self.robot.arm_upper_limits)

        return q_R_goal
    
    def get_best_valid_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples=100, time_out=60):
        q_H_trajs = []
        q_R_trajs = []
        q_H_goals = []
        q_R_goals = []
        count = 0
        start_time = time.time()
        while True:
            valid_grasp = False
            if count >= n_samples:
                break
            if time.time() - start_time >= time_out and count >= 1:  # timeout = 60 sec
                print(f'timeout.. score available configs ({count})')
                break
            self.reset_robot(self.robot_2, q_robot_2)
            q_H_goal, world_to_right_elbow = self.get_valid_q_H()
            q_H_traj, q_R_traj = self.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=q_robot)
            q_R_goal = q_R_traj[-1]
            valid_grasp = (self.validate_q_R(q_H=q_H_goal, q_R=q_R_goal, check_goal=True) 
                           and self.is_not_discontinuous(q_old=q_R_traj[0], q_new=q_R_traj[len(q_R_traj)//2])
                           and self.is_not_discontinuous(q_old=q_R_traj[len(q_R_traj)//2], q_new=q_R_traj[-1]))
            if valid_grasp:
                q_H_trajs.append(q_H_traj)
                q_R_trajs.append(q_R_traj)
                q_H_goals.append(q_H_goal)
                q_R_goals.append(q_R_goal)
                count += 1

        q_H_scores = []
        for q_H_goal, q_R_goal in zip(q_H_goals, q_R_goals):
            self.lock_human_joints(q_H_goal)
            self.lock_robot_arm_joints(self.robot, q_R_goal)
            score = self.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal, q_robot=q_R_goal)
            q_H_scores.append(score)

        # sort by scores
        combined = zip(q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals)  # zip
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)  # in descending order
        q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = zip(*sorted_combined)  # unzip

        q_H_scores = list(q_H_scores)
        q_H_trajs = list(q_H_trajs)
        q_R_trajs = list(q_R_trajs)
        q_H_goals = list(q_H_goals)
        q_R_goals = list(q_R_goals)

        # validate each waypoint
        for idx, (q_H_traj, q_R_traj) in enumerate(zip(q_H_trajs, q_R_trajs)):
            is_valid = True
            for q_H, q_R in zip(q_H_traj, q_R_traj):
                self.reset_human_arm(q_H)
                self.reset_robot(self.robot, q_R)
                is_valid = is_valid and self.validate_q_R(q_H, q_R)
            if is_valid:
                # print(idx)
                break

        # reset environment
        self.lock_human_joints(q_H_init)
        self.lock_robot_arm_joints(self.robot, q_robot)
        self.reset_robot(self.robot_2, q_robot_2)

        return q_H_scores[idx], q_H_trajs[idx], q_R_trajs[idx], q_H_goals[idx], q_R_goals[idx]

    def reset_base_pose(self, obj_id, base_pos, base_orn):
        self.bc.resetBasePositionAndOrientation(obj_id, posObj=base_pos, ornObj=base_orn)

    def get_new_human_robot_configs(self, q_H_init, q_robot, q_robot_2):
        valid_grasp = False
        while True:
            self.reset_robot(self.robot_2, q_robot_2)
            q_H_goal, world_to_right_elbow = self.get_valid_q_H()
            q_R_goal = self.get_q_R_from_elbow_pose(prev_q_R=q_robot)
            valid_grasp = self.validate_q_R(q_H_goal, q_R_goal, check_goal=True)
            if valid_grasp:
                break

        return q_H_goal, q_R_goal

    #### MODIFIED VERSIONS
    def get_valid_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples, time_out=60):
        q_H_trajs = []
        q_R_trajs = []
        q_H_goals = []
        q_R_goals = []
        count = 0
        start_time = time.time()
        while True:
            valid_grasp = False
            # print(count)
            if count >= n_samples:
                break
            if time.time() - start_time >= time_out:  # timeout = 60 sec
                print(f'timeout with {count} configs..')
                break
            self.reset_robot(self.robot_2, q_robot_2)
            q_H_goal, world_to_right_elbow = self.get_valid_q_H()
            q_H_traj, q_R_traj = self.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=q_robot)
            q_R_goal = q_R_traj[-1]
            valid_grasp = (self.validate_q_R(q_H=q_H_goal, q_R=q_R_goal, check_goal=True) 
                           and self.is_not_discontinuous(q_old=q_R_traj[0], q_new=q_R_traj[len(q_R_traj)//2])
                           and self.is_not_discontinuous(q_old=q_R_traj[len(q_R_traj)//2], q_new=q_R_traj[-1]))
            if valid_grasp:
                q_H_trajs.append(q_H_traj)
                q_R_trajs.append(q_R_traj)
                q_H_goals.append(q_H_goal)
                q_R_goals.append(q_R_goal)
                count += 1

        return q_H_trajs, q_R_trajs, q_H_goals, q_R_goals

    def get_valid_goal_configs_with_best_score(self, q_H_init, q_robot, q_robot_2,
                                               q_H_trajs, q_R_trajs, q_H_goals, q_R_goals):
        q_H_scores = []
        for q_H_goal, q_R_goal in zip(q_H_goals, q_R_goals):
            self.lock_human_joints(q_H_goal)
            self.lock_robot_arm_joints(self.robot, q_R_goal)
            score = self.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal, q_robot=q_R_goal)
            q_H_scores.append(score)

        # sort by scores
        combined = zip(q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals)  # zip
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)  # in descending order
        q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = zip(*sorted_combined)  # unzip

        q_H_scores = list(q_H_scores)
        q_H_trajs = list(q_H_trajs)
        q_R_trajs = list(q_R_trajs)
        q_H_goals = list(q_H_goals)
        q_R_goals = list(q_R_goals)

        # validate each waypoint
        for idx, (q_H_traj, q_R_traj) in enumerate(zip(q_H_trajs, q_R_trajs)):
            is_valid = True
            for q_H, q_R in zip(q_H_traj, q_R_traj):
                self.reset_human_arm(q_H)
                self.reset_robot(self.robot, q_R)
                is_valid = is_valid and self.validate_q_R(q_H, q_R)
            if is_valid:
                break

        # reset environment
        self.lock_human_joints(q_H_init)
        self.lock_robot_arm_joints(self.robot, q_robot)
        self.reset_robot(self.robot_2, q_robot_2)

        return q_H_scores[idx], q_H_trajs[idx], q_R_trajs[idx], q_H_goals[idx], q_R_goals[idx]

    ########### POINT CLOUD ###########
    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            if obstacle == self.bed_id:
                # half_extents = [0.5, 1.7, 0.2]
                point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle, client_id=self.bc._client))
            else:
                point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle, client_id=self.bc._client))
        return np.array(point_cloud)

    def get_robot_point_cloud(self, robot):
        robot_pcd = get_point_cloud_from_collision_shapes(robot.id, client_id=self.bc._client)
        upper_arm_link = self.bc.getLinkState(robot.id, 2)[:2]
        forearm_link = self.bc.getLinkState(robot.id, 3)[:2]
        upper_arm_pcd = generate_capsule_vertices(radius=0.04, height=0.3, position=upper_arm_link[0], 
                                                  orientation=upper_arm_link[1], client_id=self.bc._client)
        forearm_pcd = generate_capsule_vertices(radius=0.04, height=0.27, position=forearm_link[0], 
                                                orientation=forearm_link[1], client_id=self.bc._client)
        pcd = np.vstack((robot_pcd, upper_arm_pcd, forearm_pcd))
        return pcd

    def compute_env_pcd(self, robot, resolution=8):
        # get 'static' obstacle point cloud
        static_obstacles = [self.bed_id, self.cube_id, self.cube_2_id]
        static_obs_pcd = self.get_obstacle_point_cloud(static_obstacles)

        link_to_separate = [self.right_elbow, self.right_wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate, client_id=self.bc._client, resolution=resolution)
        robot_pcd = self.get_robot_point_cloud(robot)

        env_pcd = np.vstack((static_obs_pcd, robot_pcd, human_pcd))
        right_arm_pcd = np.array(separate_pcd)
        right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.right_shoulder, resolution=resolution, client_id=self.bc._client)

        return env_pcd, right_arm_pcd, right_shoulder_pcd

    def visualize_point_cloud(self, pcd):
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([pc_ply])
    ########### POINT CLOUD ###########

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


if __name__ == '__main__':
    wiping_env = WipingDemo(gui=True)
    wiping_env.reset()