from manip4care.envs.utils.config_utils import find_goal_configs
import numpy as np
import time
import math
import pybullet as p

# utils
from wiping_task.targets_util import TargetsUtil
from wiping_task.score_util import ScoreUtil
from utils.collision_utils import get_collision_fn
from utils.point_cloud_utils import *
from manip4care.envs.utils.trajectory_utils import (
    interpolate_trajectory,
    get_init_traj_from_q_H,
    get_q_R_from_elbow_pose,
    get_eef_pose,
    is_not_discontinuous,
)
from manip4care.envs.utils.distance_utils import (
    get_human_to_robot_dist,
    is_near_goal_W_space,
    get_bed_to_robot_dist,
    get_cube_to_robot_dist,
    get_robot_to_robot_2_dist,
)
from manip4care.envs.utils.config_utils import (
    validate_q_R,
    validate_q_robot_2,
    get_valid_q_H,
    get_new_human_robot_configs,
    find_goal_configs,
)

# environment
from envs.base_env import BaseEnv

class WipingEnv(BaseEnv):
    def __init__(self, gui=False, seated=False, wiping=True):
        super().__init__(gui=gui, seated=seated, wiping=wiping)
        self.targets_util = TargetsUtil(self.bc._client, self.util)
        self.score_util = ScoreUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.shoulder, self.elbow, self.human_arm,
                                            self.robot_w, self.tool,
                                            self.target_closer_to_eef, self.robot_w_in_collision)
        self.score_util.init_score_util(self.humanoid._humanoid, self.shoulder, self.elbow, self.human_controllable_joints,
                                        self.robot_m, self.robot_w, self.tool,
                                        self.target_closer_to_eef, self.robot_w_in_collision, self.robot_m_in_collision)

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
        super().create_world()

        self.targets_pos_on_upperarm = None
        self.targets_pos_on_forearm = None

        # initialize collision checker (robot 2)
        obstacles = [self.bed_id, self.humanoid._humanoid, self.robot_m.id, self.cube_m_id]
        self.robot_w_in_collision = get_collision_fn(self.robot_w.id, self.robot_w.arm_controllable_joints, obstacles=obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # initialize collision checker (robot)
        robot_obstacles = [self.bed_id, self.robot_w.id, self.cube_w_id, self.humanoid._humanoid]
        self.robot_m_in_collision = get_collision_fn(self.robot_m.id, self.robot_m.arm_controllable_joints, obstacles=robot_obstacles,
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_w.id, self.robot_w.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
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
            self.robot_w.move_gripper(0.04)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

        # initialize tool for wiping task
        world_to_eef = self.bc.getLinkState(self.robot_w.id, self.robot_w.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        world_to_tool = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14], [0,0,0,1]]
        eef_to_tool = self.bc.multiplyTransforms(positionA=eef_to_world[0], orientationA=eef_to_world[1], 
                                                 positionB=world_to_tool[0], orientationB=world_to_tool[1], physicsClientId=self.bc._client)
        self.eef_to_tool = eef_to_tool
        self.tool = self.bc.loadURDF("manip4care/envs/urdf/wiper.urdf", basePosition=world_to_tool[0], baseOrientation=world_to_tool[1], physicsClientId=self.bc._client)

        # disable collisions between the tool and robot
        for j in self.robot_w.arm_controllable_joints:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.robot_w.id, self.tool, j, tj, False, physicsClientId=self.bc._client)

        if not self.gui:
            self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)
        
    def attach_tool(self):
        # reset tool and attach it to eef
        world_to_eef = self.bc.getLinkState(self.robot_w.id, self.robot_w.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
        self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)

        # create constraint that keeps the tool in the gripper
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot_w.id,
                            parentLinkIndex=self.robot_w.eef_id,
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
        def compute_and_validate_ik_sol(target_pos_world, target_orn_world, target_alpha_to_eef, 
                                        pos_threshold, orn_threshold, q_R_prev):
            """Helper function to compute IK for robot_2 and return valid wiping joint states."""
            # compute desired world_to_eef (check if it can get closer to the target point)
            world_to_eef = self.bc.multiplyTransforms(target_pos_world, target_orn_world,
                                                      target_alpha_to_eef[0], target_alpha_to_eef[1], physicsClientId=self.bc._client)

            # set robot initial joint state
            q_robot_2 = self.bc.calculateInverseKinematics(self.robot_w.id, self.robot_w.eef_id, world_to_eef[0], world_to_eef[1],
                                                           lowerLimits=self.robot_w.arm_lower_limits, upperLimits=self.robot_w.arm_upper_limits, 
                                                           jointRanges=self.robot_w.arm_joint_ranges, restPoses=q_R_prev,
                                                           maxNumIterations=50, physicsClientId=self.bc._client)
            q_robot_2 = [q_robot_2[i] for i in range(len(self.robot_w.arm_controllable_joints))]

            for i, joint_id in enumerate(self.robot_w.arm_controllable_joints):
                self.bc.resetJointState(self.robot_w.id, joint_id, q_robot_2[i], physicsClientId=self.bc._client)

            # check if config is valid
            eef_pose = p.getLinkState(self.robot_w.id, self.robot_w.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
            pos_dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
            dot_product = np.abs(np.dot(world_to_eef[1], eef_pose[1]))
            orn_dist = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            if not self.robot_w_in_collision(q_robot_2) and pos_dist <= pos_threshold and orn_dist < orn_threshold:
                return q_robot_2
            else:
                return None
        
        robot_traj = []
        q_robot_previous = self.robot_w.arm_rest_poses

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

        self.robot_w.reset()

        return robot_traj
    
    def interpolate_trajectory(self, robot_traj, alpha=0.5):
        return interpolate_trajectory(robot_traj, alpha)

    def get_valid_q_H(self):
        return get_valid_q_H(self)

    def get_eef_pose(self, robot, current_joint_angles, target_joint_angles):
        return get_eef_pose(self, robot, current_joint_angles, target_joint_angles)

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
    
    
    ###### ROBOT CONFIG VALIDATION ######
    def get_human_to_robot_dist(self, robot, q_H, q_robot):
        return get_human_to_robot_dist(self, robot, q_H, q_robot)
    
    def get_bed_to_robot_dist(self, robot, q_robot):
        return get_bed_to_robot_dist(self, robot, q_robot)
    
    def get_cube_to_robot_dist(self, robot, q_robot, cube_id):
        return get_cube_to_robot_dist(self, robot, q_robot, cube_id)
    
    def get_robot_to_robot_2_dist(self, q_robot, q_robot_2):
        return get_robot_to_robot_2_dist(self, q_robot, q_robot_2)
    
    def validate_q_R(self, q_H, q_R, check_goal=False):
        return validate_q_R(self, q_H, q_R, check_goal=check_goal)
    
    def validate_q_robot_2(self, q_H, q_robot, q_robot_2):
        return validate_q_robot_2(self, q_H, q_robot, q_robot_2)
    
    def is_not_discontinuous(self, q_old, q_new, angle_threshold = math.pi/2):
        return is_not_discontinuous(q_old, q_new, angle_threshold)
    ###### ROBOT CONFIG VALIDATION ######

    def get_init_traj_from_q_H(self, q_H_init, q_H_goal, q_R_init):
        return get_init_traj_from_q_H(self, q_H_init, q_H_goal, q_R_init)

    def get_q_R_from_elbow_pose(self, prev_q_R):
        return get_q_R_from_elbow_pose(self, prev_q_R)
    
    def get_best_valid_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples=100, time_out=60):
        return find_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples=n_samples, time_out=time_out)

    def get_new_human_robot_configs(self, q_H_init, q_robot, q_robot_2):
        return get_new_human_robot_configs(self, q_H_init, q_robot, q_robot_2)

    #### MODIFIED VERSIONS
    def get_valid_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples, time_out=60):
        return find_goal_configs(self, q_H_init, q_robot, q_robot_2, n_samples=n_samples, time_out=time_out, return_all=True)

    def get_valid_goal_configs_with_best_score(self, q_H_init, q_robot, q_robot_2, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals):
        return find_goal_configs(self, q_H_init, q_robot, q_robot_2, existing_trajs=(q_H_trajs, q_R_trajs, q_H_goals, q_R_goals))

    ########### POINT CLOUD ###########
    def compute_env_pcd(self, robot, resolution=8):
        # get 'static' obstacle point cloud
        static_obstacles = [self.bed_id, self.cube_m_id, self.cube_w_id]
        static_obs_pcd = self.get_obstacle_point_cloud(static_obstacles)

        link_to_separate = [self.elbow, self.wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate, client_id=self.bc._client, resolution=resolution)
        robot_pcd = self.get_robot_point_cloud(robot)

        env_pcd = np.vstack((static_obs_pcd, robot_pcd, human_pcd))
        arm_pcd = np.array(separate_pcd)
        shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.shoulder, resolution=resolution, client_id=self.bc._client)

        return env_pcd, arm_pcd, shoulder_pcd
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
    wiping_env = WipingEnv(gui=True)
    wiping_env.reset()