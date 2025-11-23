# general imports
import os
import pybullet as p
import math
import sys
import time
import numpy as np
import argparse

# environments
from manip4care.envs.manipulation_env import ManipulationEnv
from manip4care.envs.wiping_env import WipingEnv

# simulation loop
from manip4care.sim_loops import arm_manipulation_loop


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable GUI. (Default: False)"
    )

    parser.add_argument(
        "--group",
        type= str,
        default="None",
        help="Specify the group for shoulder range reduction. (Options: A, B, C, D. Default: None)."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    manip_env = ManipulationEnv(gui=args.gui, wiping=False)
    wiping_env = WipingEnv()
    manip_env.reset()
    wiping_env.reset()

    # initial configurations
    power_off_pose = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    q_robot_init = power_off_pose
    q_robot_2_init = power_off_pose
    q_H_init = manip_env.human_rest_poses

    off_base_pose = ((100,100,100), (0,0,0,1))
    wiping_env.reset_base_pose(wiping_env.robot_w.id, off_base_pose[0], off_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.cube_w_id, off_base_pose[0], off_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.tool, off_base_pose[0], off_base_pose[1])

    # set lower and upper joint limits
    manip_env.set_arm_joint_range(shoulder_reduction_group=args.group)
    wiping_env.set_arm_joint_range(shoulder_reduction_group=args.group)

    ### 1) save example grasp parameters
    best_q_R_grasp = [-2.2567504, -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
    best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [0.84583597, -0.13011431, -0.49919509, 0.13577936]]
    best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))

    manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
    (elbow_to_cp, cp_to_elbow, eef_to_cp, cp_to_eef,
    elbow_joint_to_cp, cp_to_elbow_joint,
    wrist_joint_to_cp, cp_to_wrist_joint) = manip_env.get_grasp_parameters()
    wiping_env.set_grasp_parameters(elbow_to_cp, cp_to_elbow, eef_to_cp, cp_to_eef,
                                    elbow_joint_to_cp, cp_to_elbow_joint,
                                    wrist_joint_to_cp, cp_to_wrist_joint)
    valid_grasp = wiping_env.validate_q_R(q_H_init, best_q_R_grasp, check_goal=True)
    if not valid_grasp:
        raise ValueError("grasp not valid!")

    ### 2) example init+goal configs
    q_H_init = manip_env.human_rest_poses
    wiping_env.reset_human_arm(q_H_init)
    q_R_init = wiping_env.get_q_R_from_elbow_pose(prev_q_R=best_q_R_grasp)
    manip_env.reset_human_arm(q_H_init)
    manip_env.reset_robot(robot=manip_env.robot_m, q_robot=q_R_init)

    # generate new goal configs
    q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = wiping_env.get_valid_goal_configs(q_H_init=q_H_init,
                                                                                    q_robot=q_R_init,
                                                                                    q_robot_2=q_robot_2_init,
                                                                                    n_samples=1,
                                                                                    time_out=100)
    q_H_traj, q_R_traj, q_H_goal, q_R_goal = q_H_trajs[0], q_R_trajs[0], q_H_goals[0], q_R_goals[0]

    ### 3) manipulation
    # save goal parameters
    wiping_env.reset_robot(wiping_env.robot_m, q_R_goal)
    world_to_eef_goal = wiping_env.bc.getLinkState(wiping_env.robot_m.id, wiping_env.robot_m.eef_id)[:2]

    # arm manipulaton
    manip_env.detach_human_arm_from_eef()
    manip_env.reset_human_arm(q_H_init)
    manip_env.reset_robot(manip_env.robot_m, q_R_init)

    (arm_manip_planning_times, arm_manip_following_times, 
    arm_manip_loop_failed, arm_manip_total_dist, 
    arm_manip_success_times, arm_manip_human_joint_angles) = arm_manipulation_loop(manip_env=manip_env,
                                                                                    q_robot_2=q_robot_2_init,
                                                                                    q_robot_init=q_R_init,
                                                                                    q_robot_goal=q_R_goal,
                                                                                    q_H_init=q_H_init,
                                                                                    world_to_eef_goal=world_to_eef_goal,
                                                                                    q_R_init_traj=q_R_traj,
                                                                                    manip_demo=True)

    manip_env.bc.disconnect()
    wiping_env.bc.disconnect()
