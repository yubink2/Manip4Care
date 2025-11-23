# general imports
import os
import pybullet as p
import math
import sys
import time
import numpy as np
import argparse

# environments
from manip4care.envs.grasp_generation_env import GraspEnv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable GUI. (Default: False)"
    )

    parser.add_argument(
        "--seated",
        action="store_true",
        default=False,
        help="Enable human seated environment. (Default: False)"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    grasp_env = GraspEnv(gui=args.gui, seated=args.seated, wiping=False)
    grasp_env.reset()

    # initial configurations
    power_off_pose = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    q_robot_init = power_off_pose
    q_robot_2_init = power_off_pose
    q_H_init = grasp_env.human_rest_poses

    if args.seated:
        base_pose = ((0.3, -0.6, 0.65), grasp_env.bc.getQuaternionFromEuler((0, 0, 0)))
        cube_base_pose = ((0.3, -0.6, 0.65-0.15), (0, 0, 0, 1))
    else:
        base_pose = ((0.65, -0.1, 0.3), grasp_env.bc.getQuaternionFromEuler((0, 0, 3.14)))
        cube_base_pose = ((0.65, -0.1, 0.3-0.15), (0, 0, 0, 1))
    grasp_env.reset_base_pose(grasp_env.robot_m.id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.cube_m_id, cube_base_pose[0], cube_base_pose[1])

    (q_R_grasp_samples, grasp_pose_samples, world_to_eef_goals, 
        best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal) = grasp_env.generate_grasps(q_H_init)
    print(f'best_q_R_grasp = {best_q_R_grasp}')
    print(f'best_world_to_grasp = {best_world_to_grasp}')
    print(f'best_world_to_eef_goal = {best_world_to_eef_goal}')

    grasp_env.reset_robot(grasp_env.robot_m, best_q_R_grasp)

    grasp_env.bc.disconnect()
