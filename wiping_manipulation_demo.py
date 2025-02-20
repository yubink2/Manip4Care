# ur5, pybullet
import os
import pybullet as p
import math
import sys

import time
import numpy as np
import argparse

# environments
from envs.manipulation_env import ManipulationDemo
from envs.wiping_env import WipingDemo
from envs.grasp_generation_env import GraspDemo

# arm config NN model
import torch
import numpy as np
from arm_config_NN_train import CombinedModel, PointNetEncoder, ArmConfigPredictor, normalize_point_cloud
from arm_config_dataset_generation import label_pcd

# video recording
import cv2


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


################################################
#                                              #
#               VIDEO RECORDING                #
#                                              #
################################################
def capture_frame(bc, frame_dir, frame_count, width=1280, height=1024):
    """Capture a frame from the PyBullet simulation."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                      distance=2,
                                                      yaw=0,
                                                      pitch=-30,
                                                      roll=0,
                                                      upAxisIndex=2,
                                                      physicsClientId=bc._client)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height,
                                               nearVal=0.1, farVal=100.0, physicsClientId=bc._client)

    (_, _, px, _, _) = p.getCameraImage(width=width, height=height,
                                        viewMatrix=view_matrix,
                                        projectionMatrix=proj_matrix,
                                        physicsClientId=bc._client)
    img = np.reshape(px, (height, width, 4))  # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(f"{frame_dir}/frame_{frame_count:04d}.png", img)


def save_video_from_frames(frame_dir, output_file, fps=20):
    """Convert captured frames into a video."""
    img_array = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(frame_dir, filename))
            img_array.append(img)

    height, width, _ = img_array[0].shape
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in img_array:
        out.write(img)
    out.release()

################################################
#                                              #
#               HELPER FUNCTIONS               #
#                                              #
################################################
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI. (Default: False)"
    )

    parser.add_argument(
        "--grasp",
        action="store_true",
        help="Generate grasp only. (Default: False)"
    )

    parser.add_argument(
        "--use-follower",
        default=True,
        action="store_true",
        help="Enable trajectory follower for moving the wiping robot. (Default: True)"
    )

    parser.add_argument(
        "--no-use-follower",
        action="store_false",
        dest="use_follower",
        help="Disable trajectory follower."
    )

    parser.add_argument(
        "--use-predictor",
        default=True,
        action="store_true",
        help="Use arm config predictor NN model to get next human arm. (Default=True)"
    )

    parser.add_argument(
        "--no-use-predictor",
        action="store_false",
        dest="use_predictor",
        help="Use random config generator to get next human arm."
    )

    parser.add_argument(
        "--trials",
        type= int,
        default=1,
        help="Specify number of trials to profile the results."
    )

    parser.add_argument(
        "--file",
        type= str,
        default="results",
        help="Specify name for the output file."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print messages for debugging. (Default=False)"
    )

    parser.add_argument(
        "--wiping-threshold",
        type=float,
        default=0.8,
        help="Specify number for the wiping threshold."
    )

    parser.add_argument(
        "--iter",
        type= int,
        default=50,
        help="Specify number for the simulation iteration number."
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a video. (Default=False)"
    )

    args = parser.parse_args()
    return args

def profile_function(func_name, func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def reset_base_poses(manip_env, wiping_env, grasp_env):
    base_pose = manip_env.get_obj_base_pose("bed")
    wiping_env.reset_base_pose(wiping_env.bed_id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.bed_id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("robot")
    wiping_env.reset_base_pose(wiping_env.robot.id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.robot.id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("robot_2")
    wiping_env.reset_base_pose(wiping_env.robot_2.id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.robot_2.id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("humanoid")
    wiping_env.reset_base_pose(wiping_env.humanoid._humanoid, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.humanoid._humanoid, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("cube")
    wiping_env.reset_base_pose(wiping_env.cube_id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.cube_id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("cube_2")
    wiping_env.reset_base_pose(wiping_env.cube_2_id, base_pose[0], base_pose[1])
    grasp_env.reset_base_pose(grasp_env.cube_2_id, base_pose[0], base_pose[1])


################################################
#                                              #
#               FOR PREDICTOR NN               #
#                                              #
################################################
def normalize_data(q_H_init):
    q_H_min = np.array([-3.141368118925281, -0.248997453133789, -2.6643015908664056, 0.0])
    q_H_max = np.array([3.1415394736319917, 1.2392816988875348, -1.3229245882839409, 2.541304])
    q_H_init_normalized = (q_H_init - q_H_min) / (q_H_max - q_H_min)
    return q_H_init_normalized

def unnormalize_data(q_H_goal_normalized):
    q_H_min = np.array([-3.141368118925281, -0.248997453133789, -2.6643015908664056, 0.0])
    q_H_max = np.array([3.1415394736319917, 1.2392816988875348, -1.3229245882839409, 2.541304])
    q_H_goal = q_H_goal_normalized * (q_H_max - q_H_min) + q_H_min
    return q_H_goal

def load_combined_model(model_path, device='cuda'):
    encoder = PointNetEncoder(latent_dim=512)
    predictor = ArmConfigPredictor(latent_dim=512, input_dim=4, output_dim=4)
    model = CombinedModel(encoder, predictor)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Combined model loaded from {model_path}")
    return model

def predict_q_H_goal(model, q_H_init, labeled_pcd, device='cuda'):
    # Normalize q_H_init
    q_H_init_norm = normalize_data(q_H_init)
    q_H_init_tensor = torch.tensor(q_H_init_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4)

    # Normalize and prepare point cloud
    normalized_pcd = normalize_point_cloud(np.array(labeled_pcd))  # (N,4)
    points = normalized_pcd.T  # (4, N)
    pcd_init_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, N)

    # Predict q_H_goal
    with torch.no_grad():
        q_H_goal_tensor = model(q_H_init_tensor, pcd_init_tensor)  # (1,4)
    q_H_goal_normalized = q_H_goal_tensor.cpu().numpy().squeeze()
    q_H_goal = unnormalize_data(q_H_goal_normalized)

    return q_H_goal

def get_best_noise_sample(q_H_init, q_H_goal, q_R_init, max_noise_trials=1000):
    # add noise around q_H_goal_predicted to generate candidates and pick best
    noise_std = 0.3
    N = 30
    best_score = -np.inf
    best_q_H_goal = None
    best_q_H_traj = None
    best_q_R_goal = None
    best_q_R_traj = None

    # Evaluate the base prediction
    q_H_goal = np.clip(q_H_goal, wiping_env.human_arm_lower_limits, wiping_env.human_arm_upper_limits)
    wiping_env.reset_human_arm(q_H_goal)
    wiping_env.targets_util.update_targets()
    q_H_traj, q_R_traj = wiping_env.get_init_traj_from_q_H(q_H_init=q_H_init,
                                                           q_H_goal=q_H_goal,
                                                           q_R_init=q_R_init)
    q_R_goal = q_R_traj[-1]
    wiping_env.reset_robot(wiping_env.robot, q_R_goal)
    
    valid_grasp = (wiping_env.validate_q_R(q_H=q_H_goal, q_R=q_R_goal, check_goal=True) 
                    and wiping_env.is_not_discontinuous(q_old=q_R_traj[0], q_new=q_R_traj[len(q_R_traj)//2])
                    and wiping_env.is_not_discontinuous(q_old=q_R_traj[len(q_R_traj)//2], q_new=q_R_traj[-1]))

    predicted_score = wiping_env.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal, q_robot=q_R_goal)
    print(f'predicted score: {predicted_score}')
    if valid_grasp and predicted_score > best_score:
        best_score = predicted_score
        best_q_H_goal = q_H_goal.copy()
        best_q_H_traj = q_H_traj
        best_q_R_goal = q_R_goal
        best_q_R_traj = q_R_traj

    noise_trial = 0
    while noise_trial < max_noise_trials:
        noise_trial += N
        # Generate and evaluate noisy samples
        for _ in range(N):
            q_H_goal_noisy = q_H_goal + np.random.normal(0, noise_std, size=q_H_goal.shape)
            q_H_goal_noisy = np.clip(q_H_goal_noisy, wiping_env.human_arm_lower_limits, wiping_env.human_arm_upper_limits)

            wiping_env.reset_human_arm(q_H_goal_noisy)
            wiping_env.targets_util.update_targets()
            q_H_traj, q_R_traj = wiping_env.get_init_traj_from_q_H(q_H_init=q_H_init,
                                                                    q_H_goal=q_H_goal_noisy,
                                                                    q_R_init=q_R_init)
            q_R_goal = q_R_traj[-1]
            wiping_env.reset_robot(wiping_env.robot, q_R_goal)

            valid_grasp = (wiping_env.validate_q_R(q_H=q_H_goal_noisy, q_R=q_R_goal, check_goal=True) 
                           and wiping_env.is_not_discontinuous(q_old=q_R_traj[0], q_new=q_R_traj[len(q_R_traj)//2])
                           and wiping_env.is_not_discontinuous(q_old=q_R_traj[len(q_R_traj)//2], q_new=q_R_traj[-1]))
            
            candidate_score = wiping_env.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal_noisy, q_robot=q_R_goal)
            
            if valid_grasp and candidate_score > best_score:
                best_score = candidate_score
                best_q_H_goal = q_H_goal_noisy.copy()
                best_q_H_traj = q_H_traj
                best_q_R_goal = q_R_goal
                best_q_R_traj = q_R_traj

        if best_q_H_goal is not None:
            break

    # Use best_q_H_goal found
    print(f"Best score: {best_score}, trial: {noise_trial}")

    if best_q_H_goal is None:
        return candidate_score, q_H_goal_noisy, q_H_traj, q_R_goal, q_R_traj

    return best_score, best_q_H_goal, best_q_H_traj, best_q_R_goal, best_q_R_traj

def compute_travel_distance(eef_pos_list):
    # iterate over the trajectory to calculate the distance
    total_distance = 0.0
    for i in range(len(eef_pos_list) - 1):
        eef_pos_current = np.array(eef_pos_list[i])
        eef_pos_next = np.array(eef_pos_list[i + 1])
        distance = np.linalg.norm(eef_pos_next - eef_pos_current)
        total_distance += distance

    return total_distance

################################################
#                                              #
#               SIMULATION LOOPS               #
#                                              #
################################################
def wiping_loop(wiping_env, manip_env, q_H, total_targets_cleared, q_robot, q_robot_2_init, use_follower=True, gui=False):
    wiping_planning_times = []
    wiping_dist = 0
    wiping_eef_pos_list = []
    wiping_move_success_times = []

    # initialize environments
    current_joint_angles = q_robot_2_init
    manip_env.lock_human_joints(q_H)
    manip_env.targets_util.update_targets()
    wiping_env.lock_robot_arm_joints(wiping_env.robot, q_robot)

    arms = ['upperarm', 'forearm']
    targets_cleared = 0
    wiping_robot_failed = 0
    wiping_robot_count = 0
    for arm in arms:
        for _ in range(5):
            # compute feasible targets & wiping trajectory
            feasible_targets_found = wiping_env.reset_wiping_setup(q_H, arm)
            if not feasible_targets_found:
                print(f'{arm} feasible targets not found!')
                continue
            
            robot_traj = wiping_env.compute_feasible_targets_robot_traj()
            if len(robot_traj) <= 1:
                print(f'{arm} valid trajectory not found!')
                continue

            # validate start and end of the wiping traj
            if not wiping_env.validate_q_robot_2(q_H, q_robot, robot_traj[0]) or not wiping_env.validate_q_robot_2(q_H, q_robot, robot_traj[-1]):
                print(f'start or end of the wiping traj not valid!')
                continue

            if len(robot_traj) <= 5:
                robot_traj.extend(robot_traj[::-1])

            robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.5)
            robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.5)

            # compute feasible targets parameters
            feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side = wiping_env.targets_util.get_feasible_targets_lists()
            feasible_targets = manip_env.targets_util.get_feasible_targets_given_indices(feasible_targets_indices, arm_side)
            manip_env.targets_util.set_feasible_targets_lists(feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side)
            manip_env.targets_util.mark_feasible_targets()

            # move robot_2 to wiping initial config
            eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_2, 
                                                    current_joint_angles=current_joint_angles, target_joint_angles=robot_traj[0])
            if use_follower:
                (move_robot_failed, move_robot_total_dist, 
                 move_robot_total_time, move_robot_planning_times) = move_robot_loop(manip_env, 
                                                                                     robot=manip_env.robot_2, other_robot=manip_env.robot, 
                                                                                     q_robot_init=current_joint_angles, q_robot_goal=robot_traj[0], 
                                                                                     world_to_robot_eef_goal=eef_goal_pose,
                                                                                     q_other_robot=q_robot, q_H=q_H)
                wiping_robot_count += 1
                wiping_dist += move_robot_total_dist
                wiping_planning_times += move_robot_planning_times
                if move_robot_failed > 0:
                    wiping_robot_failed += 1
                    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)
                    manip_env.targets_util.unmark_feasible_targets()
                    print('wiping move robot failed!')
                    break
                else:
                    wiping_move_success_times.append(move_robot_total_time)
            else:
                traj, move_robot_planning_times = move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
                                                                  q_robot_init=current_joint_angles, q_robot_goal=robot_traj[0], 
                                                                  world_to_robot_eef_goal=eef_goal_pose,
                                                                  q_other_robot=q_robot, q_H=q_H, 
                                                                  early_terminate=True)
                wiping_robot_count += 1
                wiping_planning_times += move_robot_planning_times
                is_robot_2_in_collision = False
                for q_R in traj:
                    manip_env.reset_robot(manip_env.robot_2, q_R)
                    manip_env.reset_robot(manip_env.robot, q_robot)
                    wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_2.id, manip_env.robot_2.eef_id)[0]))
                    if manip_env.robot_2_in_collision(q_R):
                        is_robot_2_in_collision = True
                        break
                    if gui:
                        time.sleep(0.05)
                if is_robot_2_in_collision:
                    wiping_robot_failed += 1
                    break
                manip_env.attach_tool()  ####

            # execute wiping trajectory
            for q_R in robot_traj:
                for _ in range(50):
                    manip_env.move_robot(manip_env.robot_2, q_R)
                    manip_env.move_robot(manip_env.robot, q_robot)
                    manip_env.bc.stepSimulation()
                if gui:
                    time.sleep(0.05)
                new_target, indices_to_delete = manip_env.targets_util.get_new_contact_points(targeted_arm=arm)
                manip_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)
                wiping_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)

                targets_cleared += new_target
                total_targets_cleared += new_target
                wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_2.id, manip_env.robot_2.eef_id)[0]))

            manip_env.targets_util.remove_targets()
            manip_env.targets_util.unmark_feasible_targets()
            manip_env.targets_util.update_targets()
            if not use_follower:
                manip_env.detach_tool()  ####

            wiping_env.targets_util.remove_targets()
            wiping_env.targets_util.update_targets()

            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)

        # if wiping robot failed, terminate
        if wiping_robot_failed > 0:
            break

    # move robot_2 back to rest poses
    eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_2, 
                                            current_joint_angles=current_joint_angles, target_joint_angles=q_robot_2_init)
    if use_follower:
        (move_robot_failed, move_robot_total_dist, 
         move_robot_total_time, move_robot_planning_times) = move_robot_loop(manip_env, 
                                                                             robot=manip_env.robot_2, other_robot=manip_env.robot, 
                                                                             q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, 
                                                                             world_to_robot_eef_goal=eef_goal_pose,
                                                                             q_other_robot=q_robot, q_H=q_H)
        wiping_robot_count += 1
        wiping_dist += move_robot_total_dist
        wiping_planning_times += move_robot_planning_times

        if move_robot_failed > 0:
            wiping_robot_failed += 1
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)
            manip_env.detach_tool()
            manip_env.reset_robot(wiping_env.robot_2, q_robot_2_init)
            manip_env.attach_tool()
            print('wiping move robot failed!')
        else:
            wiping_move_success_times.append(move_robot_total_time)
    else:
        traj, move_robot_planning_times = move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
                                                          q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, 
                                                          world_to_robot_eef_goal=eef_goal_pose,
                                                          q_other_robot=q_robot, q_H=q_H, 
                                                          early_terminate=True)
        wiping_robot_count += 1
        wiping_planning_times += move_robot_planning_times
        is_robot_2_in_collision = False
        for q_R in traj:
            manip_env.reset_robot(manip_env.robot_2, q_R)
            manip_env.reset_robot(manip_env.robot, q_robot)
            wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_2.id, manip_env.robot_2.eef_id)[0]))
            if manip_env.robot_2_in_collision(q_R):
                is_robot_2_in_collision = True
                break
            if gui:
                time.sleep(0.05)
        if is_robot_2_in_collision:
            wiping_robot_failed += 1
            manip_env.reset_robot(manip_env.robot_2, q_robot_2_init)
        manip_env.detach_tool()  ####

    wiping_env.reset_robot(wiping_env.robot_2, q_robot_2_init)
    manip_env.unlock_human_joints(q_H)

    wiping_dist += compute_travel_distance(wiping_eef_pos_list)
    print('wiping loop is done')
    return targets_cleared, total_targets_cleared, wiping_robot_failed, wiping_robot_count, wiping_dist, wiping_move_success_times, wiping_planning_times

def arm_manipulation_loop(manip_env, 
                          q_robot_2, 
                          q_robot_init, q_robot_goal, 
                          q_H_init, 
                          world_to_eef_goal, q_R_init_traj, 
                          manip_demo=False, 
                          record=False, frame_dir="frames/", frame_count=0):
    arm_manip_planning_times = []
    arm_manip_following_times = []
    arm_manip_eef_pos_list = []
    if manip_demo:
        arm_manip_human_joint_angles = []

    manip_env.lock_robot_arm_joints(manip_env.robot_2, q_robot_2)  #####

    # Step 0: instantiate a new motion planning problem
    trajectory_planner = manip_env.init_traj_planner(manip_env.world_to_robot_base, clamp_by_human=True,
                                                     q_H_init=q_H_init, q_R_init=q_robot_init)
    trajectory_follower = manip_env.init_traj_follower(manip_env.world_to_robot_base)
    
    # Step 1: move robot to grasping pose
    manip_env.reset_robot(manip_env.robot_2, q_robot_2)
    manip_env.reset_robot(manip_env.robot, q_robot_init)
    manip_env.reset_human_arm(q_H_init)
    manip_env.targets_util.update_targets()

    # Step 2: attach human arm to eef
    env_pcd, right_arm_pcd, _ = manip_env.compute_env_pcd(robot=manip_env.robot_2, 
                                                          add_bed_padding=False, add_chest_padding=False)
    T_eef_to_object, T_object_to_world = manip_env.attach_human_arm_to_eef(attach_to_gripper=True, 
                                                                           right_arm_pcd=right_arm_pcd, trajectory_planner=trajectory_planner)

    # Step 3: trajectory after grasping
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=True, init_traj=q_R_init_traj)
    planning_time_start = time.time()
    traj = manip_env.get_mppi_trajectory(trajectory_planner, q_robot_init)
    previous_update_time = time.time()
    update_second = 5  # sec
    arm_manip_planning_times.append(previous_update_time-planning_time_start)

    # Step 4: initialize trajectory planner & follower
    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_trajectory(traj)
    trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=right_arm_pcd,
                                          T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                          T_world_to_human_base=manip_env.T_world_to_human_base, T_right_elbow_joint_to_cp=manip_env.T_right_elbow_joint_to_cp,
                                          human_arm_lower_limits=manip_env.human_arm_lower_limits, human_arm_upper_limits=manip_env.human_arm_upper_limits)
    
    current_joint_angles = q_robot_init
    current_human_joint_angles = q_H_init
    world_to_eef = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    trajectory_follower._init_H_clamping(manip_env.eef_to_cp, manip_env.right_elbow_joint_to_cp, 
                                         manip_env.robot_base_pose, manip_env.human_base_pose,
                                         manip_env.human_arm_lower_limits, manip_env.human_arm_upper_limits, manip_env.human_controllable_joints,
                                         human_rest_poses=q_H_init, robot_rest_poses=q_robot_init, seated=manip_env.seated)
    trajectory_follower.set_collision_threshold(collision_threshold=0.01)

    # Step 5: simulation loop
    arm_manip_start = time.time()
    arm_manip_time_threshold = 30  # sec
    arm_manip_loop_failed = 0
    arm_manip_success_times = 0
    while True:
        # save to total list
        arm_manip_eef_pos_list.append(list(world_to_eef[0]))

        # if near goal, execute rest of trajectory and end simulation loop
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_eef_goal, threshold=0.05):
            for _ in range(300):
                manip_env.move_robot(manip_env.robot, q_robot_goal)
                manip_env.bc.stepSimulation()
            manip_env.targets_util.update_targets()
            
            current_time = time.time()
            arm_manip_success_times = current_time - arm_manip_start
            break

        # get position command
        following_time_start = time.time()
        next_joint_angles = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles, time_step=0.05)
        current_time = time.time()
        arm_manip_following_times.append(current_time-following_time_start)

        # if exceed time threshold, end simulation loop
        if current_time-arm_manip_start > arm_manip_time_threshold:
            arm_manip_loop_failed = 1
            print('arm manip failed!')
            break

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            planning_time_start = time.time()
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            previous_update_time = time.time()
            trajectory_follower.update_trajectory(traj)
            arm_manip_planning_times.append(previous_update_time-planning_time_start)

        # move robot
        else:
            for _ in range(300):
                manip_env.move_robot(manip_env.robot, next_joint_angles)
                manip_env.bc.stepSimulation()
            manip_env.targets_util.update_targets()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
            current_human_joint_angles = manip_env.get_human_joint_angles()
            world_to_eef = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

            if manip_demo:
                arm_manip_human_joint_angles.append(current_human_joint_angles)

    # Step 6: reinforce the grasp
    manip_env.detach_human_arm_from_eef()
    world_to_right_elbow = manip_env.bc.getLinkState(manip_env.humanoid._humanoid, manip_env.right_elbow)[:2]
    world_to_cp = manip_env.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                  manip_env.right_elbow_to_cp[0], manip_env.right_elbow_to_cp[1])
    world_to_eef = manip_env.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                   manip_env.cp_to_eef[0], manip_env.cp_to_eef[1])
    q_R_goal = manip_env.bc.calculateInverseKinematics(manip_env.robot.id, manip_env.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                    manip_env.robot.arm_lower_limits, manip_env.robot.arm_upper_limits, manip_env.robot.arm_joint_ranges, manip_env.robot.arm_rest_poses,
                                                    maxNumIterations=50)
    q_R_goal = [q_R_goal[i] for i in range(len(manip_env.robot.arm_controllable_joints))]
    manip_env.reset_robot(manip_env.robot, q_R_goal)
    manip_env.attach_human_arm_to_eef()

    manip_env.unlock_robot_arm_joints(manip_env.robot_2, q_robot_2)  #####
    arm_manip_total_dist = compute_travel_distance(arm_manip_eef_pos_list)
    print('arm manipulation loop is done')

    # delete trajectory planner & follower after done
    del trajectory_planner
    del trajectory_follower
    del traj

    if manip_demo:
        return (arm_manip_planning_times, arm_manip_following_times, 
                arm_manip_loop_failed, arm_manip_total_dist, arm_manip_success_times, arm_manip_human_joint_angles)

    return arm_manip_planning_times, arm_manip_following_times, arm_manip_loop_failed, arm_manip_total_dist, arm_manip_success_times

def move_robot_loop(manip_env, robot, other_robot, q_robot_init, q_robot_goal, world_to_robot_eef_goal, q_other_robot, q_H, early_terminate=False):
    move_robot_planning_times = []
    move_robot_eef_pos_list = []
    move_robot_total_traj = []

    # Step 0: instantiate a new motion planning problem
    if robot == manip_env.robot:
        world_to_robot_base = manip_env.world_to_robot_base
    elif robot == manip_env.robot_2:
        world_to_robot_base = manip_env.world_to_robot_2_base
    else:
        raise ValueError('invalid robot!')
    
    trajectory_planner = manip_env.init_traj_planner(world_to_robot_base, clamp_by_human=False, q_H_init=None, q_R_init=None)
    trajectory_follower = manip_env.init_traj_follower(world_to_robot_base)
    
    # Step 1: initialize trajectory planner & follower
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=False, init_traj=[])
    env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(other_robot, resolution=12, 
                                                                           add_bed_padding=False, add_chest_padding=False,
                                                                           exclude_robot_fingers=True)
    env_pcd = np.vstack((env_pcd, right_arm_pcd, right_shoulder_pcd))

    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    current_joint_angles = q_robot_init
    world_to_eef = manip_env.bc.getLinkState(robot.id, robot.eef_id)[:2]

    # robot_2 (wiping): attach tool to planner and follower
    if robot == manip_env.robot_2:
        tool_pcd = manip_env.compute_obj_pcd(manip_env.tool)
        T_eef_to_object, T_object_to_world = manip_env.attach_tool(attach_to_gripper=True, 
                                                                   tool_pcd=tool_pcd, trajectory_planner=trajectory_planner)
    
    # Step 2: compute trajectory
    planning_time_start = time.time()
    traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles=q_robot_init)
    move_robot_planning_times.append(time.time()-planning_time_start)
    trajectory_follower.update_trajectory(traj)
    previous_update_time = time.time()
    update_second = 3  # sec

    # robot_2 (wiping): attach tool to planner and follower
    if robot == manip_env.robot_2:
        trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=tool_pcd,
                                              T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)
        trajectory_follower.set_collision_threshold(collision_threshold=0.01)

    if early_terminate:
        return traj, move_robot_planning_times

    # Step 3: simulation loop
    move_robot_start = time.time()
    move_robot_time_threshold = 60  # sec
    move_robot_failed = 0
    move_robot_total_time = 0
    while True:
        # save to total list
        move_robot_eef_pos_list.append(list(world_to_eef[0]))

        # if near goal, execute rest of trajectory and terminate the simulation loop
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_robot_eef_goal, threshold=0.1):
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            for q_R in traj:
                move_robot_total_traj.append(list(q_R))
                for _ in range(300):
                    manip_env.move_robot(robot, q_R)
                    manip_env.move_robot(other_robot, q_other_robot)
                    manip_env.bc.stepSimulation()
            current_time = time.time()
            move_robot_total_time = current_time - move_robot_start
            break

        # get position command
        next_joint_angles = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles=[], time_step=0.05)
        current_time = time.time()

        # if exceed time threshold, end simulation loop
        if current_time-move_robot_start > move_robot_time_threshold:
            move_robot_failed = 1
            break

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            planning_time_start = time.time()
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            move_robot_planning_times.append(time.time()-planning_time_start)
            trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # move robot
        else:
            for _ in range(300):
                manip_env.move_robot(robot, next_joint_angles)
                manip_env.move_robot(other_robot, q_other_robot)
                manip_env.bc.stepSimulation()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(robot)
            world_to_eef = manip_env.bc.getLinkState(robot.id, robot.eef_id)[:2]

    # delete trajectory planner & follower after done
    del trajectory_planner
    del trajectory_follower
    del traj

    move_robot_total_dist = compute_travel_distance(move_robot_eef_pos_list)
    print('move robot loop is done')
    return move_robot_failed, move_robot_total_dist, move_robot_total_time, move_robot_planning_times

################################################
#                                              #
#                  MAIN LOOP                   #
#                                              #
################################################
if __name__ == '__main__':
    args = parse_args()

    # ###### DEUBGGGGG ######
    # args.debug = True
    args.iter = 10
    args.gui = True
    # args.record = True
    # args.grasp = True
    # args.use_follower = False
    # args.use_predictor = False
    # ###### DEUBGGGGG ######

    # computation results for total trials
    total_trial_success_rates = []
    total_trial_wiping_coverages = []
    total_trial_sim_times = []
    total_trial_sim_iters = []
    
    total_trial_wiping_loop_times = []
    total_trial_arm_manip_loop_times = []
    total_trial_next_goal_times = []

    total_trial_wiping_planning_times = []
    total_trial_arm_manip_planning_times = []
    total_trial_arm_manip_dists = []
    total_trial_wiping_dists = []

    total_move_robot_failed_counts = []

    for trial in range(args.trials):
        # store computation times
        total_sim_times = []
        total_sim_iters = []
        wiping_loop_times = []
        next_goal_configs_times = []
        arm_manipulation_loop_times = []

        arm_manip_planning_times = []
        arm_manip_following_times = []
        wiping_planning_times_list = []

        wiping_coverages = []
        total_targets_cleared_list = []
        total_move_robot_count_list = []
        total_arm_manip_count_list = []

        total_arm_manip_dist_list = []
        total_wiping_dist_list = []
        
        trial_success = 1

        # error termination flag
        skip_trial = False

        # simulation environments
        wiping_env = WipingDemo()
        manip_env = ManipulationDemo(gui=args.gui)
        grasp_env = GraspDemo()
        wiping_env.reset()
        manip_env.reset()
        grasp_env.reset()
        reset_base_poses(manip_env, wiping_env, grasp_env)

        # initial joint states
        power_off_pose = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
        q_robot_init = power_off_pose
        q_robot_2_init = power_off_pose
        q_H_init = manip_env.human_rest_poses
        manip_env.reset_robot(manip_env.robot, q_robot_init)
        manip_env.reset_robot(manip_env.robot_2, q_robot_2_init)
        manip_env.reset_human_arm(q_H_init)

        manip_env.lock_robot_gripper_joints(manip_env.robot)  ######
        manip_env.attach_tool()

        ### grasp generation
        if args.grasp:
            print('generating grasps...')
            grasp_start_time = time.time()
            (q_R_grasp_samples, grasp_pose_samples, world_to_eef_goals, 
             best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal) = grasp_env.generate_grasps(q_H_init)
            grasp_end_time = time.time()
            print(f'best_q_R_grasp: {best_q_R_grasp}')
            print(f'best_world_to_grasp: {best_world_to_grasp}')
            print(f'best_world_to_eef_goal: {best_world_to_eef_goal}')
            print(f'grasp generation time: {grasp_end_time-grasp_start_time:.4f}')

            sys.exit(1)

        best_q_R_grasp = [-2.2567504 , -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
        best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [ 0.84583597, -0.13011431, -0.49919509,  0.13577936]]
        best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))

        # Load the predictor model
        if args.use_predictor:
            model_path = "models/arm_config_predictor.pth"
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_combined_model(model_path, device=device)
            print(f'predictor NN model loaded.')

        # save grasp parameters
        manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
        (right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
        right_elbow_joint_to_cp, cp_to_right_elbow_joint,
        right_wrist_joint_to_cp, cp_to_right_wrist_joint) = manip_env.get_grasp_parameters()
        wiping_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                        right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                        right_wrist_joint_to_cp, cp_to_right_wrist_joint)
        grasp_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                    right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                    right_wrist_joint_to_cp, cp_to_right_wrist_joint)
        valid_grasp = wiping_env.validate_q_R(q_H_init, best_q_R_grasp, check_goal=True)
        if not valid_grasp:
            raise ValueError("grasp not valid!")

        ### Profile wiping_loop(): 1st wiping iter (with rest poses)
        wiping_coverage = 0.0
        total_move_robot_count = 0
        total_arm_manip_count = 0
        total_targets_cleared = 0
        targets_cleared = 0

        total_arm_manip_dist = 0.0
        total_wiping_dist = 0.0

        total_targets = wiping_env.targets_util.total_target_count
        manip_env.reset_human_arm(q_H_init)
        manip_env.reset_robot(manip_env.robot, q_robot_init)

        start_time = time.time()
        result, elapsed_time = profile_function(
                "wiping_loop", wiping_loop, wiping_env, manip_env, q_H_init, total_targets_cleared, 
                q_robot_init, q_robot_2_init, args.use_follower, args.gui)
        (targets_cleared, total_targets_cleared, 
         wiping_robot_failed, wiping_robot_count, wiping_dist, 
         wiping_move_success_times, wiping_planning_times) = result

        if wiping_robot_failed > 0:
            trial_success = 0
            total_move_robot_failed_counts.append(1)
        else:
            total_move_robot_failed_counts.append(0)
            trial_success = 1
        
        total_move_robot_count += wiping_robot_count
        wiping_loop_times.append(elapsed_time)
        wiping_planning_times_list += wiping_planning_times
        total_wiping_dist += wiping_dist

        wiping_coverage = total_targets_cleared/total_targets
        print(f'total_targets_cleared: {total_targets_cleared}/{total_targets}')

        # simulation loop until threshold is met...
        i = 0
        if trial_success == 1:
            current_human_joint_angles = q_H_init
            current_joint_angles = q_robot_init
            target_joint_angles = best_q_R_grasp
            current_robot_2_joint_angles = q_robot_2_init
            for i in range(args.iter):
                # reset to grasp pose
                if manip_env.human_cid is None:
                    traj, _ = move_robot_loop(manip_env, robot=manip_env.robot, other_robot=manip_env.robot_2, 
                                            q_robot_init=current_joint_angles, q_robot_goal=target_joint_angles, 
                                            world_to_robot_eef_goal=best_world_to_eef_goal,
                                            q_other_robot=current_robot_2_joint_angles, q_H=current_human_joint_angles,
                                            early_terminate=True)
                    for q_R in traj:
                        manip_env.reset_human_arm(q_H_init)
                        manip_env.reset_robot(manip_env.robot, q_R)
                        if args.gui:
                            time.sleep(0.05)
                    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
                
                ### a. Profiling get_best_valid_goal_configs(): find q_H_goal and q_R_goal using the grasp (human config with best score)
                if not args.use_predictor:
                    n_samples = 50
                    time_out = 60
                    print('finding for goal configs...')
                    result, elapsed_time = profile_function(
                            "get_best_valid_goal_configs", wiping_env.get_best_valid_goal_configs,
                            current_human_joint_angles, current_joint_angles, current_robot_2_joint_angles, n_samples, time_out)
                    q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal = result
                    next_goal_configs_times.append(elapsed_time)
                    print(f'q_H score: {q_H_score}')
                    print(f'q_H_traj = {q_H_traj}')
                    print(f'q_R_traj = {q_R_traj}')

                ### b. Get prediction from the arm config predictor NN model
                if args.use_predictor:
                    next_goal_configs_start = time.time()
                    # Label the pcd
                    env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(robot=manip_env.robot_2, resolution=15)
                    right_arm_pcd_init = np.vstack((right_arm_pcd, right_shoulder_pcd))
                    targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world
                    labeled_pcd = label_pcd(right_arm_pcd_init, targets)
                    
                    # Predict q_H_goal directly using the combined model
                    q_H_goal = predict_q_H_goal(model, q_H_init=current_human_joint_angles, labeled_pcd=labeled_pcd, device=device)
                    print(f"predicted q_H_goal: {q_H_goal}")

                    # get noise sample q_H_goal with best score
                    best_score, best_q_H_goal, best_q_H_traj, q_R_goal, q_R_traj = get_best_noise_sample(q_H_init=current_human_joint_angles,
                                                                                        q_H_goal=q_H_goal,
                                                                                        q_R_init=current_joint_angles)
                    next_goal_configs_times.append(time.time()-next_goal_configs_start)
                    print(f'best_q_H_traj: {best_q_H_traj}')
                    print(f'q_R_traj: {q_R_traj}')
                    # print(best_score, best_q_H_goal, best_q_H_traj, q_R_goal, q_R_traj)
                    if best_q_H_goal is None:
                        trial_success = 0
                        skip_trial = True
                        break

                # save goal parameters
                wiping_env.reset_robot(wiping_env.robot, q_R_goal)
                world_to_eef_goal = wiping_env.bc.getLinkState(wiping_env.robot.id, wiping_env.robot.eef_id)[:2]

                ### Profiling arm_manipulation_loop(): arm manipulation
                result, elapsed_time = profile_function("arm_manipulation_loop", arm_manipulation_loop, 
                                        manip_env, current_robot_2_joint_angles, current_joint_angles, q_R_goal, 
                                        current_human_joint_angles, world_to_eef_goal, q_R_traj)
                arm_manip_planning_time, arm_manip_following_time, arm_manip_loop_failed, arm_manip_total_dist, arm_manip_success_times = result

                if arm_manip_loop_failed > 0:
                    trial_success = 0
                    skip_trial = True
                    break
                else:
                    trial_success = 1
                
                arm_manipulation_loop_times.append(elapsed_time)
                arm_manip_planning_times.extend(arm_manip_planning_time)
                arm_manip_following_times.extend(arm_manip_following_time)
                total_arm_manip_count += 1
                total_arm_manip_dist += arm_manip_total_dist

                # if arm manip failed, reset to initial configs
                if arm_manip_loop_failed == 1:
                    manip_env.detach_human_arm_from_eef()
                    manip_env.reset_human_arm(q_H_init)
                    manip_env.reset_robot(manip_env.robot, best_q_R_grasp)
                    manip_env.attach_human_arm_to_eef()
                    manip_env.targets_util.update_targets()
                
                current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
                current_human_joint_angles = manip_env.get_human_joint_angles()

                ### Profile wiping_loop(): n-th wiping iter
                result, elapsed_time = profile_function(
                            "wiping_loop", wiping_loop, wiping_env, manip_env, current_human_joint_angles, total_targets_cleared, 
                            current_joint_angles, current_robot_2_joint_angles, args.use_follower, args.gui)
                (targets_cleared, total_targets_cleared, 
                 wiping_robot_failed, wiping_robot_count, wiping_dist, 
                 wiping_move_success_times, wiping_planning_times) = result

                if wiping_robot_failed > 0:
                    trial_success = 0
                    skip_trial = True
                    total_move_robot_failed_counts.append(1)
                    break
                else:
                    total_move_robot_failed_counts.append(0)
                    trial_success = 1
                
                wiping_loop_times.append(elapsed_time)
                wiping_planning_times_list += wiping_planning_times
                total_move_robot_count += wiping_robot_count
                total_wiping_dist += wiping_dist

                # check if wiping threshold is reached
                wiping_coverage = total_targets_cleared/total_targets
                print(f'trial {trial} | iter {i+1} | wiping_coverage: {wiping_coverage}, new targets cleared: {targets_cleared}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
                if wiping_coverage >= args.wiping_threshold:
                    break

                # reinforce the grasp
                manip_env.detach_human_arm_from_eef()
                manip_env.attach_human_arm_to_eef()

                # save states
                current_human_joint_angles = manip_env.get_human_joint_angles()
                current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
                current_robot_2_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)

        # end of simulation loop
        total_time = time.time() - start_time

        # collect results
        total_trial_success_rates.append(trial_success)

        # failure termination
        if skip_trial:
            continue

        total_trial_wiping_coverages.append(wiping_coverage)
        total_trial_sim_times.append(total_time)
        total_trial_sim_iters.append(i+1)
        
        total_trial_wiping_loop_times += wiping_loop_times
        total_trial_arm_manip_loop_times += arm_manipulation_loop_times
        total_trial_next_goal_times += next_goal_configs_times

        total_trial_wiping_planning_times += wiping_planning_times_list
        total_trial_arm_manip_planning_times += arm_manip_planning_times
        total_trial_arm_manip_dists.append(total_arm_manip_dist)
        total_trial_wiping_dists.append(total_wiping_dist)

        print(f'\nwiping_coverage: {wiping_coverage}')
        print(f'iteration: {i+1}, total simulation time: {total_time}')
        manip_env.bc.disconnect()
        wiping_env.bc.disconnect()
        grasp_env.bc.disconnect()
        print('done\n')

        # clear up some space...
        torch.cuda.empty_cache()



    # Print the results
    if args.debug:
        # Print the results
        print(f"\n=== Input Arguments ===")
        print(f"use_follower: {args.use_follower}")
        print(f"use_predictor: {args.use_predictor}")

        print(f"\n=== Results in {args.trials} trials ===")
        print("(mean, std, min, max)")
        print(f"success rates: {sum(total_trial_success_rates)/len(total_trial_success_rates):.4f}, {sum(total_trial_success_rates)}/{len(total_trial_success_rates)}")
        
        if sum(total_trial_success_rates) > 0:
            print(f"wiping_coverages: {np.mean(total_trial_wiping_coverages):.4f}, {np.std(total_trial_wiping_coverages):.4f}, {min(total_trial_wiping_coverages):.4f}, {max(total_trial_wiping_coverages):.4f}")
            print(f'total time: {np.mean(total_trial_sim_times):.4f}, {np.std(total_trial_sim_times):.4f}, {min(total_trial_sim_times):.4f}, {max(total_trial_sim_times):.4f}')
            print(f'total iterations: {np.mean(total_trial_sim_iters):.4f}, {np.std(total_trial_sim_iters):.4f}, {min(total_trial_sim_iters):.4f}, {max(total_trial_sim_iters):.4f}')

            print("\n=== Computation Times (in seconds) ===")
            print("(mean, std, min, max)")
            print(f"wiping_loop times: {np.mean(total_trial_wiping_loop_times):.4f}, {np.std(total_trial_wiping_loop_times):.4f}, {min(total_trial_wiping_loop_times):.4f}, {max(total_trial_wiping_loop_times):.4f}")
            print(f"next_goal_configs_times times: {np.mean(total_trial_next_goal_times):.4f}, {np.std(total_trial_next_goal_times):.4f}, {min(total_trial_next_goal_times):.4f}, {max(total_trial_next_goal_times):.4f}")
            print(f"arm_manipulation_loop times: {np.mean(total_trial_arm_manip_loop_times):.4f}, {np.std(total_trial_arm_manip_loop_times):.4f}, {min(total_trial_arm_manip_loop_times):.4f}, {max(total_trial_arm_manip_loop_times):.4f}")

            print("\n=== Computation Times by Components (in seconds) ===")
            print(f"wiping_planning times: {np.mean(total_trial_wiping_planning_times):.4f}, {np.std(total_trial_wiping_planning_times):.4f}, {min(total_trial_wiping_planning_times):.4f}, {max(total_trial_wiping_planning_times):.4f}")
            print(f"arm_manip planning times: {np.mean(total_trial_arm_manip_planning_times):.4f}, {np.std(total_trial_arm_manip_planning_times):.4f}, {min(total_trial_arm_manip_planning_times):.4f}, {max(total_trial_arm_manip_planning_times):.4f}")

            print(f"\n=== Robot Move Distance (in meters) ===")
            print("(mean, std, min, max)")
            print(f"total_arm_manip_dist: {np.mean(total_trial_arm_manip_dists):.4f}, {np.std(total_trial_arm_manip_dists):.4f}, {min(total_trial_arm_manip_dists):.4f}, {max(total_trial_arm_manip_dists):.4f}")
            print(f"total_wiping_dist: {np.mean(total_trial_wiping_dists):.4f}, {np.std(total_trial_wiping_dists):.4f}, {min(total_trial_wiping_dists):.4f}, {max(total_trial_wiping_dists):.4f}\n")

    else:
        dir = "./profiling_results/"
        output_file = f"{args.file}.txt"
        with open(dir+output_file, "w") as f:  
            f.write(f"\n=== Input Arguments ===\n")
            f.write(f"use_follower: {args.use_follower}\n")
            f.write(f"use_predictor: {args.use_predictor}\n")
            f.write(f"max iter: {args.iter}\n")
            f.write(f"wiping threshold: {args.wiping_threshold}\n")

            f.write(f"\n=== Results in {args.trials} trials ===\n")
            f.write("(mean, std, min, max)\n")
            f.write(f"success rates: {sum(total_trial_success_rates)/len(total_trial_success_rates):.4f}, {sum(total_trial_success_rates)}/{len(total_trial_success_rates)}\n")
            
            if sum(total_trial_success_rates) > 0:
                f.write(f"wiping_coverages: {np.mean(total_trial_wiping_coverages):.4f}, {np.std(total_trial_wiping_coverages):.4f}, {min(total_trial_wiping_coverages):.4f}, {max(total_trial_wiping_coverages):.4f}\n")
                f.write(f'total time: {np.mean(total_trial_sim_times):.4f}, {np.std(total_trial_sim_times):.4f}, {min(total_trial_sim_times):.4f}, {max(total_trial_sim_times):.4f}\n')
                f.write(f'total iterations: {np.mean(total_trial_sim_iters):.4f}, {np.std(total_trial_sim_iters):.4f}, {min(total_trial_sim_iters):.4f}, {max(total_trial_sim_iters):.4f}\n')

                f.write("\n=== Computation Times (in seconds) ===\n")
                f.write("(mean, std, min, max)\n")
                f.write(f"wiping_loop times: {np.mean(total_trial_wiping_loop_times):.4f}, {np.std(total_trial_wiping_loop_times):.4f}, {min(total_trial_wiping_loop_times):.4f}, {max(total_trial_wiping_loop_times):.4f}\n")
                f.write(f"next_goal_configs_times times: {np.mean(total_trial_next_goal_times):.4f}, {np.std(total_trial_next_goal_times):.4f}, {min(total_trial_next_goal_times):.4f}, {max(total_trial_next_goal_times):.4f}\n")
                f.write(f"arm_manipulation_loop times: {np.mean(total_trial_arm_manip_loop_times):.4f}, {np.std(total_trial_arm_manip_loop_times):.4f}, {min(total_trial_arm_manip_loop_times):.4f}, {max(total_trial_arm_manip_loop_times):.4f}\n")
                
                f.write("\n=== Computation Times by Components (in seconds) ===\n")
                f.write(f"wiping_planning times: {np.mean(total_trial_wiping_planning_times):.4f}, {np.std(total_trial_wiping_planning_times):.4f}, {min(total_trial_wiping_planning_times):.4f}, {max(total_trial_wiping_planning_times):.4f}\n")
                f.write(f"arm_manip planning times: {np.mean(total_trial_arm_manip_planning_times):.4f}, {np.std(total_trial_arm_manip_planning_times):.4f}, {min(total_trial_arm_manip_planning_times):.4f}, {max(total_trial_arm_manip_planning_times):.4f}\n")

                f.write(f"\n=== Robot Move Distance (in meters) ===\n")
                f.write("(mean, std, min, max)\n")
                f.write(f"total_arm_manip_dist: {np.mean(total_trial_arm_manip_dists):.4f}, {np.std(total_trial_arm_manip_dists):.4f}, {min(total_trial_arm_manip_dists):.4f}, {max(total_trial_arm_manip_dists):.4f}\n")
                f.write(f"total_wiping_dist: {np.mean(total_trial_wiping_dists):.4f}, {np.std(total_trial_wiping_dists):.4f}, {min(total_trial_wiping_dists):.4f}, {max(total_trial_wiping_dists):.4f}\n")

                f.write(f"\n=== Evaluation of Wiping Move Robot ===\n")
                f.write(f"failure rates: {sum(total_move_robot_failed_counts)/len(total_move_robot_failed_counts):.4f}, {sum(total_move_robot_failed_counts)}/{len(total_move_robot_failed_counts)}\n")

            f.close()
    
        print(f"Results written to {dir+output_file}")