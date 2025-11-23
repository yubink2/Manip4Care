# general imports
import os
import pybullet as p
import math
import sys
import time
import numpy as np
import argparse
import torch

# environments
from manip4care.envs.manipulation_env import ManipulationEnv
from manip4care.envs.wiping_env import WipingEnv

# arm config NN model
from nlc_predictor.arm_config_NN_train import CombinedModel, PointNetEncoder, ArmConfigPredictor, normalize_point_cloud
from nlc_predictor.arm_config_dataset_generation import label_pcd

# simulation loop
from manip4care.sim_loops import arm_manipulation_loop, wiping_loop, move_robot_loop


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
        "--wiping-threshold",
        type=float,
        default=0.8,
        help="Specify number for the wiping threshold."
    )

    parser.add_argument(
        "--iter",
        type= int,
        default=10,
        help="Specify number for the simulation iteration number."
    )

    args = parser.parse_args()
    return args

def profile_function(func_name, func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def reset_base_poses(manip_env, wiping_env):
    base_pose = manip_env.get_obj_base_pose("bed")
    wiping_env.reset_base_pose(wiping_env.bed_id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("robot_m")
    wiping_env.reset_base_pose(wiping_env.robot_m.id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("robot_w")
    wiping_env.reset_base_pose(wiping_env.robot_w.id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("humanoid")
    wiping_env.reset_base_pose(wiping_env.humanoid._humanoid, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("cube_m")
    wiping_env.reset_base_pose(wiping_env.cube_m_id, base_pose[0], base_pose[1])

    base_pose = manip_env.get_obj_base_pose("cube_w")
    wiping_env.reset_base_pose(wiping_env.cube_w_id, base_pose[0], base_pose[1])


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

def get_best_noise_sample(wiping_env, q_H_init, q_H_goal, q_R_init, max_noise_trials=1000):
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
    wiping_env.reset_robot(wiping_env.robot_m, q_R_goal)
    
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
            wiping_env.reset_robot(wiping_env.robot_m, q_R_goal)

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

################################################
#                                              #
#                  MAIN LOOP                   #
#                                              #
################################################
if __name__ == '__main__':
    args = parse_args()

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
        wiping_env = WipingEnv(seated=False, wiping=True)
        manip_env = ManipulationEnv(gui=args.gui, seated=False, wiping=True)
        wiping_env.reset()
        manip_env.reset()
        reset_base_poses(manip_env, wiping_env)

        # initial joint states
        power_off_pose = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
        q_robot_init = power_off_pose
        q_robot_2_init = power_off_pose
        q_H_init = manip_env.human_rest_poses
        manip_env.reset_robot(manip_env.robot_m, q_robot_init)
        manip_env.reset_robot(manip_env.robot_w, q_robot_2_init)
        manip_env.reset_human_arm(q_H_init)

        manip_env.lock_robot_gripper_joints(manip_env.robot_m)
        manip_env.attach_tool()

        # example grasp parameters
        best_q_R_grasp = [-2.2567504, -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
        best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [ 0.84583597, -0.13011431, -0.49919509,  0.13577936]]
        best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))

        # Load the predictor model
        if args.use_predictor:
            model_path = "nlc_predictor/models/arm_config_predictor.pth"
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_combined_model(model_path, device=device)
            print(f'predictor NN model loaded.')

        # save grasp parameters
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
        manip_env.reset_robot(manip_env.robot_m, q_robot_init)

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
                    traj, _ = move_robot_loop(manip_env, robot=manip_env.robot_m, other_robot=manip_env.robot_w, 
                                            q_robot_init=current_joint_angles, q_robot_goal=target_joint_angles, 
                                            world_to_robot_eef_goal=best_world_to_eef_goal,
                                            q_other_robot=current_robot_2_joint_angles, q_H=current_human_joint_angles,
                                            early_terminate=True)
                    for q_R in traj:
                        manip_env.reset_human_arm(q_H_init)
                        manip_env.reset_robot(manip_env.robot_m, q_R)
                        if args.gui:
                            time.sleep(0.05)
                    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_m)
                
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
                    env_pcd, arm_pcd, shoulder_pcd = manip_env.compute_env_pcd(robot=manip_env.robot_w, resolution=15)
                    arm_pcd_init = np.vstack((arm_pcd, shoulder_pcd))
                    targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world
                    labeled_pcd = label_pcd(arm_pcd_init, targets)
                    
                    # Predict q_H_goal directly using the combined model
                    q_H_goal = predict_q_H_goal(model, q_H_init=current_human_joint_angles, labeled_pcd=labeled_pcd, device=device)
                    print(f"predicted q_H_goal: {q_H_goal}")

                    # get noise sample q_H_goal with best score
                    best_score, best_q_H_goal, best_q_H_traj, q_R_goal, q_R_traj = get_best_noise_sample(wiping_env=wiping_env, 
                                                                                                         q_H_init=current_human_joint_angles,
                                                                                                         q_H_goal=q_H_goal,
                                                                                                         q_R_init=current_joint_angles)
                    next_goal_configs_times.append(time.time()-next_goal_configs_start)
                    print(f'best_q_H_traj: {best_q_H_traj}')
                    print(f'q_R_traj: {q_R_traj}')
                    if best_q_H_goal is None:
                        trial_success = 0
                        skip_trial = True
                        break

                # save goal parameters
                wiping_env.reset_robot(wiping_env.robot_m, q_R_goal)
                world_to_eef_goal = wiping_env.bc.getLinkState(wiping_env.robot_m.id, wiping_env.robot_m.eef_id)[:2]

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
                    manip_env.reset_robot(manip_env.robot_m, best_q_R_grasp)
                    manip_env.attach_human_arm_to_eef()
                    manip_env.targets_util.update_targets()
                
                current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_m)
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
                current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_m)
                current_robot_2_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_w)

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
        print('done\n')

        # clear up some space...
        torch.cuda.empty_cache()



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
