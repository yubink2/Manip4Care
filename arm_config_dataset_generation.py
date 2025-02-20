
import numpy as np
import pickle
import matplotlib.pyplot as plt

# from envs.utils.transform_utils import *
# from envs.utils.point_cloud_utils import *
from envs.manipulation_env import ManipulationDemo
from envs.wiping_env import WipingDemo


def label_pcd(right_arm_pcd, targets, radius=0.03):
    # label each point in the following format: [x, y, z, label]
    labeled_pcd = []
    radius_squared = radius ** 2
    
    for point in right_arm_pcd:
        x, y, z = point
        label = 0
        
        for target in targets:
            tx, ty, tz = target
            distance_squared = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
            
            if distance_squared <= radius_squared:
                # mark as within sphere
                label = 1
                break
                
        labeled_pcd.append([x, y, z, label])
    
    return labeled_pcd

def visualize_labeled_pcd(labeled_pcd):
    # Separate points by label
    points_label_1 = [point[:3] for point in labeled_pcd if point[3] == 1]  # Points with label 1
    points_label_0 = [point[:3] for point in labeled_pcd if point[3] == 0]  # Points with label 0
    
    # Convert to numpy arrays for plotting
    points_label_1 = np.array(points_label_1)
    points_label_0 = np.array(points_label_0)
    print('points_label_1: ', len(points_label_1))
    print('points_label_0: ', len(points_label_0))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with label 1 (red)
    if points_label_1.size > 0:
        ax.scatter(points_label_1[:, 0], points_label_1[:, 1], points_label_1[:, 2], 
                   c='red', label='Label 1', s=20)
    
    # Plot points with label 0 (blue)
    if points_label_0.size > 0:
        ax.scatter(points_label_0[:, 0], points_label_0[:, 1], points_label_0[:, 2], 
                   c='blue', label='Label 0', s=20)
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Labeled Point Cloud Visualization')
    ax.legend()
    
    plt.show()

def save_dataset(iteration):
    file_prefix = f'data/{iteration}'
    with open(f'{file_prefix}_q_H_init_list.dat', 'wb') as file:
        pickle.dump(np.array(q_H_init_list), file)
    with open(f'{file_prefix}_labeled_pcd_init_list.dat', 'wb') as file:
        pickle.dump(np.array(labeled_pcd_init_list), file)
    with open(f'{file_prefix}_q_H_goal_list.dat', 'wb') as file:
        pickle.dump(np.array(q_H_goal_list), file)
    with open(f'{file_prefix}_labeled_pcd_goal_list.dat', 'wb') as file:
        pickle.dump(np.array(labeled_pcd_goal_list), file)

    with open(f'{file_prefix}_targets_pos_on_upperarm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_pos_on_upperarm_list), file)
    with open(f'{file_prefix}_targets_orn_on_upperarm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_orn_on_upperarm_list), file)
    with open(f'{file_prefix}_targets_upperarm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_upperarm_list), file)
    with open(f'{file_prefix}_targets_pos_on_forearm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_pos_on_forearm_list), file)
    with open(f'{file_prefix}_targets_orn_on_forearm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_orn_on_forearm_list), file)
    with open(f'{file_prefix}_targets_forearm_list.dat', 'wb') as file:
        pickle.dump(np.array(targets_forearm_list), file)

    with open(f'{file_prefix}_score_list.dat', 'wb') as file:
        pickle.dump(np.array(score_list), file)

    print(f'Dataset saved successfully at iteration {iteration}.')

if __name__ == '__main__':
    # simulation environments
    wiping_env = WipingDemo(gui=False)
    manip_env = ManipulationDemo(gui=False)
    wiping_env.reset()
    manip_env.reset()
    manip_env.lock_robot_gripper_joints(manip_env.robot)

    # save grasp parameters
    q_H_init = manip_env.human_rest_poses
    best_q_R_grasp = [-2.2567504 , -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
    best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [ 0.84583597, -0.13011431, -0.49919509,  0.13577936]]
    best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))

    manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
    (right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
     right_elbow_joint_to_cp, cp_to_right_elbow_joint,
     right_wrist_joint_to_cp, cp_to_right_wrist_joint) = manip_env.get_grasp_parameters()
    wiping_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                    right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                    right_wrist_joint_to_cp, cp_to_right_wrist_joint)

    # initial joint states
    q_robot_init = best_q_R_grasp
    q_robot_2_init = manip_env.robot_2.arm_rest_poses
    q_H_init = manip_env.human_rest_poses

    # datasets to be collected
    q_H_init_list = []
    labeled_pcd_init_list = []
    q_H_goal_list = []
    labeled_pcd_goal_list = []
    targets_pos_on_upperarm_list = []
    targets_orn_on_upperarm_list = []
    targets_upperarm_list = []
    targets_pos_on_forearm_list = []
    targets_orn_on_forearm_list = []
    targets_forearm_list = []
    score_list = []

    # trials
    cases = [1, 2, 3, 4, 5, 6, 7]
    biases = [True, False]
    i = 1

    # initial wiping targets set up
    manip_env.targets_util.reorder_targets()

    # synchronize targets in manip_env & wiping_env
    (targets_pos_on_upperarm, targets_orn_on_upperarm, targets_upperarm, 
    targets_pos_on_forearm, targets_orn_on_forearm, targets_forearm) = manip_env.targets_util.get_targets()
    wiping_env.targets_util.set_targets(targets_pos_on_upperarm, targets_orn_on_upperarm, targets_upperarm, 
                                        targets_pos_on_forearm, targets_orn_on_forearm, targets_forearm)

    # dataset collection loop
    while True:
        print('collecting valid goal configs...')
        q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = wiping_env.get_valid_goal_configs(q_H_init=q_H_init, 
                                                                                       q_robot=q_robot_init, 
                                                                                       q_robot_2=q_robot_2_init,
                                                                                       n_samples=30,
                                                                                       time_out=100)
        if len(q_H_trajs) <= 0:
            # compute new q_H_init & q_robot_init
            q_H_goal, q_R_goal = wiping_env.get_new_human_robot_configs(q_H_init=q_H_init, q_robot=q_robot_init, q_robot_2=q_robot_2_init)
            q_H_init = q_H_goal
            q_robot_init = q_R_goal
            print('test next q_H_init..\n')
            continue

        for case in cases:
            for bias in biases:
                print(f'{i} | case: {case}, bias: {bias}')
                print(f'q_H_init: {q_H_init}')
                targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world
                # print(f'number of original targets: {len(targets)}')

                ### q_H_init
                # generate new wiping targets
                manip_env.lock_human_joints(q_H_init)
                wiping_env.lock_human_joints(q_H_init)
                manip_env.targets_util.remove_some_targets(case=case, use_bias=bias)
                manip_env.targets_util.update_targets()
                targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world
                print(f'number of targets: {len(targets)}')

                # get right arm pcd
                manip_env.reset_human_arm(q_H_init)
                wiping_env.reset_human_arm(q_H_init)
                manip_env.reset_robot(manip_env.robot, q_robot_init)
                wiping_env.reset_robot(manip_env.robot, q_robot_init)
                env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(robot=manip_env.robot_2, resolution=15)
                total_right_arm_pcd = np.vstack((right_arm_pcd, right_shoulder_pcd))

                # label pcd
                labeled_pcd_init = label_pcd(total_right_arm_pcd, targets)

                ### q_H_goal
                # synchronize targets in manip_env & wiping_env
                (targets_pos_on_upperarm, targets_orn_on_upperarm, targets_upperarm, 
                targets_pos_on_forearm, targets_orn_on_forearm, targets_forearm) = manip_env.targets_util.get_targets()
                wiping_env.targets_util.set_targets(targets_pos_on_upperarm, targets_orn_on_upperarm, targets_upperarm, 
                                                    targets_pos_on_forearm, targets_orn_on_forearm, targets_forearm)
                wiping_env.targets_util.update_targets()
                targets = wiping_env.targets_util.targets_pos_upperarm_world + wiping_env.targets_util.targets_pos_forearm_world

                # find q_H_goal and q_R_goal using the grasp (human config with best score)
                print('finding for goal config with highest score...')
                q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal = wiping_env.get_valid_goal_configs_with_best_score(q_H_init=q_H_init, 
                                                                                                                      q_robot=q_robot_init, 
                                                                                                                      q_robot_2=q_robot_2_init,
                                                                                                                      q_H_trajs=q_H_trajs,
                                                                                                                      q_R_trajs=q_R_trajs,
                                                                                                                      q_H_goals=q_H_goals,
                                                                                                                      q_R_goals=q_R_goals)
                print(f'q_H score: {q_H_score}, q_H_goal: {q_H_goal}')
                
                # skip if score is too low
                if q_H_score < 0.2:
                    print('excluding from the dataset..\n')
                    # restore original targets    
                    manip_env.targets_util.restore_removed_targets()
                    targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world

                    # compute new q_H_init & q_robot_init
                    q_H_goal, q_R_goal = wiping_env.get_new_human_robot_configs(q_H_init=q_H_init, q_robot=q_robot_init, q_robot_2=q_robot_2_init)
                    q_H_init = q_H_goal
                    q_robot_init = q_R_goal
                    continue

                # get right arm pcd
                manip_env.reset_human_arm(q_H_goal)
                wiping_env.reset_human_arm(q_H_goal)
                manip_env.reset_robot(manip_env.robot, q_R_goal)
                wiping_env.reset_robot(manip_env.robot, q_R_goal)
                env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(robot=manip_env.robot_2, resolution=15)
                total_right_arm_pcd = np.vstack((right_arm_pcd, right_shoulder_pcd))
                
                # get targets
                manip_env.lock_human_joints(q_H_goal)
                wiping_env.lock_human_joints(q_H_goal)
                manip_env.targets_util.update_targets()
                targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world

                # label pcd
                labeled_pcd_goal = label_pcd(total_right_arm_pcd, targets)

                # store dataset
                q_H_init_list.append(q_H_init)
                labeled_pcd_init_list.append(labeled_pcd_init)
                q_H_goal_list.append(q_H_goal)
                score_list.append(q_H_score)
                labeled_pcd_goal_list.append(labeled_pcd_goal)
                targets_pos_on_upperarm_list.append(targets_pos_on_upperarm)
                targets_orn_on_upperarm_list.append(targets_orn_on_upperarm)
                targets_upperarm_list.append(targets_upperarm)
                targets_pos_on_forearm_list.append(targets_pos_on_forearm)
                targets_orn_on_forearm_list.append(targets_orn_on_forearm)
                targets_forearm_list.append(targets_forearm)
                
                # restore original targets    
                manip_env.targets_util.restore_removed_targets()
                targets = manip_env.targets_util.targets_pos_upperarm_world + manip_env.targets_util.targets_pos_forearm_world
                print('')

        # compute new q_H_init & q_robot_init
        q_H_goal, q_R_goal = wiping_env.get_new_human_robot_configs(q_H_init=q_H_init, q_robot=q_robot_init, q_robot_2=q_robot_2_init)
        q_H_init = q_H_goal
        q_robot_init = q_R_goal
        print('\n')
        
        # save the dataset
        if i in [500, 1000, 1500, 2000]:
            save_dataset(i)

        if i == 2000:
            break

        # increment counter
        i += 1


    print(f'q_H_init_list: {len(q_H_init_list)}, {len(q_H_init_list[0])}')
    print(f'labeled_pcd_init_list: {len(labeled_pcd_init_list)}, {len(labeled_pcd_init_list[0])}, {len(labeled_pcd_init_list[0][0])}')
    print(f'q_H_goal_list: {len(q_H_goal_list)}, {len(q_H_goal_list[0])}')
    print(f'labeled_pcd_goal_list: {len(labeled_pcd_goal_list)}, {len(labeled_pcd_goal_list[0])}, {len(labeled_pcd_goal_list[0][0])}')

    # store dataset
    with open('data/q_H_init_list.dat', 'wb') as file:
        pickle.dump(np.array(q_H_init_list), file)
    with open('data/labeled_pcd_init_list.dat', 'wb') as file:
        pickle.dump(np.array(labeled_pcd_init_list), file)
    with open('data/q_H_goal_list.dat', 'wb') as file:
        pickle.dump(np.array(q_H_goal_list), file)
    with open('data/labeled_pcd_goal_list.dat', 'wb') as file:
        pickle.dump(np.array(labeled_pcd_goal_list), file)
    print('dataset saved successfully -- final')

    # print dataset
    with open('data/q_H_init_list.dat', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data, '\n')
    with open('data/labeled_pcd_init_list.dat', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data, '\n')
    with open('data/q_H_goal_list.dat', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data, '\n')
    with open('data/labeled_pcd_goal_list.dat', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data, '\n')