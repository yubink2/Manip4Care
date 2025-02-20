from wiping_manipulation_demo import *


## supine position
## 2 grasp, 2 robot base poses (left, right)
## 5 initial configs
## for each, 10 valid goal configs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable GUI. (Default: False)"
    )

    parser.add_argument(
        "--grasp",
        action="store_true",
        default=False,
        help="Generate grasp only. (Default: False)"
    )

    parser.add_argument(
        "--group",
        type= str,
        default="None",
        help="Specify the group for shoulder range reduction. (Options: A, B, C, D. Default: None)."
    )

    args = parser.parse_args()
    return args

def check_out_of_range(manip_env, human_joint_angles):
    """
    For a list of human_joint_angles (each of shape [4] for 4 DoFs),
    we check how much each angle is off from its allowed range.
    """
    # Separate lists for each DoF
    out_of_range_0 = []
    out_of_range_1 = []
    out_of_range_2 = []
    out_of_range_3 = []

    for angles in human_joint_angles:
        for j, angle in enumerate(angles):
            if angle < manip_env.human_arm_lower_limits[j]:
                # angle is below the lower limit
                off_by = manip_env.human_arm_lower_limits[j] - angle
            elif angle > manip_env.human_arm_upper_limits[j]:
                # angle is above the upper limit
                off_by = angle - manip_env.human_arm_upper_limits[j]
            else:
                off_by = 0

            # Append to the corresponding DoF list
            if j == 0:
                out_of_range_0.append(off_by)
            elif j == 1:
                out_of_range_1.append(off_by)
            elif j == 2:
                out_of_range_2.append(off_by)
            elif j == 3:
                out_of_range_3.append(off_by)
    
    return (
        out_of_range_0,
        out_of_range_1,
        out_of_range_2,
        out_of_range_3
    )

if __name__ == '__main__':
    args = parse_args()

    args.gui = True

    manip_env = ManipulationDemo(gui=args.gui)
    wiping_env = WipingDemo()
    grasp_env = GraspDemo()
    manip_env.reset()
    wiping_env.reset()
    grasp_env.reset()

    # initial configurations
    power_off_pose = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    q_robot_init = power_off_pose
    q_robot_2_init = power_off_pose
    q_H_init = manip_env.human_rest_poses

    off_base_pose = ((100,100,100), (0,0,0,1))
    manip_env.reset_base_pose(manip_env.robot_2.id, off_base_pose[0], off_base_pose[1])
    manip_env.reset_base_pose(manip_env.cube_2_id, off_base_pose[0], off_base_pose[1])
    manip_env.reset_base_pose(manip_env.tool, off_base_pose[0], off_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.robot_2.id, off_base_pose[0], off_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.cube_2_id, off_base_pose[0], off_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.tool, off_base_pose[0], off_base_pose[1])
    grasp_env.reset_base_pose(grasp_env.robot_2.id, off_base_pose[0], off_base_pose[1])
    grasp_env.reset_base_pose(grasp_env.cube_2_id, off_base_pose[0], off_base_pose[1])

    # set lower and upper joint limits
    manip_env.set_arm_joint_range(shoulder_reduction_group=args.group)
    wiping_env.set_arm_joint_range(shoulder_reduction_group=args.group)

    if args.grasp:
        base_pose = ((0.65, -0.1, 0.3), manip_env.bc.getQuaternionFromEuler((0, 0, 3.14)))
        cube_base_pose = ((0.65, -0.1, 0.3-0.15), (0, 0, 0, 1))
        grasp_env.reset_base_pose(grasp_env.robot.id, base_pose[0], base_pose[1])
        grasp_env.reset_base_pose(grasp_env.cube_id, cube_base_pose[0], cube_base_pose[1])

        (q_R_grasp_samples, grasp_pose_samples, world_to_eef_goals, 
            best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal) = grasp_env.generate_grasps(q_H_init)
        print(f'best_q_R_grasp = {best_q_R_grasp}')
        print(f'best_world_to_grasp = {best_world_to_grasp}')
        print(f'best_world_to_eef_goal = {best_world_to_eef_goal}')
        sys.exit()

    # (1) save grasp parameters
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
    valid_grasp = wiping_env.validate_q_R(q_H_init, best_q_R_grasp, check_goal=True)
    if not valid_grasp:
        raise ValueError("grasp not valid!")


    
    # (1) initial configs to be tested
    q_H_init_list = [
        [1.706688846002959, -0.005017477384511082, -1.8918996283379863, 0.8342766491976187],
        [1.8458923812102945, 0.11890604614311356, -1.3088823667360043, 0.7979126167771062],
        [2.0730411898481407, -0.002577228317448499, -2.010925458024876, 1.201286960930556], 
        [1.8745536686038013, 0.1285049446091689, -1.295355269053017, 1.2429816530424103],
        [1.9301600285681788, -0.24998806613325258, -1.6972369443545605, 0.2652148931773943],
    ]
    q_R_init_list = [
        [-2.7471681838109645, -1.65491557227954, 2.251861694429345, -0.9553326768283779, -0.5238059926996428, -0.9455807027874162],
        [-2.798941752558967, -1.7123013522402524, 2.523092056085701, -1.7485637523489164, -0.8995531555693265, -0.8222116327014303],
        [-2.626995375706796, -1.3676339396770405, 1.8608075024564998, -0.6375865646071406, -0.7149545533922784, -0.8446012290390407],
        [-2.676445050004803, -1.4498327735846255, 2.198920206418347, -1.4455034095071295, -1.1775125402273956, -1.0060983059735826],
        [1.7106566457528243, -1.6165119012856406, -2.102349497816576, 3.0614661299905177, -1.5166772359511809, -1.383736088499672],
    ]


    ### evaluation loop
    total_arm_manip_success = []
    total_arm_manip_planning_times = []
    total_arm_manip_following_times = []
    total_arm_manip_loop_times = []
    total_arm_manip_dist_list = []

    total_arm_manip_out_of_range_0 = []
    total_arm_manip_out_of_range_1 = []
    total_arm_manip_out_of_range_2 = []
    total_arm_manip_out_of_range_3 = []

    for q_H_init, q_R_init in zip(q_H_init_list, q_R_init_list):
        # generate new goal configs
        q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = wiping_env.get_valid_goal_configs(q_H_init=q_H_init,
                                                                                       q_robot=q_R_init,
                                                                                       q_robot_2=q_robot_2_init,
                                                                                       n_samples=5,
                                                                                       time_out=500)

        for q_H_traj, q_R_traj, q_H_goal, q_R_goal in zip(q_H_trajs, q_R_trajs, q_H_goals, q_R_goals):
            # save goal parameters
            wiping_env.reset_robot(wiping_env.robot, q_R_goal)
            world_to_eef_goal = wiping_env.bc.getLinkState(wiping_env.robot.id, wiping_env.robot.eef_id)[:2]

            # arm manip
            manip_env.detach_human_arm_from_eef()
            manip_env.reset_human_arm(q_H_init)
            manip_env.reset_robot(manip_env.robot, q_R_init)

            start_time = time.time()
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
            end_time = time.time()

            # save results
            if arm_manip_loop_failed > 0:
                total_arm_manip_success.append(0)
            else:
                total_arm_manip_success.append(1)
                total_arm_manip_planning_times.extend(arm_manip_planning_times)
                total_arm_manip_following_times.extend(arm_manip_following_times)
                total_arm_manip_loop_times.append(end_time-start_time)
                total_arm_manip_dist_list.append(arm_manip_total_dist)

                (out_of_range_0, 
                 out_of_range_1, 
                 out_of_range_2, 
                 out_of_range_3) = check_out_of_range(manip_env=manip_env, human_joint_angles=arm_manip_human_joint_angles)
                total_arm_manip_out_of_range_0.extend(out_of_range_0)
                total_arm_manip_out_of_range_1.extend(out_of_range_1)
                total_arm_manip_out_of_range_2.extend(out_of_range_2)
                total_arm_manip_out_of_range_3.extend(out_of_range_3)

        print(f'\nq_H_init: {q_H_init}')
        print(f'success: {sum(total_arm_manip_success)} / {len(total_arm_manip_success)}')
        print(f'out of range q_0: {max(total_arm_manip_out_of_range_0)}')
        print(f'out of range q_1: {max(total_arm_manip_out_of_range_1)}')
        print(f'out of range q_2: {max(total_arm_manip_out_of_range_2)}')
        print(f'out of range q_3: {max(total_arm_manip_out_of_range_3)}\n')

    # write results
    dir = "./profiling_results/"
    output_file = f"arm_manip_{args.group}_1.txt"
    with open(dir+output_file, "w") as f:  
        f.write(f"\n=== Input Arguments ===\n")
        f.write(f"success rates: {sum(total_arm_manip_success)/len(total_arm_manip_success):.4f}, {sum(total_arm_manip_success)}/{len(total_arm_manip_success)}\n")
        f.write(f"planning time: {np.mean(total_arm_manip_planning_times):.4f}, {np.std(total_arm_manip_planning_times):.4f}, {min(total_arm_manip_planning_times):.4f}, {max(total_arm_manip_planning_times):.4f}\n")
        f.write(f"following time: {np.mean(total_arm_manip_following_times):.4f}, {np.std(total_arm_manip_following_times):.4f}, {min(total_arm_manip_following_times):.4f}, {max(total_arm_manip_following_times):.4f}\n")
        f.write(f"arm_manip_total_times: {np.mean(total_arm_manip_loop_times):.4f}, {np.std(total_arm_manip_loop_times):.4f}, {min(total_arm_manip_loop_times):.4f}, {max(total_arm_manip_loop_times):.4f}\n")
        f.write(f"move distance: {np.mean(total_arm_manip_dist_list):.4f}, {np.std(total_arm_manip_dist_list):.4f}, {min(total_arm_manip_dist_list):.4f}, {max(total_arm_manip_dist_list):.4f}\n")
        f.write(f"out of range q_0: {np.mean(total_arm_manip_out_of_range_0):.4f}, {np.std(total_arm_manip_out_of_range_0):.4f}, {min(total_arm_manip_out_of_range_0):.4f}, {max(total_arm_manip_out_of_range_0):.4f}\n")
        f.write(f"out of range q_1: {np.mean(total_arm_manip_out_of_range_1):.4f}, {np.std(total_arm_manip_out_of_range_1):.4f}, {min(total_arm_manip_out_of_range_1):.4f}, {max(total_arm_manip_out_of_range_1):.4f}\n")
        f.write(f"out of range q_2: {np.mean(total_arm_manip_out_of_range_2):.4f}, {np.std(total_arm_manip_out_of_range_2):.4f}, {min(total_arm_manip_out_of_range_2):.4f}, {max(total_arm_manip_out_of_range_2):.4f}\n")
        f.write(f"out of range q_3: {np.mean(total_arm_manip_out_of_range_3):.4f}, {np.std(total_arm_manip_out_of_range_3):.4f}, {min(total_arm_manip_out_of_range_3):.4f}, {max(total_arm_manip_out_of_range_3):.4f}\n")
        
        f.close()
    
    print(f"Results written to {dir+output_file}")