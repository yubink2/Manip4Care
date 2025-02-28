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

    manip_env = ManipulationEnv(gui=args.gui, seated=True)
    wiping_env = WipingEnv(seated=True)
    grasp_env = GraspEnv(seated=True)
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
        base_pose = ((0.3, -0.7, 0.65), manip_env.bc.getQuaternionFromEuler((0, 0, 3.14)))
        cube_base_pose = ((0.3, -0.7, 0.65-0.15), (0, 0, 0, 1))
        manip_env.reset_base_pose(manip_env.robot.id, base_pose[0], base_pose[1])
        manip_env.reset_base_pose(manip_env.cube_id, cube_base_pose[0], cube_base_pose[1])
        grasp_env.reset_base_pose(grasp_env.robot.id, base_pose[0], base_pose[1])
        grasp_env.reset_base_pose(grasp_env.cube_id, cube_base_pose[0], cube_base_pose[1])
        wiping_env.reset_base_pose(wiping_env.robot.id, base_pose[0], base_pose[1])
        wiping_env.reset_base_pose(wiping_env.cube_id, cube_base_pose[0], cube_base_pose[1])

        q_H_init = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        (q_R_grasp_samples, grasp_pose_samples, world_to_eef_goals, 
            best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal) = grasp_env.generate_grasps(q_H_init)
        print(f'best_q_R_grasp = {best_q_R_grasp}')
        print(f'best_world_to_grasp = {best_world_to_grasp}')
        print(f'best_world_to_eef_goal = {best_world_to_eef_goal}')

        manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
        (right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
        right_elbow_joint_to_cp, cp_to_right_elbow_joint,
        right_wrist_joint_to_cp, cp_to_right_wrist_joint) = manip_env.get_grasp_parameters()
        wiping_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                        right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                        right_wrist_joint_to_cp, cp_to_right_wrist_joint)
        valid_grasp = wiping_env.validate_q_R(q_H_init, best_q_R_grasp, check_goal=True)
        print(valid_grasp)

        sys.exit()

    base_pose = ((0.3, -0.6, 0.65), manip_env.bc.getQuaternionFromEuler((0, 0, 0)))
    cube_base_pose = ((0.3, -0.6, 0.65-0.15), (0, 0, 0, 1))
    manip_env.reset_base_pose(manip_env.robot.id, base_pose[0], base_pose[1])
    manip_env.reset_base_pose(manip_env.cube_id, cube_base_pose[0], cube_base_pose[1])
    wiping_env.reset_base_pose(wiping_env.robot.id, base_pose[0], base_pose[1])
    wiping_env.reset_base_pose(wiping_env.cube_id, cube_base_pose[0], cube_base_pose[1])

    # (1) save grasp parameters
    best_q_R_grasp = [ 1.36108435, -1.13212489,  1.74734537, -2.31192644, -2.03635668, -1.5895078 ]
    best_world_to_grasp = [[0.41957364, 0.00900415, 0.7129398 ], [ 0.84922636, -0.05141733, -0.51590207, -0.10007946]]
    best_world_to_eef_goal = ((0.354740709066391, 0.006798400543630123, 0.8370037078857422), (0.8492263555526733, -0.05141732469201088, -0.5159021019935608, -0.10007942467927933))


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
    

    manip_env.bc.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    manip_env.targets_util.remove_all_targets()

    
    # (1) initial configs to be tested
    q_H_init_list = [
        [2.1615857835273147, 0.28523470754658165, -1.699560641477742, 0.06216124712917465],
        [-1.7621760436073104, -0.060200294668236815, -0.6650875491270478, 0.031496442162199406],
        [-2.1396446040938635, -0.121193301421046, -0.2253920960814666, 0.4790230806191823],
        [1.5823759154602755, 0.6395197102000857, -1.056530654479765, 0.3616205407113223],
        [-1.418814812836498, -0.23906902429044288, -1.8962327467280808, 0.265519568846039],
    ]
    q_R_init_list = [
        [1.2165636362567418, -1.0946090555846615, 1.1590054333794166, 1.0275864806115618, 1.7696791770019096, 1.3592200925372608],
        [1.1289008698672323, -1.0599987927582284, 2.5119855144293055, -2.707534485313804, 1.3773382749136187, 1.0758546700534009],
        [1.4839680012564769, -1.2722090263460177, 2.5117296692716415, -2.512233685588152, 1.534053353567689, 0.32910465654503523],
        [1.6020339427183123, -0.548511726298289, 0.2804852211031354, -1.376230118935496, -2.507162709136695, -2.2594294712003946],
        [1.2360983161096315, -0.41676313102000084, 1.2813738770800078, -2.3112198687275933, 1.5433024602274519, 2.2098209015851675],
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
        print('genearting new goal configs...')
        q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = wiping_env.get_valid_goal_configs(q_H_init=q_H_init,
                                                                                       q_robot=q_R_init,
                                                                                       q_robot_2=q_robot_2_init,
                                                                                       n_samples=10,
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
        if sum(total_arm_manip_success) > 0:
            print(f'out of range q_0: {max(total_arm_manip_out_of_range_0)}')
            print(f'out of range q_1: {max(total_arm_manip_out_of_range_1)}')
            print(f'out of range q_2: {max(total_arm_manip_out_of_range_2)}')
            print(f'out of range q_3: {max(total_arm_manip_out_of_range_3)}\n')

    # write results
    dir = "./profiling_results/"
    output_file = f"arm_manip_seated_{args.group}_2.txt"
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