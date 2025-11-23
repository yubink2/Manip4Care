# general imports
import numpy as np
import time

################################################
#                                              #
#               SIMULATION LOOPS               #
#                                              #
################################################
def compute_travel_distance(eef_pos_list):
    # iterate over the trajectory to calculate the distance
    total_distance = 0.0
    for i in range(len(eef_pos_list) - 1):
        eef_pos_current = np.array(eef_pos_list[i])
        eef_pos_next = np.array(eef_pos_list[i + 1])
        distance = np.linalg.norm(eef_pos_next - eef_pos_current)
        total_distance += distance

    return total_distance

def wiping_loop(wiping_env, manip_env, q_H, total_targets_cleared, q_robot, q_robot_2_init, use_follower=True, gui=False):
    wiping_planning_times = []
    wiping_dist = 0
    wiping_eef_pos_list = []
    wiping_move_success_times = []

    # initialize environments
    current_joint_angles = q_robot_2_init
    manip_env.lock_human_joints(q_H)
    manip_env.targets_util.update_targets()
    wiping_env.lock_robot_arm_joints(wiping_env.robot_m, q_robot)

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
            eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_w, 
                                                    current_joint_angles=current_joint_angles, target_joint_angles=robot_traj[0])
            if use_follower:
                (move_robot_failed, move_robot_total_dist, 
                 move_robot_total_time, move_robot_planning_times) = move_robot_loop(manip_env, 
                                                                                     robot=manip_env.robot_w, other_robot=manip_env.robot_m, 
                                                                                     q_robot_init=current_joint_angles, q_robot_goal=robot_traj[0], 
                                                                                     world_to_robot_eef_goal=eef_goal_pose,
                                                                                     q_other_robot=q_robot, q_H=q_H)
                wiping_robot_count += 1
                wiping_dist += move_robot_total_dist
                wiping_planning_times += move_robot_planning_times
                if move_robot_failed > 0:
                    wiping_robot_failed += 1
                    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_w)
                    manip_env.targets_util.unmark_feasible_targets()
                    print('wiping move robot failed!')
                    break
                else:
                    wiping_move_success_times.append(move_robot_total_time)
            else:
                traj, move_robot_planning_times = move_robot_loop(manip_env, robot=manip_env.robot_w, other_robot=manip_env.robot_m, 
                                                                  q_robot_init=current_joint_angles, q_robot_goal=robot_traj[0], 
                                                                  world_to_robot_eef_goal=eef_goal_pose,
                                                                  q_other_robot=q_robot, q_H=q_H, 
                                                                  early_terminate=True)
                wiping_robot_count += 1
                wiping_planning_times += move_robot_planning_times
                is_robot_2_in_collision = False
                for q_R in traj:
                    manip_env.reset_robot(manip_env.robot_w, q_R)
                    manip_env.reset_robot(manip_env.robot_m, q_robot)
                    wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_w.id, manip_env.robot_w.eef_id)[0]))
                    if manip_env.robot_w_in_collision(q_R):
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
                    manip_env.move_robot(manip_env.robot_w, q_R)
                    manip_env.move_robot(manip_env.robot_m, q_robot)
                    manip_env.bc.stepSimulation()
                if gui:
                    time.sleep(0.05)
                new_target, indices_to_delete = manip_env.targets_util.get_new_contact_points(targeted_arm=arm)
                manip_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)
                wiping_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)

                targets_cleared += new_target
                total_targets_cleared += new_target
                wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_w.id, manip_env.robot_w.eef_id)[0]))

            manip_env.targets_util.remove_targets()
            manip_env.targets_util.unmark_feasible_targets()
            manip_env.targets_util.update_targets()
            if not use_follower:
                manip_env.detach_tool()  ####

            wiping_env.targets_util.remove_targets()
            wiping_env.targets_util.update_targets()

            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_w)

        # if wiping robot failed, terminate
        if wiping_robot_failed > 0:
            break

    # move robot_2 back to rest poses
    eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_w, 
                                            current_joint_angles=current_joint_angles, target_joint_angles=q_robot_2_init)
    if use_follower:
        (move_robot_failed, move_robot_total_dist, 
         move_robot_total_time, move_robot_planning_times) = move_robot_loop(manip_env, 
                                                                             robot=manip_env.robot_w, other_robot=manip_env.robot_m, 
                                                                             q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, 
                                                                             world_to_robot_eef_goal=eef_goal_pose,
                                                                             q_other_robot=q_robot, q_H=q_H)
        wiping_robot_count += 1
        wiping_dist += move_robot_total_dist
        wiping_planning_times += move_robot_planning_times

        if move_robot_failed > 0:
            wiping_robot_failed += 1
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_w)
            manip_env.detach_tool()
            manip_env.reset_robot(wiping_env.robot_w, q_robot_2_init)
            manip_env.attach_tool()
            print('wiping move robot failed!')
        else:
            wiping_move_success_times.append(move_robot_total_time)
    else:
        traj, move_robot_planning_times = move_robot_loop(manip_env, robot=manip_env.robot_w, other_robot=manip_env.robot_m, 
                                                          q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, 
                                                          world_to_robot_eef_goal=eef_goal_pose,
                                                          q_other_robot=q_robot, q_H=q_H, 
                                                          early_terminate=True)
        wiping_robot_count += 1
        wiping_planning_times += move_robot_planning_times
        is_robot_2_in_collision = False
        for q_R in traj:
            manip_env.reset_robot(manip_env.robot_w, q_R)
            manip_env.reset_robot(manip_env.robot_m, q_robot)
            wiping_eef_pos_list.append(list(manip_env.bc.getLinkState(manip_env.robot_w.id, manip_env.robot_w.eef_id)[0]))
            if manip_env.robot_w_in_collision(q_R):
                is_robot_2_in_collision = True
                break
            if gui:
                time.sleep(0.05)
        if is_robot_2_in_collision:
            wiping_robot_failed += 1
            manip_env.reset_robot(manip_env.robot_w, q_robot_2_init)
        manip_env.detach_tool()  ####

    wiping_env.reset_robot(wiping_env.robot_w, q_robot_2_init)
    manip_env.unlock_human_joints(q_H)

    wiping_dist += compute_travel_distance(wiping_eef_pos_list)
    print('wiping loop is done')
    return targets_cleared, total_targets_cleared, wiping_robot_failed, wiping_robot_count, wiping_dist, wiping_move_success_times, wiping_planning_times

def arm_manipulation_loop(manip_env, 
                          q_robot_2, 
                          q_robot_init, q_robot_goal, 
                          q_H_init, 
                          world_to_eef_goal, q_R_init_traj, 
                          manip_demo=False):
    arm_manip_planning_times = []
    arm_manip_following_times = []
    arm_manip_eef_pos_list = []
    if manip_demo:
        arm_manip_human_joint_angles = []

    if manip_env.robot_w is not None:
        manip_env.lock_robot_arm_joints(manip_env.robot_w, q_robot_2)  #####

    # Step 0: instantiate a new motion planning problem
    trajectory_planner = manip_env.init_traj_planner(manip_env.world_to_robot_m_base, clamp_by_human=True,
                                                     q_H_init=q_H_init, q_R_init=q_robot_init)
    trajectory_follower = manip_env.init_traj_follower(manip_env.world_to_robot_m_base)
    
    # Step 1: move robot to grasping pose
    if manip_env.robot_w is not None:
        manip_env.reset_robot(manip_env.robot_w, q_robot_2)
    manip_env.reset_robot(manip_env.robot_m, q_robot_init)
    manip_env.reset_human_arm(q_H_init)
    if manip_env.wiping:
        manip_env.targets_util.update_targets()

    # Step 2: attach human arm to eef
    env_pcd, arm_pcd, _ = manip_env.compute_env_pcd(robot=manip_env.robot_w, 
                                                          add_bed_padding=False, add_chest_padding=False)
    T_eef_to_object, T_object_to_world = manip_env.attach_human_arm_to_eef(attach_to_gripper=True, 
                                                                           arm_pcd=arm_pcd, trajectory_planner=trajectory_planner)

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
    trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=arm_pcd,
                                          T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                          T_world_to_human_base=manip_env.T_world_to_human_base, T_elbow_joint_to_cp=manip_env.T_elbow_joint_to_cp,
                                          human_arm_lower_limits=manip_env.human_arm_lower_limits, human_arm_upper_limits=manip_env.human_arm_upper_limits)
    
    current_joint_angles = q_robot_init
    current_human_joint_angles = q_H_init
    world_to_eef = manip_env.bc.getLinkState(manip_env.robot_m.id, manip_env.robot_m.eef_id)[:2]

    trajectory_follower._init_H_clamping(manip_env.eef_to_cp, manip_env.elbow_joint_to_cp, 
                                         manip_env.robot_m_base_pose, manip_env.human_base_pose,
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
                manip_env.move_robot(manip_env.robot_m, q_robot_goal)
                manip_env.bc.stepSimulation()
            if manip_env.wiping:
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
                manip_env.move_robot(manip_env.robot_m, next_joint_angles)
                manip_env.bc.stepSimulation()
            if manip_env.wiping:
                manip_env.targets_util.update_targets()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_m)
            current_human_joint_angles = manip_env.get_human_joint_angles()
            world_to_eef = manip_env.bc.getLinkState(manip_env.robot_m.id, manip_env.robot_m.eef_id)[:2]

            if manip_demo:
                arm_manip_human_joint_angles.append(current_human_joint_angles)

    # Step 6: reinforce the grasp
    manip_env.detach_human_arm_from_eef()
    world_to_elbow = manip_env.bc.getLinkState(manip_env.humanoid._humanoid, manip_env.elbow)[:2]
    world_to_cp = manip_env.bc.multiplyTransforms(world_to_elbow[0], world_to_elbow[1],
                                                  manip_env.elbow_to_cp[0], manip_env.elbow_to_cp[1])
    world_to_eef = manip_env.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                   manip_env.cp_to_eef[0], manip_env.cp_to_eef[1])
    q_R_goal = manip_env.bc.calculateInverseKinematics(manip_env.robot_m.id, manip_env.robot_m.eef_id, world_to_eef[0], world_to_eef[1],
                                                    manip_env.robot_m.arm_lower_limits, manip_env.robot_m.arm_upper_limits, manip_env.robot_m.arm_joint_ranges, manip_env.robot_m.arm_rest_poses,
                                                    maxNumIterations=50)
    q_R_goal = [q_R_goal[i] for i in range(len(manip_env.robot_m.arm_controllable_joints))]
    manip_env.reset_robot(manip_env.robot_m, q_R_goal)
    manip_env.attach_human_arm_to_eef()

    if manip_env.robot_w is not None:
        manip_env.unlock_robot_arm_joints(manip_env.robot_w, q_robot_2)  #####
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
    assert robot is not None
    if robot == manip_env.robot_m:
        world_to_robot_base = manip_env.world_to_robot_m_base
    elif robot == manip_env.robot_w:
        world_to_robot_base = manip_env.world_to_robot_w_base
    else:
        raise ValueError('invalid robot!')
    
    trajectory_planner = manip_env.init_traj_planner(world_to_robot_base, clamp_by_human=False, q_H_init=None, q_R_init=None)
    trajectory_follower = manip_env.init_traj_follower(world_to_robot_base)
    
    # Step 1: initialize trajectory planner & follower
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=False, init_traj=[])
    env_pcd, arm_pcd, shoulder_pcd = manip_env.compute_env_pcd(other_robot, resolution=12, 
                                                               add_bed_padding=False, add_chest_padding=False,
                                                               exclude_robot_fingers=True)
    env_pcd = np.vstack((env_pcd, arm_pcd, shoulder_pcd))

    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    current_joint_angles = q_robot_init
    world_to_eef = manip_env.bc.getLinkState(robot.id, robot.eef_id)[:2]

    # robot (wiping): attach tool to planner and follower
    if robot == manip_env.robot_w:
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

    # robot (wiping): attach tool to planner and follower
    if robot == manip_env.robot_w:
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
