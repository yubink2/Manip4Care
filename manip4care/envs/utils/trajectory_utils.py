import math


def is_not_discontinuous(q_old, q_new, angle_threshold=math.pi/2):
    """Check if the difference between two joint configurations is not discontinuous."""
    q_old = np.asarray(q_old)
    q_new = np.asarray(q_new)
    diff = q_new - q_old
    if np.any(np.abs(diff) > angle_threshold):
        return False
    return True
import numpy as np

def interpolate_trajectory(robot_traj, alpha=0.5):
    """Interpolate adjacent points in a robot trajectory.

    Args:
        robot_traj (list[list[float]]): list of joint-space waypoints
        alpha (float): interpolation factor between consecutive points

    Returns:
        list[list[float]]: new trajectory with interpolated points inserted
    """
    if robot_traj is None:
        return []

    if len(robot_traj) <= 1:
        return robot_traj

    new_traj = []
    for i in range(len(robot_traj) - 1):
        q_R_i = np.array(robot_traj[i])
        q_R_next = np.array(robot_traj[i + 1])

        interpolated_point = (1 - alpha) * q_R_i + alpha * q_R_next
        new_traj.append(robot_traj[i])  # current point
        new_traj.append(interpolated_point.tolist())  # interpolated point

    new_traj.append(robot_traj[-1])  # last point
    return new_traj

def get_init_traj_from_q_H(env, q_H_init, q_H_goal, q_R_init):
    """Return an initialization trajectory for human and robot.

    Returns:
        (q_H_traj, q_R_traj)
    """
    q_H_traj = []
    q_H_traj.append(q_H_init)
    q_H_traj.append(q_H_goal)
    q_H_traj = interpolate_trajectory(q_H_traj, 0.5)
    q_H_traj = interpolate_trajectory(q_H_traj, 0.5)

    q_R_traj = []
    q_R_traj.append(q_R_init)
    prev_q_R = q_R_init
    
    for q_H in q_H_traj[1:]:
        env.reset_human_arm(q_H)
        q_R = env.get_q_R_from_elbow_pose(prev_q_R)
        q_R_traj.append(q_R)
        prev_q_R = q_R

    return q_H_traj, q_R_traj


def get_q_R_from_elbow_pose(env, prev_q_R):
    """Compute robot IK for end-effector pose derived from human elbow->cp transform."""
    world_to_elbow_joint = env.bc.getLinkState(env.humanoid._humanoid, env.elbow)[4:6]
    world_to_cp = env.bc.multiplyTransforms(world_to_elbow_joint[0], world_to_elbow_joint[1],
                                            env.elbow_joint_to_cp[0], env.elbow_joint_to_cp[1])
    world_to_eef = env.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                             env.cp_to_eef[0], env.cp_to_eef[1])
    q_R_goal = env.bc.calculateInverseKinematics(env.robot_m.id, env.robot_m.eef_id, world_to_eef[0], world_to_eef[1],
                                                 env.robot_m.arm_lower_limits, env.robot_m.arm_upper_limits, env.robot_m.arm_joint_ranges,
                                                 restPoses=prev_q_R,
                                                 maxNumIterations=50)
    q_R_goal = [q_R_goal[i] for i in range(len(env.robot_m.arm_controllable_joints))]
    return q_R_goal


def get_eef_pose(env, robot, current_joint_angles, target_joint_angles):
    """Utility to set robot to target_joint_angles, read eef pose, then restore to current_joint_angles."""
    env.reset_robot(robot, target_joint_angles)
    eef_pose = env.bc.getLinkState(robot.id, robot.eef_id)[:2]
    env.reset_robot(robot, current_joint_angles)
    return eef_pose
