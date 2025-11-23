import numpy as np
import pybullet as p


def get_human_to_robot_dist(env, robot, q_H, q_robot, human_arm_indices=None):
    """Compute the minimum distance between a robot and the human body.

    Args:
        env: environment instance providing reset/reset_human_arm/reset_robot and bc
        robot: robot object (has id)
        q_H: human joint angles
        q_robot: robot joint angles
        human_arm_indices: list of human link indices to treat specially

    Returns:
        float: minimum distance (float('inf') if nothing found)
    """
    # reset human & robot in the simulation
    env.reset_human_arm(q_H)
    env.reset_robot(robot, q_robot)
    env.bc.stepSimulation()

    min_dist = float('inf')
    for c in p.getClosestPoints(bodyA=robot.id, bodyB=env.humanoid._humanoid, distance=100, physicsClientId=env.bc._client):
        linkA = c[3]
        linkB = c[4]

        if human_arm_indices is None:
            human_arm = getattr(env, 'human_arm', [])
        else:
            human_arm = human_arm_indices

        if robot == getattr(env, 'robot_w', None):
            if linkA >= 4 and linkB in human_arm:
                continue
        else:
            if linkB in human_arm:
                continue

        contact_distance = float(np.array(c[8]))
        if contact_distance < min_dist:
            min_dist = contact_distance

    return min_dist

def is_near_goal_W_space(world_to_eef, world_to_eef_goal, threshold=0.03):
    """Check if two end-effector poses in world-space are within a distance threshold.

    Args:
        world_to_eef: (pos, orn) pair where pos is (3,) and orn is quaternion
        world_to_eef_goal: same format for goal
        threshold: Euclidean distance threshold for position

    Returns:
        bool: True if within threshold
    """
    pos = world_to_eef[0]
    pos_goal = world_to_eef_goal[0]
    dist = np.linalg.norm(np.array(pos) - np.array(pos_goal))
    return dist <= threshold


def get_bed_to_robot_dist(env, robot, q_robot):
    """Minimum distance between robot and bed (skips gripper fingers for manip robot)."""
    env.reset_robot(robot, q_robot)
    env.bc.stepSimulation()
    min_dist = float('inf')
    for c in p.getClosestPoints(bodyA=robot.id, bodyB=env.bed_id, distance=100, physicsClientId=env.bc._client):
        linkA = c[3]
        # if robot is manip, skip gripper fingers
        if robot == getattr(env, 'robot_m', None):
            if linkA >= 9:
                continue

        contact_distance = float(np.array(c[8]))
        if contact_distance < min_dist:
            min_dist = contact_distance

    return min_dist


def get_cube_to_robot_dist(env, robot, q_robot, cube_id):
    """Minimum distance between robot and a cube object (skips gripper fingers for manip robot)."""
    env.reset_robot(robot, q_robot)
    env.bc.stepSimulation()
    min_dist = float('inf')
    for c in p.getClosestPoints(bodyA=robot.id, bodyB=cube_id, distance=100, physicsClientId=env.bc._client):
        linkA = c[3]
        if robot == getattr(env, 'robot_m', None):
            if linkA >= 9:
                continue

        contact_distance = float(np.array(c[8]))
        if contact_distance < min_dist:
            min_dist = contact_distance

    return min_dist


def get_robot_to_robot_2_dist(env, q_robot, q_robot_2):
    """Minimum distance between robot_w and robot_m (skipping gripper fingers)."""
    env.reset_robot(env.robot_m, q_robot)
    env.reset_robot(env.robot_w, q_robot_2)
    env.bc.stepSimulation()
    min_dist = float('inf')
    for c in p.getClosestPoints(bodyA=env.robot_w.id, bodyB=env.robot_m.id, distance=100, physicsClientId=env.bc._client):
        linkA = c[3]
        linkB = c[4]
        # skip gripper fingers
        if linkB >= 9:
            continue

        contact_distance = float(np.array(c[8]))
        if contact_distance < min_dist:
            min_dist = contact_distance

    return min_dist
