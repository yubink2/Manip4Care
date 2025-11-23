import time
import numpy as np
from manip4care.envs.utils.distance_utils import (
    get_human_to_robot_dist, 
	get_bed_to_robot_dist, 
	get_cube_to_robot_dist, 
	get_robot_to_robot_2_dist
)

def generate_random_q_H(env):
    q_H_0 = np.random.uniform(env.human_arm_lower_limits[0], env.human_arm_upper_limits[0])
    q_H_1 = np.random.uniform(env.human_arm_lower_limits[1], env.human_arm_upper_limits[1])
    q_H_2 = np.random.uniform(env.human_arm_lower_limits[2], env.human_arm_upper_limits[2])
    q_H_3 = np.random.uniform(env.human_arm_lower_limits[3], env.human_arm_upper_limits[3])

    return [q_H_0, q_H_1, q_H_2, q_H_3] 

def human_in_collision(env):
    """Check if any part of the human arm collides with other objects."""
    contact_points = env.bc.getContactPoints(bodyA=env.humanoid._humanoid, physicsClientId=env.bc._client)
    for point in contact_points:
        if (point[2] in [env.bed_id]):
            return True
    return False

def get_valid_q_H(env):
	"""Reset the human arm and check for collisions until no collision is detected."""
	for _ in range(5000):
		q_H = generate_random_q_H(env)
		env.reset_human_arm(q_H)
		env.bc.stepSimulation(physicsClientId=env.bc._client)
		if not human_in_collision(env):
			env.lock_human_joints(q_H)
			world_to_elbow = env.bc.getLinkState(env.humanoid._humanoid, env.elbow)[:2]
			return q_H, world_to_elbow
	raise ValueError('valid human config not found!')

def get_new_human_robot_configs(env, q_H_init, q_robot, q_robot_2):
	valid_grasp = False
	while True:
		env.reset_robot(env.robot_w, q_robot_2)
		q_H_goal, _ = get_valid_q_H(env)
		q_R_goal = env.get_q_R_from_elbow_pose(prev_q_R=q_robot)
		valid_grasp = env.validate_q_R(q_H_goal, q_R_goal, check_goal=True)
		if valid_grasp:
			break
	return q_H_goal, q_R_goal


def validate_q_R(env, q_H, q_R, check_goal=False):
	"""Validate a manip robot configuration with respect to human and obstacles.

	Returns:
	    True if valid
	"""
	# joint limits
	if min(q_R) < min(env.robot_m.arm_lower_limits) or max(q_R) > max(env.robot_m.arm_upper_limits):
		return False

	env.reset_robot(env.robot_m, q_R)
	env.robot_w.reset()
	env.reset_human_arm(q_H)

	world_to_elbow = env.bc.getLinkState(env.humanoid._humanoid, env.elbow)[:2]
	world_to_cp = env.bc.multiplyTransforms(world_to_elbow[0], world_to_elbow[1],
											 env.elbow_to_cp[0], env.elbow_to_cp[1])
	cp_to_world = env.bc.invertTransform(world_to_cp[0], world_to_cp[1])
	eef_to_world = env.bc.multiplyTransforms(env.eef_to_cp[0], env.eef_to_cp[1],
											 cp_to_world[0], cp_to_world[1])
	world_to_eef = env.bc.invertTransform(eef_to_world[0], eef_to_world[1])

	# collision check
	if env.robot_m_in_collision(q_R):
		return False

	# reachability check
	eef_pose = env.bc.getLinkState(env.robot_m.id, env.robot_m.eef_id)[:2]
	dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
	if dist > 0.03:
		return False

	if check_goal:
		# distance to human check
		dist = get_human_to_robot_dist(env, env.robot_m, q_H, q_R)
		if dist <= 0.05:
			return False

		# distance check from robot to bed
		dist = get_bed_to_robot_dist(env, env.robot_m, q_R)
		if dist <= 0.05:
			return False

	return True


def validate_q_robot_2(env, q_H, q_robot, q_robot_2):
	"""Validate wiping robot (robot_w) configuration relative to other scene elements.
	
	Returns:
	    True if valid
	"""
	# feasibility check
	if min(q_robot_2) < min(env.robot_w.arm_lower_limits) or max(q_robot_2) > max(env.robot_w.arm_upper_limits):
		return False

	env.reset_robot(env.robot_w, q_robot_2)
	env.reset_robot(env.robot_m, q_robot)
	env.reset_human_arm(q_H)

	# distance check from human
	dist = get_human_to_robot_dist(env, env.robot_w, q_H, q_robot_2)
	if dist <= 0.05:
		return False

	# distance check from bed
	dist = get_bed_to_robot_dist(env, env.robot_w, q_robot_2)
	if dist <= 0.05:
		return False

	# distance check from cube
	dist = get_cube_to_robot_dist(env, env.robot_w, q_robot_2, env.cube_m_id)
	if dist <= 0.05:
		return False

	# distance check from manip robot
	dist = get_robot_to_robot_2_dist(env, q_robot, q_robot_2)
	if dist <= 0.05:
		return False

	return True

##################################################################################################

def find_goal_configs(
    env,
    q_H_init,
    q_robot,
    q_robot_2,
    n_samples=100,
    time_out=60,
    existing_trajs=None,
    return_all=False,
    score_threshold=None
):
    """
    Sample and/or score goal configs.

    If existing_trajs is provided, will only score and return the best.
    If return_all is True, returns all valid configs.
    Otherwise, returns the best config (highest score).

    Returns:
        If return_all:
            q_H_trajs, q_R_trajs, q_H_goals, q_R_goals
        Else:
            q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal
    """
    if existing_trajs is not None:
        # Only score and return the best from provided trajectories
        q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = existing_trajs
    else:
        # Sample valid configs
        q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = [], [], [], []
        count = 0
        start_time = time.time()
        while True:
            if count >= n_samples:
                break
            if time.time() - start_time >= time_out:
                break
            env.reset_robot(env.robot_w, q_robot_2)
            q_H_goal, _ = get_valid_q_H(env)
            q_H_traj, q_R_traj = env.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=q_robot)
            q_R_goal = q_R_traj[-1]
            valid_grasp = (
                env.validate_q_R(q_H=q_H_goal, q_R=q_R_goal, check_goal=True)
                and env.is_not_discontinuous(q_old=q_R_traj[0], q_new=q_R_traj[len(q_R_traj)//2])
                and env.is_not_discontinuous(q_old=q_R_traj[len(q_R_traj)//2], q_new=q_R_traj[-1])
            )
            if valid_grasp:
                q_H_trajs.append(q_H_traj)
                q_R_trajs.append(q_R_traj)
                q_H_goals.append(q_H_goal)
                q_R_goals.append(q_R_goal)
                count += 1

    if return_all:
        return q_H_trajs, q_R_trajs, q_H_goals, q_R_goals

    # Score and return the best
    q_H_scores = []
    for q_H_goal, q_R_goal in zip(q_H_goals, q_R_goals):
        env.lock_human_joints(q_H_goal)
        env.lock_robot_arm_joints(env.robot_m, q_R_goal)
        score = env.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal, q_robot=q_R_goal)
        q_H_scores.append(score)

    # Optionally filter by score threshold
    if score_threshold is not None:
        filtered = [
            (s, h_traj, r_traj, h_goal, r_goal)
            for s, h_traj, r_traj, h_goal, r_goal in zip(q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals)
            if s >= score_threshold
        ]
        if not filtered:
            raise ValueError("No configs above score threshold")
        q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = zip(*filtered)

    # Sort and return the best
    combined = list(zip(q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals))
    best = max(combined, key=lambda x: x[0])
    return best  # q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal