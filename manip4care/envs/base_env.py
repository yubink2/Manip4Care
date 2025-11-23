import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
from pybullet_utils.bullet_client import BulletClient

# Agents
from agents.humanoid_supine import Humanoid
from agents.humanoid_sitting import HumanoidSeated
from agents.pybullet_ur5.robot import UR5Robotiq85

# Utils
from utils.transform_utils import compute_matrix
from utils.point_cloud_utils import *
from wiping_task.util import Util


class BaseEnv:
    def __init__(self, gui=False, seated=False, wiping=False):
        self.gui = gui
        self.seated = seated
        self.wiping = wiping
        self.bc = BulletClient(connection_mode=p.GUI if self.gui else p.DIRECT)
        self.util = Util(self.bc._client)

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, 0)
        self.bc.setTimestep = 0.05

        # Load environment
        self.plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04))
        (bed_urdf, bed_pos, bed_scale) = (
            ("manip4care/envs/urdf/wheelchair.urdf", (0, 0, 0), 0.8) if self.seated else
            ("manip4care/envs/urdf/bed_0.urdf", (0.0, -0.1, 0.0), 1.0)
        )
        self.bed_id = self.bc.loadURDF(bed_urdf, bed_pos, globalScaling=bed_scale, useFixedBase=True)
        
        # Load human
        (human_base_pos, human_base_orn, self.human_rest_poses, HumanoidClass) = (
            ((0, 0, 0.4), (1.57, 0, 1.57), [2.0, -0.5, -1.8, 1.0], HumanoidSeated) if self.seated else
            ((0, 0, 0.3), (0, 1.57, 0), [2.47908, -0.016423, -1.81284, 0.452919], Humanoid)
        )
        human_base_orn = self.bc.getQuaternionFromEuler(human_base_orn)
        self.human_base_pose = (human_base_pos, human_base_orn)
        self.humanoid = HumanoidClass(self.bc, human_base_pos, human_base_orn)

        # Human joint indices
        self.chest, self.shoulder, self.elbow, self.wrist = 2, 6, 7, 8
        self.human_controllable_joints = [3, 4, 5, 7]
        self.human_arm = [3, 4, 5, 6, 7, 8]

        # Human arm joint limits
        shoulder_min, shoulder_max = [-3.14156, -1.000313, -3.14105], [3.14156, 1.46972, 3.14105]
        elbow_min, elbow_max = [0.0], [2.85735]
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max
        self.human_arm_joint_ranges = list(np.array(self.human_arm_upper_limits) - np.array(self.human_arm_lower_limits))

        # Human base transform
        human_base = self.bc.getBasePositionAndOrientation(self.humanoid._humanoid)[:2]
        self.T_world_to_human_base = compute_matrix(human_base[0], human_base[1])

        self.robot_w = None
        self.cube_w_id = None
        self.world_to_robot_w_base = None

        self.create_manip_robot()
        if self.wiping:
            self.create_wiping_robot()

    def create_manip_robot(self):
        # Load first robot (manipulation)
        if self.seated:
            self.robot_m_base_pose = ((0.3, -0.6, 0.65), (0, 0, 0))
        else:
            self.robot_m_base_pose = ((0.5, 0.8, 0.25), (0, 0, 0))
        self.cube_m_id = self.bc.loadURDF("manip4care/envs/urdf/cube_0.urdf", 
                                   (self.robot_m_base_pose[0][0], self.robot_m_base_pose[0][1], self.robot_m_base_pose[0][2]-0.15), useFixedBase=True)
        self.world_to_robot_m_base = compute_matrix(translation=self.robot_m_base_pose[0], rotation=self.robot_m_base_pose[1], rotation_type='euler')
        self.robot_m = UR5Robotiq85(self.bc, self.robot_m_base_pose[0], self.robot_m_base_pose[1])
        self.robot_m.load()
        for _ in range(50):
            self.robot_m.reset()
            self.robot_m.open_gripper()

    def create_wiping_robot(self):
        # Load second robot (wiping)
        robot_w_base_pose = ((0.65, 0, 0.25), (0, 0, 1.57))
        self.cube_w_id = self.bc.loadURDF("manip4care/envs/urdf/cube_0.urdf", 
                            (robot_w_base_pose[0][0], robot_w_base_pose[0][1], robot_w_base_pose[0][2]-0.15), useFixedBase=True)
        self.world_to_robot_w_base = compute_matrix(translation=robot_w_base_pose[0], rotation=robot_w_base_pose[1], rotation_type='euler')
        self.robot_w = UR5Robotiq85(self.bc, robot_w_base_pose[0], robot_w_base_pose[1])
        self.robot_w.load()
        self.robot_w.reset()

    #################### CONTROL ####################
    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i])
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.resetJointState(robot.id, j, 0.0)

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i])
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.setJointMotorControl2(robot.id, j, p.POSITION_CONTROL, 0.0)

    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def move_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.setJointMotorControl2(self.humanoid._humanoid, j, p.POSITION_CONTROL, q_human[i], physicsClientId=self.bc._client)

    def get_robot_joint_angles(self, robot):
        current_joint_angles = []
        for joint_id in robot.arm_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(robot.id, joint_id)[0])
        return current_joint_angles

    def get_human_joint_angles(self):
        current_joint_angles = []
        for joint_id in self.human_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(self.humanoid._humanoid, joint_id)[0])
        return current_joint_angles

    def make_human_zero_mass(self):
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0.00001, physicsClientId=self.bc._client)
        
    def lock_human_joints(self, q_human):
        # Save original mass of each joint to restore later
        self.human_joint_masses = []
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            # Get the current dynamics info to save mass
            dynamics_info = self.bc.getDynamicsInfo(self.humanoid._humanoid, j, physicsClientId=self.bc._client)
            self.human_joint_masses.append(dynamics_info[0])  # Save mass (first item in tuple is mass)
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], targetVelocity=0, physicsClientId=self.bc._client)

    def lock_robot_arm_joints(self, robot, q_robot):
        # Save original mass of each joint to restore later
        self.robot_joint_masses = []
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            dynamics_info = self.bc.getDynamicsInfo(robot.id, j, physicsClientId=self.bc._client)
            self.robot_joint_masses.append(dynamics_info[0])  # Save mass
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(robot.id, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, jointIndex=joint_id, targetValue=q_robot[i], targetVelocity=0, physicsClientId=self.bc._client)
        
    def lock_robot_gripper_joints(self, robot):
        # Set arm joints velocities to 0
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.resetJointState(robot.id, jointIndex=j, targetValue=0, targetVelocity=0, physicsClientId=self.bc._client)

    def unlock_human_joints(self, q_human):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            original_mass = self.human_joint_masses[j]
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], physicsClientId=self.bc._client)

    def unlock_robot_arm_joints(self, robot, q_robot):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            original_mass = self.robot_joint_masses[j]
            self.bc.changeDynamics(robot.id, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, jointIndex=joint_id, targetValue=q_robot[i], physicsClientId=self.bc._client)
    
    def reset_base_pose(self, obj_id, base_pos, base_orn):
        assert obj_id is not None
        self.bc.resetBasePositionAndOrientation(obj_id, posObj=base_pos, ornObj=base_orn)
    #################### CONTROL ####################

    #################### POINT CLOUD ####################
    def visualize_point_cloud(self, pcd):
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([pc_ply])

    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            if obstacle == self.bed_id:
                half_extents = [0.5, 1.7, 0.4]
                point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle, half_extents, client_id=self.bc._client, resolution=50))
            else:
                point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle, client_id=self.bc._client))
        return np.array(point_cloud)
    
    def get_bed_point_cloud(self, bed_id, add_padding=False):
        if add_padding:
            half_extents = [0.5, 1.7, 0.4]
            bed_pcd = get_point_cloud_from_collision_shapes(bed_id, half_extents, client_id=self.bc._client, resolution=50)
        else:
            half_extents = [0.5, 1.7, 0.2]
            bed_pcd = get_point_cloud_from_collision_shapes(bed_id, half_extents, client_id=self.bc._client, resolution=30)
        
        return bed_pcd

    def get_robot_point_cloud(self, robot, num_joints=None):
        robot_pcd = get_point_cloud_from_collision_shapes(robot.id, client_id=self.bc._client, num_joints=num_joints)
        upper_arm_link = self.bc.getLinkState(robot.id, 2)[:2]
        forearm_link = self.bc.getLinkState(robot.id, 3)[:2]
        upper_arm_pcd = generate_capsule_vertices(radius=0.04, height=0.3, position=upper_arm_link[0], 
                                                  orientation=upper_arm_link[1], client_id=self.bc._client)
        forearm_pcd = generate_capsule_vertices(radius=0.04, height=0.27, position=forearm_link[0], 
                                                orientation=forearm_link[1], client_id=self.bc._client)
        pcd = np.vstack((robot_pcd, upper_arm_pcd, forearm_pcd))
        return pcd
    
    def compute_obj_pcd(self, obj_id, resolution=8):
        obj_pcd = get_point_cloud_from_collision_shapes(obj_id, client_id=self.bc._client, resolution=resolution)
        return obj_pcd
    #################### POINT CLOUD ####################

    #################### GRASP PARAMETERS ####################
    def compute_grasp_parameters(self, q_H, q_R_grasp, grasp):
        # compute elbow_to_cp
        self.reset_human_arm(q_H)
        world_to_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.elbow)[:2]
        world_to_cp = (grasp[0], world_to_elbow[1])
        elbow_to_world = self.bc.invertTransform(world_to_elbow[0], world_to_elbow[1])
        elbow_to_cp = self.bc.multiplyTransforms(elbow_to_world[0], elbow_to_world[1],
                                                       world_to_cp[0], world_to_cp[1])
        cp_to_elbow = self.bc.invertTransform(elbow_to_cp[0], elbow_to_cp[1])

        # compute elbow_joint_to_cp
        world_to_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.elbow)[4:6]
        elbow_joint_to_world = self.bc.invertTransform(world_to_elbow_joint[0], world_to_elbow_joint[1])
        elbow_joint_to_cp = self.bc.multiplyTransforms(elbow_joint_to_world[0], elbow_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        T_elbow_joint_to_cp = compute_matrix(translation=elbow_joint_to_cp[0], rotation=elbow_joint_to_cp[1])

        # compute wrist_joint_to_cp
        world_to_wrist_joint = self.bc.getLinkState(self.humanoid._humanoid, self.wrist)[4:6]
        wrist_joint_to_world = self.bc.invertTransform(world_to_wrist_joint[0], world_to_wrist_joint[1])
        wrist_joint_to_cp = self.bc.multiplyTransforms(wrist_joint_to_world[0], wrist_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        
        # compute eef_to_cp
        self.reset_robot(self.robot_m, q_R_grasp)
        world_to_eef = self.bc.getLinkState(self.robot_m.id, self.robot_m.eef_id)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                               world_to_cp[0], world_to_cp[1])
        cp_to_eef = self.bc.invertTransform(eef_to_cp[0], eef_to_cp[1])
        
        self.elbow_to_cp = elbow_to_cp
        self.cp_to_elbow = cp_to_elbow
        self.T_elbow_joint_to_cp = T_elbow_joint_to_cp
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef

        self.elbow_joint_to_cp = elbow_joint_to_cp
        self.cp_to_elbow_joint = self.bc.invertTransform(elbow_joint_to_cp[0], elbow_joint_to_cp[1])

        self.wrist_joint_to_cp = wrist_joint_to_cp
        self.cp_to_wrist_joint = self.bc.invertTransform(wrist_joint_to_cp[0], wrist_joint_to_cp[1])

    def get_grasp_parameters(self):
        return (self.elbow_to_cp, self.cp_to_elbow,
                self.eef_to_cp, self.cp_to_eef,
                self.elbow_joint_to_cp, self.cp_to_elbow_joint,
                self.wrist_joint_to_cp, self.cp_to_wrist_joint)
    
    def set_grasp_parameters(self, elbow_to_cp, cp_to_elbow, eef_to_cp, cp_to_eef,
                             elbow_joint_to_cp, cp_to_elbow_joint, wrist_joint_to_cp, cp_to_wrist_joint):
        self.elbow_to_cp = elbow_to_cp
        self.cp_to_elbow = cp_to_elbow
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef
        self.elbow_joint_to_cp = elbow_joint_to_cp 
        self.cp_to_elbow_joint = cp_to_elbow_joint
        self.wrist_joint_to_cp = wrist_joint_to_cp
        self.cp_to_wrist_joint = cp_to_wrist_joint
    #################### GRASP PARAMETERS ####################

if __name__ == '__main__':
    env = BaseEnv(gui=True, wiping=False)
    env.create_world()
    env.create_manip_robot()
    env.create_wiping_robot()   
    print('here')