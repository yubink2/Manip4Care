""" original code: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap """

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class HumanoidSeated(object):

  def __init__(self, pybullet_client, baseShift, ornShift=[0,0,0,1]):
    """Constructs a humanoid and reset it to the initial states.
    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
    """
    self._baseShift = baseShift
    self._ornShift = ornShift
    self._pybullet_client = pybullet_client
    
    self._humanoid = self._pybullet_client.loadURDF("urdf/humanoid_seated.urdf", [0, 0.9, 0],
                                                    globalScaling=0.2,
                                                    useFixedBase=True)

    self._pybullet_client.resetBasePositionAndOrientation(self._humanoid, self._baseShift, self._ornShift)
    self._pybullet_client.changeDynamics(self._humanoid, -1, linearDamping=0, angularDamping=0)
    
    # change colors of the human model limbs
    humanoid_color = [255/255, 160/255, 45/255, 1]
    for j in range(self._pybullet_client.getNumJoints(self._humanoid)):  
      ji = self._pybullet_client.getJointInfo(self._humanoid, j)
      if j == 2: 
        self._pybullet_client.changeVisualShape(self._humanoid, j, rgbaColor=[1, 0, 0, 1])
        continue
      self._pybullet_client.changeDynamics(self._humanoid, j, linearDamping=0, angularDamping=0)
      self._pybullet_client.changeVisualShape(self._humanoid, j, rgbaColor=humanoid_color)

    self._initial_state = self._pybullet_client.saveState()
    self._allowed_body_parts = [11, 14]
    
    self._contact_point = [0, 0, 0, 0]

