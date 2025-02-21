## How to set up the environment

* clone the repository
```
git clone https://github.com/yubink2/AssistiveManipulation.git
```

* build the docker image
```
docker build -t assistive-manip-env .
```

* run the docker container
```
xhost +local:root
docker run -it \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    assistive-manip-env /bin/bash
```

* inside the container, install pytorch3d
```
FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
```

## How to run the limb manipulation pipeline

```
python manipulation_demo.py
```

```
python manipulation_seated_demo.py
```

## How to run the integrated bed bathing and limb manipulation pipeline

```
python wiping_manipulation_demo.py
```

## Acknowledgements

* We want to thank the authors of DeepMimic (https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap) and RAMP (https://github.com/SamsungLabs/RAMP) for their amazing work. 