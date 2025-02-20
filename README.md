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
docker run -it --rm \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    assistive-manip-env /bin/bash
```

* download the h36m dataset from here: https://drive.google.com/file/d/1lGbtOsasw5F2MjvwWd9AtzCXdIefvmpv/view?usp=sharing


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