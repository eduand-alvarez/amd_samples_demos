#!bin/bash

docker run --name r50_pytorch -v "$(pwd)":/var/lib/jenkins/working_dir/ --cap-add=SYS_PTRACE --device=/dev/dri --device=/dev/kfd --group-add video --ipc=host --network=host --privileged --rm --security-opt seccomp=unconfined -it rocm/pytorch
