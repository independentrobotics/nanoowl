docker run \
    -it \
    --env DISPLAY=localhost:10.0 \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --volume="$HOME/.config/independent-robotics.yaml:/root/.config/independent-robotics.yaml" \
    --volume="$HOME/IndependentRobotics:/root/IndependentRobotics" \
    --volume="$1:/root/out" \
    nanoowl:latest \
    /bin/bash