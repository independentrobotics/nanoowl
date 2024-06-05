docker run \
    -it \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --volume="$HOME/.config/independent-robotics.yaml:/root/.config/independent-robotics.yaml" \
    --volume="$HOME/IndependentRobotics:/root/IndependentRobotics" \
    --volume="$1:/root/out" \
    nanoowl:latest \
    /usr/bin/python3 /root/owl_predict.py --image /opt/nanoowl/assets/owl_glove_small.jpg "${@:2}"