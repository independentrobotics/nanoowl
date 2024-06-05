TARGET=$1
IR_UTILS=/home/michael/code/ir_utils/
NANOOWL=/home/michael/code/nanoowl/

docker build --tag nanoowl --file ./nanoowl.Dockerfile \
    --target=$TARGET \
    --build-context ir_utils=$IR_UTILS \
    --build-context nanoowl=$NANOOWL \
    .