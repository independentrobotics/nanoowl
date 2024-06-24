TARGET=$1
IR_UTILS=$HOME/code/ir_utils/
NANOOWL=$HOME/code/nanoowl/

docker build --tag nanoowl --file ./nanoowl.Dockerfile \
    --target=$TARGET \
    --build-context ir_utils=$IR_UTILS \
    --build-context nanoowl=$NANOOWL \
    .