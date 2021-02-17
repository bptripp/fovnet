LOCAL_PORT=${1:-8080}
REMOTE_PORT=${2:-8889}
USER=${3:-p2torabi}
nohup ssh -N -L localhost:$LOCAL_PORT:localhost:$REMOTE_PORT $USER@brainlab1.uwaterloo.ca > .tunnel.out &
