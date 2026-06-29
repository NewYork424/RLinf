# robot(nuc)
# export RLINF_NODE_RANK=3
export RLINF_NODE_RANK=1
# export RLINF_NODE_RANK=2
# export RLINF_NODE_RANK=4
export RLINF_COMM_NET_DEVICES=enp3s0
# export RLINF_COMM_NET_DEVICES=rlinf
export FRANKA_ROBOT_IP=192.168.1.2
unset PYTHONPATH
export PYTHONPATH=/home/abc/nieyi/rlinf/ #:$PYTHONPATH # python path on nuc. 
source ~/catkin_ws/devel/setup.bash
#source /home/abc/wangyitao/RLinf/.venv/bin/activate
source /home/abc/nieyi/rlinf/.venv/bin/activate
# source ../../chenyinuo/RLinf/.venv/bin/activate

# ray start --address="172.16.88.68:6379"
# ray start --address=10.126.126.101:6379
# ray start --address='172.16.88.14:6379'

