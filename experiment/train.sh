#!/usr/bin/env sh
TOOLS="/ssd/ijcai18/program/caffe/build/tools/caffe"
GPU="--gpu=0,1,2,3"
#GPU="--gpu=0"
now=$(date +"%Y%m%d_%H%M%S")
echo $TOOLS
GLOG_logtostderr=1 LD_LIBRARY_PATH=/usr/local/lib $TOOLS train $GPU -solver solver.prototxt 2>&1|tee log/train-$now.log
