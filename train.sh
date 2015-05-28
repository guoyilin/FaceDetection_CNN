#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/caffe train --solver=examples/face_detection_yahoo/alexNet/solver.prototxt --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --gpu 0 2>&1 | tee examples/face_detection_yahoo/alexNet/log
