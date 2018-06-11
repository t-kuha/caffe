#!/bin/sh

set -e

echo "First partition and run"

./build/tools/caffe train \
    --solver=./examples/INQ/alexnet/solver.prototxt \
    --weights=./models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    --gpu 0


echo "Second partition and run"

sed -i \"s/(count_\*0.7)/(count_\*0.4)/g\" ./src/caffe/blob.cpp
make all -j`nproc`

sed -i \"s/part1/part2/g\" ./examples/INQ/alexnet/solver.prototxt

./build/tools/caffe train \
    --solver=./examples/INQ/alexnet/solver.prototxt \
    --weights=./models/bvlc_alexnet/alexnet_part1_iter_63000.caffemodel \
#    --gpu 0


echo "Thrid partition and run"

sed -i \"s/(count_\*0.4)/(count_\*0.2)/g\" ./src/caffe/blob.cpp
make all -j`nproc`

sed -i \"s/part2/part3/g\" ./examples/INQ/alexnet/solver.prototxt

./build/tools/caffe train \
    --solver=./examples/INQ/alexnet/solver.prototxt \
    --weights=./models/bvlc_alexnet/alexnet_part2_iter_63000.caffemodel \
    --gpu 0


echo "Last partition and run"

sed -i \"s/(count_\*0.2)/(count_\*0.)/g\" ./src/caffe/blob.cpp
make all -j`nproc`

sed -i \"s/part3/part4/g\" ./examples/INQ/alexnet/solver.prototxt
sed -i \"s/snapshot: 3000/snapshot: 1/g\" ./examples/INQ/alexnet/solver.prototxt
sed -i \"s/max_iter: 63000/max_iter: 1/g\" ./examples/INQ/alexnet/solver.prototxt

./build/tools/caffe train \
    --solver=./examples/INQ/alexnet/solver.prototxt \
    --weights=./models/bvlc_alexnet/alexnet_part3_iter_63000.caffemodel \
#    --gpu 0


echo "All quantization done and you can enjoy the power-of-two weights!"