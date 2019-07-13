exp=${1:-efnet}

mkdir -p ckpts/$exp
rm -rf ckpts/$exp/*
cp config.py ckpts/$exp/config.py

OPENCV_IO_ENABLE_JASPER=1 python3 train.py --exp=$exp |& tee ckpts/$exp/log.txt
