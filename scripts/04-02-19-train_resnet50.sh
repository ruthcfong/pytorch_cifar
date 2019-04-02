#!/usr/bin/env bash

arch="ResNet50"
datadir="/scratch/local/ssd/ruthfong/cifar10"
checkpoint="/scratch/local/ssd/ruthfong/pytorch_cifar/checkpoints/${arch}_checkpoint.pth.tar"
batch_size=128
gpu=3

python main.py --datadir ${datadir} \
    --checkpoint ${checkpoint} \
    --arch ${arch} \
    --gpu ${gpu} \
    --batch_size ${batch_size}