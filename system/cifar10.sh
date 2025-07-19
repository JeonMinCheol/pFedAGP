#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -J cifar10
#SBATCH -p batch_grad
#SBATCH -t 4-0
#SBATCH -o logs/cifar10-%A.out

for dcl in 0.1
do
    for algo in pFedAGP Ditto FedProto FedPAC FedMTL FedKD 
    do
    python -u main.py \
        -dcl $dcl \
        -lbs 256 \
        -nc 20 \
        -jr 0.5 \
        -nb 10 \
        -data Cifar10 \
        -m mobilenet_v2 \
        -algo $algo \
        -gr 100 \
        -go mobilenet_v2 \
        -ls 5 \
        -nw 8 \
        -eg 5 \
        -did 0 \
        -ags 3 \
        -entlam 1\
        -entt 0.4 \
        -suplam 0.5 \
        -temp 0.45 \
        -alr 0.001
    done
done
