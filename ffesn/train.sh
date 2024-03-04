#!/bin/bash

if [ ! -d "./log" ]; then
    mkdir ./log
fi

iteration=5

for rho in $(seq 0.1 0.1 2.0); do
    name="ffesn_rho${rho}_iter${iteration}"
    nohup python -u main.py --rho $rho --iteration $iteration > ./log/$name.log &
done