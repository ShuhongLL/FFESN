#!/bin/bash

if [ ! -d "./log" ]; then
    mkdir ./log
fi

for f in $(seq 0.25 0.25 10); do
    name="lorenz_f$f"
    nohup python -u main.py --f $f > ./log/$name.log &
done