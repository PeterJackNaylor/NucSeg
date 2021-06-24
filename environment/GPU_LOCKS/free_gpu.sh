#!/usr/bin/bash

FILEPATH=$1/environment/GPU_LOCKS/LOCKS

rm "$FILEPATH/LOCK_$CUDA_VISIBLE_DEVICES.lock"
