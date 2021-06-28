#!/usr/bin/env bash

AVAILABLE_GPU="0"
FOUND=0
sleep $[ ( $RANDOM % 10 ) + 1 ]s
FILEPATH=$1/environment/GPU_LOCKS/LOCKS
for N in $AVAILABLE_GPU
do
    if [[ "$FOUND" == 0 ]];
    then
        if [ ! -f $FILEPATH/LOCK_$N.lock ];
        then
            export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$N
            export CUDA_VISIBLE_DEVICES=$N
            printf 'LOCKED' > $FILEPATH/LOCK_$N.lock
            FOUND=1
        fi
    fi
done

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; 
then 
    echo "Error: no node available" > logfile.log
    exit 220
else 
    echo "CUDA_VISIBLE_DEVICES set to '$CUDA_VISIBLE_DEVICES'"; 
fi