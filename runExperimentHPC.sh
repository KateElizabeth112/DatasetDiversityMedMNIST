#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=15:mem=10gb

# bash script to run generalisation experiments on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# install requirements
#pip install -r requirements.txt

# run experiments
python runExperiment.py -r "/rds/general/user/kc2322/home/" -n $NUM_SAMPLES -d $DATASET -i $IMAGE_SIZE
