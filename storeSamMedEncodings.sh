#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=15:mem=120gb
#PBS -N sammed_encode_chestmnist_128

# bash script to run generalisation experiments on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# run experiments
python storeSamMedEncodings.py -r "/rds/general/user/kc2322/home/" -d "chestmnist" -i 128 -s 57400