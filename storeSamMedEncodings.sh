#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=15:mem=120gb
#PBS -N sammed_encode_octmnist_28_3

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# run experiments
python storeSamMedEncodings.py -r "/rds/general/user/kc2322/home/" -d "octmnist" -i 28 -s 77221
