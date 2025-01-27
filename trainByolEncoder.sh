#!/bin/bash
#PBS -l walltime=71:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

python trainByolEncoder.py -r "/rds/general/user/kc2322/home/" -d $DATASET -i $IMAGE_SIZE