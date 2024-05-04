#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J spi_hpc400_.2
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -n 2
#SBATCH -o spi_hpc.out
#SBATCH -e spi_hpc.err
#SBATCH --mem=8G

python3 run.py --env_type agaid --env intvn28_act4_prec1 --d_size 400 --baseline .2 --num_baselines 2 --num_d 5 --num_reps 50 --path /nfs/hpc/share/soloww/spi_hpc/ 




