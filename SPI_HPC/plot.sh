#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J spi_plot
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -o spi_plot.out
#SBATCH -e spi_plot.err
#SBATCH --mem=12G

for b in .2 .5 .85; do
    for d in 20 75 150 400; do
        python3 plot.py --env_type agaid --env intvn28_act4_prec1 --d_size $d --baseline $b --num_baselines 1 --num_d 1 --num_reps 40 --path /nfs/hpc/share/soloww/spi_hpc/ &
    done
done




