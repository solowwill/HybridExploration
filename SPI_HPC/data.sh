#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J spi_plot
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -o spi_plot.out
#SBATCH -e spi_plot.err
#SBATCH --mem=8G


python3 data.py --env_type agaid --env intvn28_act4_prec1 --d_size 400 --baseline .85 --num_baselines 5 --num_d 50 --path /nfs/hpc/share/soloww/spi_hpc/





