#!/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -J spi_hpc
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -n 10
#SBATCH -o spi_hpc.out
#SBATCH -e spi_hpc.err
#SBATCH --mem=8G

for i in {0..50}
do
   echo $i
done





