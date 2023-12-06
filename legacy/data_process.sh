#!/bin/bash
#SBATCH --job-name=data_processing
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1 
#SBATCH -c 16
#SBATCH --output=process_data.%j.out
#SBATCH --error=process_data.%j.err
#SBATCH --partition=cpu
#SBATCH --qos=nopreemption
#SBATCH --open-mode=append
#SBATCH --time=04:00:00

python data_preprocess.py