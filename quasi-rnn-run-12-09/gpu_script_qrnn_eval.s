#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=qrnn-eval
#SBATCH --mail-type=END
#SBATCH --mail-user=cer446@nyu.edu
#SBATCH --output=qrnn_output_eval.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module purge

module load numpy/intel/1.13.1

module load pytorch/python2.7/0.3.0_4

python2.7 eval_set_params.py 

