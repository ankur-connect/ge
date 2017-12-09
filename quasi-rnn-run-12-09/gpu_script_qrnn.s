#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=qrnn
#SBATCH --mail-type=END
#SBATCH --mail-user=cer446@nyu.edu
#SBATCH --output=qrnn_output.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module purge

module load numpy/intel/1.13.1

module load pytorch/python2.7/0.3.0_4


python2.7 train.py --src_vocab vocab.p \
--tgt_vocab vocab.p \
--src_train train_source_seqs.txt \
--tgt_train train_target_seqs.txt \
--src_valid dev_source_seqs.txt \
--tgt_valid dev_target_seqs.txt \
--hidden_size 256
