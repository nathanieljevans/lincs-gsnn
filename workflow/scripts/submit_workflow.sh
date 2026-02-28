#!/bin/zsh
#SBATCH --job-name=lincs-gsnn
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-gsnn/snakemake_%j.out
#SBATCH --error=/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-gsnn/snakemake_%j.err

source ~/.zshrc

cd /home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/lincs-gsnn/workflow/train/

conda activate lincs-gsnn

snakemake --unlock
snakemake -j 1 --rerun-incomplete
