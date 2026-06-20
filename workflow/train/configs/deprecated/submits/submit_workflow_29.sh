#!/bin/zsh
#SBATCH --job-name=lg-29
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-gsnn/snakemake__exp_29__%j.out
#SBATCH --error=/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-gsnn/snakemake__exp_29__%j.err

source ~/.zshrc

cd /home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/lincs-gsnn/workflow/train/

conda activate lincs-gsnn

# Pass --configfile for alternate configs; each run_id gets its own workdir
# under <dirs.runs>/<run_id>/.snakemake/ (see Snakefile workdir: directive).
# Run `snakemake --unlock` manually only when recovering from a crashed run.

snakemake -j 1 --configfile ./configs/config_exp_29.yaml --unlock
snakemake -j 1 --configfile ./configs/config_exp_29.yaml --forcerun train_gsnn --rerun-incomplete
