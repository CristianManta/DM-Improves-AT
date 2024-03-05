#!/bin/bash

echo 'Submitting SBATCH jobs...'

################### Define a few global run parameters #######################
time="5:00:00"
ram="32G" # Amount of RAM
vram="32gb" # Amount of GPU memory

# Modify this according to your own directory structure!
project_path="/home/mila/c/cristian-dragos.manta/adversarial-ml/hacker-paper/DM-Improves-AT"
anaconda_path="/home/mila/c/cristian-dragos.manta/anaconda3/bin/activate"
python_env_name="adversarial-ml"
##############################################################################

# Boilerplate
job_setup () {
    # Job resources requirements
    echo "#!/bin/bash" >> temprun.sh
    echo "#SBATCH --partition=long"  >> temprun.sh
    echo "#SBATCH --cpus-per-task=2" >> temprun.sh
    echo "#SBATCH --gres=gpu:$vram:1" >> temprun.sh
    echo "#SBATCH --mem=$ram" >> temprun.sh
    echo "#SBATCH --time=$time " >>  temprun.sh
    echo "#SBATCH -o $project_path/slurm-%j.out" >> temprun.sh

    # Environment setup
    echo "module purge" >> temprun.sh
    echo "module load cuda/11.1/cudnn/8.1" >> temprun.sh
    echo "source $anaconda_path" >> temprun.sh
    echo "conda activate $python_env_name" >> temprun.sh
    echo "export PYTHONPATH=${PYTHONPATH}:$project_path" >> temprun.sh
}

test_exp () {
    job_setup
    echo "python train-wa.py --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'edm_data/cifar10-1m.npz' \
    --ls 0.1" >> temprun.sh

    eval "sbatch temprun.sh"
    rm temprun.sh
}

test_exp
