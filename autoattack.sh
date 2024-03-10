#!/bin/bash

echo 'Submitting SBATCH jobs...'

################### Define a few global run parameters #######################
time="4:00:00"
ram="16G" # Amount of RAM
vram="32gb" # Amount of GPU memory
num_gpus="4" # Number of GPUs

# Modify this according to your own directory structure!
project_path="/home/mila/c/cristian-dragos.manta/adversarial-ml/DM-Improves-AT"
anaconda_path="/home/mila/c/cristian-dragos.manta/anaconda3/bin/activate"
python_env_name="adversarial-ml"
##############################################################################

# Boilerplate
job_setup () {
    # Job resources requirements
    echo "#!/bin/bash" >> temprun.sh
    echo "#SBATCH --partition=long"  >> temprun.sh
    echo "#SBATCH --cpus-per-task=2" >> temprun.sh
    echo "#SBATCH --gres=gpu:$vram:$num_gpus" >> temprun.sh
    echo "#SBATCH --mem=$ram" >> temprun.sh
    echo "#SBATCH --time=$time " >>  temprun.sh
    echo "#SBATCH -o $project_path/slurm/slurm-%j.out" >> temprun.sh

    # Environment setup
    echo "module purge" >> temprun.sh
    echo "module load cuda/11.1/cudnn/8.1" >> temprun.sh
    echo "source $anaconda_path" >> temprun.sh
    echo "conda activate $python_env_name" >> temprun.sh
    echo "export PYTHONPATH=${PYTHONPATH}:$project_path" >> temprun.sh
}

table7_only_real_data () {
    # Experiment name
    experiment_name_base="table7_only_real_data"

    # Augmentations
    augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh")
    # augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh" "cutmix")

    for augment in "${augments[@]}"
    do
        experiment_name="$experiment_name_base-augment-$augment"

        job_setup    
        echo "python eval-aa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name" >> temprun.sh

        eval "sbatch temprun.sh"
        rm temprun.sh
    done
}

table7_only_generated_data () {
    # Experiment name
    experiment_name_base="table7_only_generated_data"

    # Augmentations
    augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh")
    # augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh" "cutmix")

    for augment in "${augments[@]}"
    do
        experiment_name="$experiment_name_base-augment-$augment"

        job_setup    
        echo "python eval-aa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name" >> temprun.sh

        eval "sbatch temprun.sh"
        rm temprun.sh
    done
}

table7_mix_generated_real () {
    # Experiment name
    experiment_name_base="table7_mix_generated_real"

    # Augmentations
    augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh")
    # augments=("none" "base" "cutout" "autoaugment" "randaugment" "idbh" "cutmix")

    for augment in "${augments[@]}"
    do
        experiment_name="$experiment_name_base-augment-$augment"

        job_setup    
        echo "python eval-aa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name" >> temprun.sh

        eval "sbatch temprun.sh"
        rm temprun.sh
    done
}

table7_only_real_data
table7_only_generated_data
table7_mix_generated_real