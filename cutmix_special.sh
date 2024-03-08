#!/bin/bash

echo 'Submitting SBATCH jobs...'

################### Define a few global run parameters #######################
time="50:00:00"
ram="16G" # Amount of RAM
vram="32gb" # Amount of GPU memory
num_gpus="6" # Number of GPUs

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
    # Data
    data="cifar10s"
    aux_data_filename="edm_data/cifar10-1m.npz"

    # Model
    experiment_name_base="table7_only_real_data"
    model="wrn-28-10-swish"

    # Other parameters
    epochs="127"
    unsup_fraction="0.002" # NOTE: This is the fraction of generated data per batch. Must not use exactly 0.0 since that would cause an error.

    augment="cutmix"
    experiment_name="$experiment_name_base-augment-$augment"

    resume_path="trained_models/$experiment_name"

    job_setup    
    echo "python train-wa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name --data $data --batch-size 512 --model $model --num-adv-epochs $epochs --lr 0.2 --beta 5.0 --unsup-fraction $unsup_fraction --aux-data-filename $aux_data_filename --ls 0.1 --CutMix --resume_path $resume_path" >> temprun.sh

    eval "sbatch temprun.sh"
    rm temprun.sh
}

table7_only_generated_data () {
    # Data
    data="cifar10s"
    aux_data_filename="edm_data/cifar10-48k.npz"

    # Model
    experiment_name_base="table7_only_generated_data"
    model="wrn-28-10-swish"

    # Other parameters
    epochs="200"
    unsup_fraction="0.999" # NOTE: This is the fraction of generated data per batch. Must not use exactly 1.0 since that would cause an error.

    augment="cutmix"
    experiment_name="$experiment_name_base-augment-$augment"

    job_setup    
    echo "python train-wa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name --data $data --batch-size 512 --model $model --num-adv-epochs $epochs --lr 0.2 --beta 5.0 --unsup-fraction $unsup_fraction --aux-data-filename $aux_data_filename --ls 0.1 --CutMix" >> temprun.sh

    eval "sbatch temprun.sh"
    rm temprun.sh
}

table7_mix_generated_real () {
    # Data
    data="cifar10s"
    aux_data_filename="edm_data/cifar10-1m.npz"

    # Model
    experiment_name_base="table7_mix_generated_real"
    model="wrn-28-10-swish"

    # Other parameters
    epochs="153"
    unsup_fraction="0.7" # NOTE: This is the fraction of generated data per batch.

    augment="cutmix"
    experiment_name="$experiment_name_base-augment-$augment"

    resume_path="trained_models/$experiment_name"

    job_setup    
    echo "python train-wa.py --data-dir dataset-data --log-dir trained_models --desc $experiment_name --data $data --batch-size 512 --model $model --num-adv-epochs $epochs --lr 0.2 --beta 5.0 --unsup-fraction $unsup_fraction --aux-data-filename $aux_data_filename --ls 0.1 --CutMix --resume_path $resume_path" >> temprun.sh

    eval "sbatch temprun.sh"
    rm temprun.sh
}

table7_only_real_data
table7_only_generated_data
table7_mix_generated_real