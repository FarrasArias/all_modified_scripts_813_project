#!/bin/bash
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-72:00
#SBATCH --account=def-pasquier
#SBATCH --mail-user raa60@sfu.ca
#SBATCH --mail-type ALL

source $SCRATCH/mmm_training_with_distilgpt_011222/env_1024/bin/activate
module load StdEnv/2020 protobuf python/3.6.10
source $SCRATCH/mmm_training_with_distilgpt_011222/env_1024/bin/activate
cd $SCRATCH/mmm_training_with_distilgpt_011222/MMM_TRAINING-master/
python train.py --arch gpt2 --config config/distilgpt2.json --encoding EL_VELOCITY_DURATION_POLYPHONY_ENCODER --ngpu 4 --dataset /home/raa60/scratch/mmm_training_with_distilgpt_011222/data_v2_NUM_BARS\=4_OPZ_False.arr --batch_size 32 --label GENRE_DISCOGS --num_bars 4 --accum_steps 1 --ckpt ""
