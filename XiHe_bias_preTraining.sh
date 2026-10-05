#!/bin/bash
#SBATCH -J continueMkt10day_bais_surface_9day
#SBATCH --nodelist=gnode04
#SBATCH --nodes=1
#SBATCH --mem=400g
#SBATCH --cpus-per-task=4
#SBATCH -p zs
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/%j.loop
#SBATCH -e logs/%j.loop

#surface

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python src/climax/global_forecast/test_1to22.py --config configs/global_forecast_1to22_gpunode57_Xihe_bias.yaml --trainer.strategy=ddp --trainer.num_nodes=1 --trainer.devices=7 --trainer.max_epochs=200 --trainer.log_every_n_steps=10 --data.root_dir=/public/home/acct230421094230/ --data.predict_range=0 --data.batch_size=1 --data.num_workers=4 --data.num_nodes=1 --model.max_epochs=200 --model.lr=5e-5 --model.beta_1=0.9 --model.beta_2=0.95 --model.weight_decay=1e-5

#deep

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/climax/global_forecast/test_23to33.py --config configs/global_forecast_23to33_gpunode52_Xihe_bias.yaml --trainer.strategy=ddp_find_unused_parameters_true --trainer.num_nodes=1 --trainer.devices=8 --trainer.max_epochs=200 --trainer.log_every_n_steps=10 --data.root_dir=/public/home/xujianbo/all_data/2022_2023_mkt/ --data.predict_range=0 --data.batch_size=1 --data.num_workers=4 --data.num_nodes=1 --model.max_epochs=200 --model.lr=1e-4 --model.beta_1=0.9 --model.beta_2=0.95 --model.weight_decay=1e-5