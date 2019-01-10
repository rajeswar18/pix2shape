#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running pwdon $HOSTNAME

source activate vikky_pix2scene

cd /u/voletivi/GitHubRepos/pix2scene/diffrend/torch/GAN

# Cube obj
# python main.py --width 128 --height 128 --splats_img_size 128 --pixel_samples=1 --lr 2e-4 --name baseline_plots --disc_type cnn --cam_dist 0.8 --fovy 26 --batchSize 6 --gz_gi_loss 0.2 --est_normals --zloss 0.05  --unit_normalloss 0.0 --normal_consistency_loss_weight 10.0 --spatial_var_loss_weight 0.0 --grad_img_depth_loss 0.0 --spatial_loss_weight 0.0 --root_dir /data/lisa/data/pix2scene/obj/cube --out_dir /data/milatmp1/voletivi/pix2scene

# Sculpture obj
python main.py --width 128 --height 128 --splats_img_size 128 --pixel_samples=1 --lr 2e-4 --name baseline_plots --disc_type cnn --cam_dist 0.8 --fovy 26 --batchSize 6 --gz_gi_loss 0.2 --est_normals --zloss 0.05  --unit_normalloss 0.0 --normal_consistency_loss_weight 10.0 --spatial_var_loss_weight 0.0 --grad_img_depth_loss 0.0 --spatial_loss_weight 0.0 --root_dir /data/lisa/data/pix2scene/obj/sculptures/einstein_10000 --out_dir /data/milatmp1/voletivi/pix2scene

# RUN: sbatch --gres=gpu:p100-16gb -c 1 --mem=4000 --job-name eins_pix2scene pix2scene_run.sh
# READ: tail -f slurm-000000.out
# WATCH: watch -n 1 sacct -u voletivi
