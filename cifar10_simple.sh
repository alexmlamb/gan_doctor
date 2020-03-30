#!/bin/bash

#cd ..

export RESULTS_DIR=results/gan_doctor/

#todo
export SLURM_TMP_DIR=/tmp

#export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-10.1/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}"

python3 task_launcher.py \
--gen=networks/cosgrove/gen.py \
--disc=networks/cosgrove/disc.py \
--disc_args="{'spec_norm': True, 'sigmoid': False, 'nf': 128}" \
--gen_args="{'nf': 128}" \
--gan=models/gan.py \
--gan_args="{'loss': 'hinge'}" \
--trial_id=99999999 \
--name=cifar10_simple_hinge_gan \
--val_batch_size=128 \
--batch_size=64 \
--z_dim=64 \
--img_size=32 \
--dataset=iterators/cifar10.py \
--compute_is_every=10 \
--n_samples_is=50000 \
--use_tf_metrics \
--subset_train=50000
