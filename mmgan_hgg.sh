#!/usr/bin/env bash
python train_mmgan_brats2018.py --grade=HGG --train_patient_idx=200 --test_pats=10 --batch_size=4 --dataset=BRATS2018 --n_epochs=60 --model_name=mmgan_hgg_zeros_cl --log_level=info --n_cpu=4 --c_learning=1 --z_type=zeros
