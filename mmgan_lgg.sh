#!/usr/bin/env bash
python train_mmgan_brats2018.py --grade=LGG --train_patient_idx=70 --test_pats=5 --batch_size=4 --dataset=BRATS2018 --n_epochs=60 --model_name=mmgan_lgg_zeros_cl --log_level=info --n_cpu=8 --c_learning=1 --z_type=zeros
