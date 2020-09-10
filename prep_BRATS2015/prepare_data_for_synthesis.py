"""
========================================================================
            Prepare BRATS 2013 Validation data for MM_Synthesis
========================================================================
AUTHOR: Anmol Sharma
Description: This file prepares the BRATS 2013 validation set, which is
             divided into challenge and leaderboard, and further divided
             into the grade types, into NPZ format that the mm_synthesis
             module expects.

             This is similar to how BRATS 2018 data was prepared. Check
             the original mm_synthesis codebase for more info.

"""

import os, sys, h5py
sys.path.append('..')
from modules.configfile import config, mount_path_prefix
import numpy as np

def saveNPZ(data, save_path, pat_names):
    np.save(open(os.path.join(save_path, 'pat_names_validation.npz'), 'wb'), pat_names)
    t1 = data[:,0,...]
    t1 = np.swapaxes(t1, 3, 2)
    t1 = np.swapaxes(t1, 2, 1)
    np.save(open(save_path + 'T1.npz', 'wb'), t1)
    del t1

    t2 = data[:,1,...]
    t2 = np.swapaxes(t2, 3, 2)
    t2 = np.swapaxes(t2, 2, 1)
    np.save(open(save_path + 'T2.npz', 'wb'), t2)
    del t2

    t1ce = data[:,2,...]
    t1ce = np.swapaxes(t1ce, 3, 2)
    t1ce = np.swapaxes(t1ce, 2, 1)
    np.save(open(save_path + 'T1CE.npz', 'wb'), t1ce)
    del t1ce

    t2flair = data[:,3,...]
    t2flair = np.swapaxes(t2flair, 3, 2)
    t2flair = np.swapaxes(t2flair, 2, 1)
    np.save(open(save_path + 'T2FLAIR.npz', 'wb'), t2flair)
    del t2flair

    print('Done!')

hf = h5py.File(config['hdf5_filepath_prefix'], 'r')
hf = hf['validation_data']

# SAVE CHALLENGE HGG Data
save_path_c_hg = os.path.join(mount_path_prefix, "scratch/asa224/Datasets/BRATS2015/mm_synthesis/validation_data/testing_hgglgg_patients/HGG_LGG/")
pat_names_c_hg = hf['testing_hgglgg_patients_pat_name']
challenge_data_hgg = hf['testing_hgglgg_patients']
saveNPZ(challenge_data_hgg, save_path_c_hg, pat_names_c_hg)
