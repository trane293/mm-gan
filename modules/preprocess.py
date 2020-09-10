"""
==========================================================
                Preprocess BRATS Data
==========================================================
AUTHOR: Anmol Sharma
AFFILIATION: Simon Fraser University
             Burnaby, BC, Canada
PROJECT: Analysis of Brain MRI Scans for Management of
         Malignant Tumors
COLLABORATORS: Anmol Sharma (SFU)
               Prof. Ghassan Hamarneh (SFU)
               Dr. Brian Toyota (VGH)
               Dr. Mostafa Fatehi (VGH)
DESCRIPTION: This file uses the previously generated data
             (using create_hdf5_file.py) and generates a
             new file  with similar structure, but after
             applying a couple of preprocessing steps.
             More specifically, the script applies the
             following operations on the data:

             1) Crop out the dark margins in the scans
                to only leave a concise brain area. For
                this a generous estimate of bounding box
                generated from the whole  dataset is used.
                For more  information, see checkLargestCropSize
                notebook.

             The code DOES NOT APPLY MEAN/VAR  normalization,
             but simply calculates the values and saves on disk.
             Check lines 140-143 for more information.

             The saved mean/var files are to be used before
             the training process.

LICENCE: Proprietary for now.
"""

import h5py
from modules.configfile import config
import numpy as np
import SimpleITK as sitk
import optparse
import logging
# from modules.mischelpers import *
from modules.dataloader import standardize
import os

logging.basicConfig(level=logging.DEBUG)

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

logger.warning('[IMPORTANT] The code DOES NOT APPLY mean/var normalization, rather it calculates it and saves to disk')
# ------------------------------------------------------------------------------------
# open existing datafile
# ------------------------------------------------------------------------------------
logger.info('opening previously generated HDF5 file.')

# open the existing datafile
hdf5_file_main = h5py.File(config['hdf5_filepath_prefix'], 'r')

logger.info('opened HDF5 file at {}'.format(config['hdf5_filepath_prefix']))

# get the group identifier for original dataset
hdf5_file = hdf5_file_main['original_data']

# ====================================================================================

# ------------------------------------------------------------------------------------
# create new HDF5 file to hold cropped data.
# ------------------------------------------------------------------------------------
logger.info('creating new HDF5 dataset to hold cropped/normalized data')
filename = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS_Cropped_Normalized_Unprocessed.h5')
new_hdf5 = h5py.File(filename, mode='w')
logger.info('created new database at {}'.format(filename))

# create a folder group to  hold the datasets. The schema is similar to original one except for the name of the folder
# group
new_group_preprocessed = new_hdf5.create_group('preprocessed')

# create similar datasets in this file.
new_group_preprocessed.create_dataset("training_data_hgg", config['train_shape_hgg_crop'], np.float32)
new_group_preprocessed.create_dataset("training_data_hgg_pat_name", (config['train_shape_hgg_crop'][0],), dtype="S100")
new_group_preprocessed.create_dataset("training_data_segmasks_hgg", config['train_segmasks_shape_hgg_crop'], np.int16)

new_group_preprocessed.create_dataset("training_data_lgg", config['train_shape_lgg_crop'], np.float32)
new_group_preprocessed.create_dataset("training_data_lgg_pat_name", (config['train_shape_lgg_crop'][0],), dtype="S100")
new_group_preprocessed.create_dataset("training_data_segmasks_lgg", config['train_segmasks_shape_lgg_crop'], np.int16)

new_group_preprocessed.create_dataset("validation_data", config['val_shape_crop'], np.float32)
new_group_preprocessed.create_dataset("validation_data_pat_name", (config['val_shape_crop'][0],), dtype="S100")
# ====================================================================================

# just copy the patient  names directly
new_group_preprocessed['training_data_hgg_pat_name'][:] = hdf5_file['training_data_hgg_pat_name'][:]
new_group_preprocessed['training_data_lgg_pat_name'][:] = hdf5_file['training_data_lgg_pat_name'][:]
new_group_preprocessed['validation_data_pat_name'][:] = hdf5_file['validation_data_pat_name'][:]

# ------------------------------------------------------------------------------------
# start cropping process and standardization process
# ------------------------------------------------------------------------------------

# get the  file  where mean/var values are stored
# TODO: Use the config file global path, not this one.

saveMeanVarFilename = os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1])
logging.info('starting the Cropping/Normalization process.')

# only run thecropping steps on these datasets
run_on_list = ['training_data_segmasks_hgg', 'training_data_hgg', 'training_data_lgg', 'training_data_segmasks_lgg', 'validation_data']

#only run the mean/var normalization on these datasets
std_list = ['training_data_hgg', 'training_data_lgg']
for run_on in run_on_list:

    # we define the final shape after cropping in the config file to make it easy to access. More information available in
    # checkLargestCropSize.ipynb notebook.
    if run_on == 'training_data_hgg':
        im_np = np.empty(config['train_shape_hgg_crop'])
    elif run_on == 'training_data_lgg':
        im_np = np.empty(config['train_shape_lgg_crop'])
    elif run_on == 'validation_data':
        im_np = np.empty(config['val_shape_crop'])

    logger.info('Running on {}'.format(run_on))
    for i in range(0, hdf5_file[run_on].shape[0]):
        # cropping operation
        logger.debug('{}:- Patient {}'.format(run_on, i+1))
        im = hdf5_file[run_on][i]
        m = config['cropping_coords']
        if 'segmasks' in run_on:
            # there are no channels for segmasks
            k = im[m[0]:m[1], m[2]:m[3], m[4]:m[5]]
        else:
            k = im[:, m[0]:m[1], m[2]:m[3], m[4]:m[5]]

        if run_on in std_list:
            # save the image to this numpy array
            im_np[i] = k
        new_group_preprocessed[run_on][i] = k
    # find mean and standard deviation, and apply to data. Also write the mean/std values to disk
    if run_on in std_list:
        logger.info('The dataset {} needs standardization'.format(run_on))
        _tmp, vals = standardize(im_np, findMeanVarOnly=True, saveDump=saveMeanVarFilename + run_on + '_mean_std.p')
        logging.info('Calculated normalization values for {}:\n{}'.format(run_on, vals))
        del im_np

# ====================================================================================

hdf5_file_main.close()
new_hdf5.close()


