"""
==========================================================
                Prepare BRATS 2017 Data
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
DESCRIPTION: This file is used to generate an HDF5 dataset,
             which is easy to load and manipulate compared
             to working directly with raw data all the time.
             Loading and working with HDF5 files is much
             faster and efficient due to its asynchronous loading
             system.

             The HDF5 file generated can be hosted on a remote server
             (like CEDAR) and then accessed over SSHFS. Practically,
             this is very effective and does not hinder the performance
             by a large margin.

             This script generates a simple HDF5 data store,
             which contains the original numpy arrays of the
             data store. To perform any preprocessing, implement
             the preprocessData()function in dataloader.py to
             work directly on nibabel objects, instead of
             numpy objects.
LICENCE: Proprietary for now.
"""

import os
import glob
from modules import dataloader
import logging
import numpy as np
import h5py
import sys
sys.path.append('../')
from modules.configfile import config

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# whether or not to preprocess the data before creating the HDF5 file? Check the preprocess function in dataloader to
# know exactly what preprocessing is being performed.
PREPROCESS_DATA = False

logger.info('[IMPORTANT] This will create a new HDF5 file in SAFE MODE. It will  NOT OVERWRITE A PREVIOUS HDF5 FILE '
            'IF ITS PRESENT')
def createHDF5File(config):
    """
    Function to create a new HDF5 File to hold the BRATS 2017 data. The function will fail if there's already a file
    present with the same name (SAFE OPERATION)

    :param config: The config variable defined in configfile.py
    :return: hdf5_file object
    """

    # w- mode fails when there is a file already.
    hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='w')

    # create a new parent directory to hold the data inside it
    grp = hdf5_file.create_group("original_data")

    # the dataset is int16 originally, checked using nibabel, however we create float32 containers to make the dataset
    # compatible with further preprocessing.
    # HGG Data
    grp.create_dataset("training_data_hgg", config['train_shape_hgg'], np.float32)
    grp.create_dataset("training_data_hgg_pat_name", (config['train_shape_hgg'][0],), dtype='S100')
    grp.create_dataset("training_data_segmasks_hgg", config['train_segmasks_shape_hgg'], np.int16)

    # LGG Data
    grp.create_dataset("training_data_lgg", config['train_shape_lgg'], np.float32)
    grp.create_dataset("training_data_lgg_pat_name", (config['train_shape_lgg'][0],), dtype='S100')
    grp.create_dataset("training_data_segmasks_lgg", config['train_segmasks_shape_lgg'], np.int16)

    # Validation Data, with no segmentation masks
    grp.create_dataset("validation_data", config['val_shape'], np.float32)
    grp.create_dataset("validation_data_pat_name", (config['val_shape'][0],), dtype='S100')
    return hdf5_file

def main():
    hdf5_file_main = createHDF5File(config)
    # hdf5_file_main = h5py.File(config['hdf5_filepath_prefix'], mode='w')
    # Go inside the "original_data" parent directory.
    # we need to create the validation data dataset again since the shape has changed.
    hdf5_file = hdf5_file_main['original_data']
    del hdf5_file['validation_data']
    del hdf5_file['validation_data_pat_name']
    # Validation Data, with no segmentation masks
    hdf5_file.create_dataset("validation_data", config['val_shape'], np.float32)
    hdf5_file.create_dataset("validation_data_pat_name", (config['val_shape'][0],), dtype="S100")

    for dataset_splits in glob.glob(os.path.join(config['data_dir_prefix'], '*')): # Training/Validation data?
        if os.path.isdir(dataset_splits) and 'Validation' in dataset_splits: # make sure its a directory
            # VALIDATION data handler
            logger.info('currently loading Validation data.')
            count = 0
            # validation data does not have HGG and LGG distinctions
            for images, pats in dataloader.loadDataGenerator(dataset_splits,
                                         batch_size=config['batch_size'], loadSurvival=False, csvFilePath=None,
                                         loadSeg=False, preprocess=PREPROCESS_DATA):
                hdf5_file['validation_data'][count:count+config['batch_size'],...] = images
                t = 0

                for i in range(count, count + config['batch_size']):
                    hdf5_file['validation_data_pat_name'][i] = pats[t].split('/')[-1].encode('utf-8')
                    t += 1

                # logger.debug('array equal?: {}'.format(np.array_equal(hdf5_file['validation_data'][count:count+config['batch_size'],...], images)))
                logger.info('loaded {} patient(s) from {}'.format(count + config['batch_size'], dataset_splits))
                count += config['batch_size']

        else:
        # TRAINING data handler
            if os.path.isdir(dataset_splits) and 'Training' in dataset_splits:
                for grade_type in glob.glob(os.path.join(dataset_splits, '*')):
                    # there may be other files in there (like the survival data), ignore them.
                    if os.path.isdir(grade_type):
                        count = 0
                        logger.info('currently loading Training data.')
                        for images, segmasks, pats in dataloader.loadDataGenerator(grade_type,
                                                            batch_size=config['batch_size'], loadSurvival=False,
                                                            csvFilePath=None, loadSeg=True,
                                                            preprocess=PREPROCESS_DATA):
                            logger.info('loading patient {} from {}'.format(count, grade_type))
                            if 'HGG' in grade_type:
                                hdf5_file['training_data_hgg'][count:count+config['batch_size'],...] = images
                                hdf5_file['training_data_segmasks_hgg'][count:count+config['batch_size'], ...] = segmasks
                                t = 0
                                for i in range(count, count + config['batch_size']):
                                    hdf5_file['training_data_hgg_pat_name'][i] = pats[t].split('/')[-1].encode('utf-8')
                                    t += 1
                            elif 'LGG' in grade_type:
                                hdf5_file['training_data_lgg'][count:count+config['batch_size'], ...] = images
                                hdf5_file['training_data_segmasks_lgg'][count:count+config['batch_size'], ...] = segmasks
                                t = 0
                                for i in range(count, count + config['batch_size']):
                                    hdf5_file['training_data_lgg_pat_name'][i] = pats[t].split('/')[-1].encode('utf-8')
                                    t += 1

                            logger.info('loaded {} patient(s) from {}'.format(count + config['batch_size'], grade_type))
                            count += config['batch_size']
    # close the HDF5 file
    hdf5_file_main.close()

if __name__ == '__main__':
    main()