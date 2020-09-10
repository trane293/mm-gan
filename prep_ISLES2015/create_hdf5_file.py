"""
==========================================================
                Prepare BRATS 2015 Data
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
from modules.configfile import config

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

try:
    logger.info('Dataloader file in use: {}'.format(dataloader.__file__))
except:
    logger.info('Cannot determine the dataloader class being used, check  carefully before proceeding')


# whether or not to preprocess the data before creating the HDF5 file? Check the preprocess function in dataloader to
# know exactly what preprocessing is being performed.
# NO BIAS FIELD CORRECTION FOR ISLES DATASET FOR COMPARISON
PREPROCESS_DATA = False

logger.info('[IMPORTANT] This will create a new HDF5 file in SAFE MODE. It will  NOT OVERWRITE A PREVIOUS HDF5 FILE '
            'IF ITS PRESENT')
def createHDF5File(config):
    """
    Function to create a new HDF5 File to hold the BRATS 2015 data. The function will fail if there's already a file
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

    grp.create_dataset("training_data", config['train_shape'], np.float32)
    grp.create_dataset("training_data_pat_name", (config['train_shape'][0],), dtype="S100")
    grp.create_dataset("training_data_segmasks", config['train_segmasks_shape'], np.int16)

    return hdf5_file

def main():
    hdf5_file_main = createHDF5File(config)
    # hdf5_file_main = h5py.File(config['hdf5_filepath_prefix'], mode='a')
    # Go inside the "original_data" parent directory.
    # we need to create the validation data dataset again since the shape has changed.
    hdf5_file = hdf5_file_main['original_data']
    contents = glob.glob(os.path.join(config['data_dir_prefix'], '*'))

    # for debugging, making sure Training set is loaded first not Testing, since that is tested already.
    contents.reverse()
    for dataset_splits in contents: # Challenge/LeaderBoard data?
        if os.path.isdir(dataset_splits): # make sure its a directory
            for grade_type in glob.glob(os.path.join(dataset_splits, '*')):
                # there may be other files in there (like the survival data), ignore them.
                if os.path.isdir(grade_type):
                    count = 0
                    if 'Testing' in dataset_splits:
                        logger.info('currently loading Testing -> {} data.'.format(os.path.basename(grade_type)))
                        ty = 'Testing'

                        for images, pats in dataloader.loadDataGenerator(grade_type,
                                                            batch_size=config['batch_size'], loadSurvival=False,
                                                            csvFilePath=None, loadSeg=False,
                                                            preprocess=PREPROCESS_DATA, dataset='ISLES'):
                            logger.info('loading patient {} from {}'.format(count, grade_type))
                            if 'HGG_LGG' in grade_type:
                                if ty == 'Testing':
                                    main_data_name = 'testing_hgglgg_patients'
                                    main_data_pat_name = 'testing_hgglgg_patients_pat_name'

                                hdf5_file[main_data_name][count:count+config['batch_size'],...] = images
                                t = 0
                                for i in range(count, count + config['batch_size']):
                                    hdf5_file[main_data_pat_name][i] = pats[t].split('.')[-2]
                                    t += 1

                            logger.info('loaded {} patient(s) from {}'.format(count + config['batch_size'], grade_type))
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
                                                                        batch_size=config['batch_size'],                                                                                    loadSurvival=False,
                                                                        csvFilePath=None, loadSeg=True,
                                                                        preprocess=PREPROCESS_DATA,
                                                                        dataset='ISLES'):
                                        logger.info('loading patient {} from {}'.format(count, grade_type))

                                        hdf5_file['training_data'][count:count+config['batch_size'],...] = images
                                        hdf5_file['training_data_segmasks'][count:count+config['batch_size'], ...] = segmasks
                                        t = 0
                                        for i in range(count, count + config['batch_size']):
                                            hdf5_file['training_data_pat_name'][i] = pats[t].split('/')[-1]
                                            t += 1

                                        logger.info('loaded {} patient(s) from {}'.format(count + config['batch_size'], grade_type))
                                        count += config['batch_size']
        # close the HDF5 file
    # close the HDF5 file
    hdf5_file_main.close()

if __name__ == '__main__':
    main()