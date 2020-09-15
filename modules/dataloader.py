"""
==========================================================
                Load BRATS 2017 Data
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
DESCRIPTION: The script has multiple functions to load,
             preprocess, and standardize the BRATS
             2017 dataset, along with its survival annotations.
             Main function is the loadDataGenerator which loads
             the data using a generator, and doesn't hog memory.

             The loadDataGenerator is capable of applying
             arbitrary preprocessing steps to the data. This can be
             achieved by implementing the function preprocessData.
LICENCE: Proprietary for now.
"""

from __future__ import print_function
import glob as glob
import numpy as np
import pickle
import sys as sys
from pandas import read_csv
import os
import logging
from configfile import config
import SimpleITK as sitk

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocessData(img_obj, process=False):
    """
    Perform preprocessing on the original nibabel object.
    Use this function to:
        1) Resize/Resample the 3D Volume
        2) Crop the brain region
        3) Do (2) then (1).

    When you do preprocessing, especially something that
    changes the spatial size of the volume, make sure you
    update config['spatial_size_for_training'] = (240, 240)
    value in the config file.

    :param img_obj:
    :param process:
    :return:
    """
    if process == False:
        return img_obj
    else:
        maskImage = sitk.OtsuThreshold(img_obj, 0, 1, 200)
        image = sitk.Cast(img_obj, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFilltingLevels = 4
        corrector.SetMaximumNumberOfIterations([4] * numberFilltingLevels)
        output = corrector.Execute(image, maskImage)
        return output


def loadDataGenerator(data_dir, batch_size=1, preprocess=False, loadSurvival=False,
                      csvFilePath=None, loadSeg=True):
    """
    Main function to load BRATS 2017 dataset.

    :param data_dir: path to the folder where patient data resides, needs individual paths for HGG and LGG
    :param batch_size: size of batch to load  (default=1)
    :param loadSurvival: load survival data (True/False) (default=False)
    :param csvFilePath: If loadSurvival is True, provide path to survival data (default=False)
    :param loadSeg: load segmentations (True/False) (default=True)
    :return:
    """

    patID = 0  # used to keep count of how many patients loaded already.
    num_sequences = 4  # number of sequences in the data. BRATS has 4.
    num_slices = config['num_slices']
    running_pats = []
    out_shape = config['spatial_size_for_training']  # shape of the training data

    # create placeholders, currently only supports theano type convention (num_eg, channels, x, y, z)
    images = np.empty((batch_size, num_sequences, out_shape[0], out_shape[1], num_slices)).astype(np.int16)
    labels = np.empty((batch_size, 1)).astype(np.int16)
    slices = None  # not used anymore

    if loadSeg == True:
        # create placeholder for the segmentation mask
        seg_masks = np.empty((batch_size, out_shape[0], out_shape[1], num_slices)).astype(np.int16)

    csv_flag = 0

    if loadSurvival == True:
        logger.info('Loading annotations as well..')
        if csvFilePath != None:
            logger.info('trying to load CSV file...')
            csv_file = read_csv(csvFilePath)
            csv_flag = 1
            logger.info('opened CSV file successfully')
        else:
            logger.debug('loadSurvival is True but no csvFilePath provided!')
            raise Exception

    batch_id = 1  # counter for batches loaded
    logger.info('starting to load images..')
    for patients in glob.glob(data_dir + '/*'):
        if os.path.isdir(patients):
            logger.debug('{} is a directory.'.format(patients))

            # save the name of the patient
            running_pats.append(patients)
            if csv_flag == 1:
                patName = patients.split('/')[-1]
                try:
                    labels[patID] = csv_file[csv_file['Brats17ID'] == patName]['Survival'].tolist()[0]
                    logger.debug('Added survival data..')
                except IndexError:
                    labels[patID] = None

            # this hacky piece of code is to reorder the filenames, so that segmentation file is always at the end.
            # get all the filepaths
            files = glob.glob(patients + '/*')

            # create list without the "seg" filepath inside it
            files_new = [x for x in files if 'seg' not in x]

            # create another list with only seg folder inside it
            seg_filename = [x for x in files if 'seg' in x]
            if seg_filename != []:
                # concatenate the list, now the seg filepath is at the end
                files_new = files_new + seg_filename

            for imagefile in files_new:  # get the filepath of the image (nii.gz)
                if 'seg' in imagefile:
                    if loadSeg == True:
                        logger.debug('loading segmentation for this patient..')

                        # open using SimpleITK
                        # SimpleITK would allow me to add number of preprocessing steps that are well defined and
                        # implemented in SITK for their own object type. We can leverage those functions if we preserve
                        # the image object.

                        img_obj = sitk.ReadImage(imagefile)
                        pix_data = sitk.GetArrayViewFromImage(img_obj)

                        # check Practice - SimpleiTK.ipynb notebook for more info on why this swapaxes operation is req
                        pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                        pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)

                        seg_masks[patID, :, :, :] = pix_data_swapped
                    else:
                        continue
                else:
                    # this is to ensure that each channel stays at the same place
                    if 't1.' in imagefile:
                        i = 0
                        seq_name = 't1'
                    elif 't2.' in imagefile:
                        i = 1
                        seq_name = 't2'
                    elif 't1ce.' in imagefile:
                        i = 2
                        seq_name = 't1ce'
                    elif 'flair.' in imagefile:
                        i = 3
                        seq_name = 'flair'

                    img_obj = sitk.ReadImage(imagefile)
                    if preprocess == True:
                        logger.debug('performing N4ITK Bias Field Correction on {} modality'.format(seq_name))
                    img_obj = preprocessData(img_obj, process=preprocess)

                    pix_data = sitk.GetArrayViewFromImage(img_obj)

                    pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                    pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)

                    images[patID, i, :, :, :] = pix_data_swapped

            patID += 1

            if batch_id % batch_size == 0:
                patID = 0
                if csv_flag == 1 and loadSeg == True:
                    yield images, labels, seg_masks, running_pats
                elif csv_flag == 0 and loadSeg == True:
                    yield images, seg_masks, running_pats
                elif csv_flag == 0 and loadSeg == False:
                    yield images, running_pats

                running_pats = []

            batch_id += 1


def standardize(images, findMeanVarOnly=True, saveDump=None, applyToTest=None):
    """
    This function standardizes the input data to zero mean and unit variance. It is capable of calculating the
    mean and std values from the input data, or can also apply user specified mean/std values to the images.

    :param images: numpy ndarray of shape (num_qg, channels, x, y, z) to apply mean/std normalization to
    :param findMeanVarOnly: only find the mean and variance of the input data, do not normalize
    :param saveDump: if True, saves the calculated mean/variance values to the disk in pickle form
    :param applyToTest: apply user specified mean/var values to given images. checkLargestCropSize.ipynb has more info
    :return: standardized images, and vals (if mean/val was calculated by the function
    """

    # takes a dictionary
    if applyToTest != None:
        logger.info('Applying to test data using provided values')
        # from training_helpers import apply_mean_std
        from .helpers import apply_mean_std
        images = apply_mean_std(images, applyToTest)
        return images

    logger.info('Calculating mean value..')
    vals = {
        'mn': [],
        'var': []
    }
    for i in range(4):
        vals['mn'].append(np.mean(images[:, i, :, :, :]))

    logger.info('Calculating variance..')
    for i in range(4):
        vals['var'].append(np.var(images[:, i, :, :, :]))

    if findMeanVarOnly == False:
        logger.info('Starting standardization process..')

        for i in range(4):
            images[:, i, :, :, :] = ((images[:, i, :, :, :] - vals['mn'][i]) / float(vals['var'][i]))

        logger.info('Data standardized!')

    if saveDump != None:
        logger.info('Dumping mean and var values to disk..')
        pickle.dump(vals, open(saveDump, 'wb'))
    logger.info('Done!')

    return images, vals


if __name__ == "__main__":
    """
    Only for testing purpose, DO NOT ATTEMPT TO RUN THIS SCRIPT. ONLY IMPORT AS MODULE
    """
    data_dir = '/local-scratch/cedar-rm/scratch/asa224/Datasets/BRATS2017/MICCAI_BraTS17_Data_Training/HGG/'
    images, segmasks = loadDataGenerator(data_dir, batch_size=2, loadSurvival=False,
                                         csvFilePath=None, loadSeg=True)