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
# from pandas import read_csv
import os
import logging
from configfile import config
import SimpleITK as sitk

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

def resize_mha_volume(image, spacing=[1,1,1], size=[240,240,155]):

    # Create the reference image
    reference_origin = image.GetOrigin()
    reference_direction = np.identity(image.GetDimension()).flatten()
    reference_image = sitk.Image(size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(spacing)
    reference_image.SetDirection(reference_direction)

    # Transform which maps from the reference_image to the current image (output-to-input)
    transform = sitk.AffineTransform(image.GetDimension())
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

    # Modify the transformation to align the centers of the original and reference image
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    centering_transform = sitk.TranslationTransform(image.GetDimension())
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Using the linear interpolator
    image_rs = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0)
    return image_rs


def loadDataGenerator(data_dir, batch_size=1, preprocess=False, loadSurvival=False,
             csvFilePath=None, loadSeg=True, dataset=2018):
    """
    Main function to load BRATS 2017 dataset.

    :param data_dir: path to the folder where patient data resides, needs individual paths for HGG and LGG
    :param batch_size: size of batch to load  (default=1)
    :param loadSurvival: load survival data (True/False) (default=False)
    :param csvFilePath: If loadSurvival is True, provide path to survival data (default=False)
    :param loadSeg: load segmentations (True/False) (default=True)
    :return:
    """

    patID = 0 # used to keep count of how many patients loaded already.
    num_sequences = 4 # number of sequences in the data. BRATS has 4.
    num_slices = config['num_slices']
    running_pats = []
    out_shape = config['spatial_size_for_training'] # shape of the training data

    # create placeholders, currently only supports theano type convention (num_eg, channels, x, y, z)
    images = np.empty((batch_size, num_sequences, out_shape[0], out_shape[1], num_slices)).astype(np.int16)
    labels = np.empty((batch_size, 1)).astype(np.int16)

    if loadSeg == True:
        # create placeholder for the segmentation mask
        seg_masks = np.empty((batch_size, out_shape[0], out_shape[1], num_slices)).astype(np.int16)

    csv_flag = 0

    batch_id = 1 # counter for batches loaded
    logger.info('starting to load images..')
    for patient in glob.glob(data_dir + '/*'):
        if os.path.isdir(patient):
            logger.debug('{} is a directory.'.format(patient))

            # this hacky piece of code is to reorder the filenames, so that segmentation file is always at the end.
            # get all the filepaths
            sequence_folders = glob.glob(patient + '/*')

            vsd_id = []
            for curr_seq in sequence_folders: # get the filepath of the image (nii.gz)
                imagefile = [x for x in glob.glob(os.path.join(curr_seq, '*')) if '.txt' not in x][0]
                # save the name of the patient
                if '.OT.' in imagefile:
                    if loadSeg == True:
                        logger.debug('loading segmentation for this patient..')

                        # open using SimpleITK
                        # SimpleITK would allow me to add number of preprocessing steps that are well defined and
                        # implemented in SITK for their own object type. We can leverage those functions if we preserve
                        # the image object.

                        img_obj = sitk.ReadImage(imagefile)
                        img_obj = resize_mha_volume(img_obj, spacing=[1, 1, 1],
                                                        size=[out_shape[0], out_shape[1], num_slices])

                        pix_data = sitk.GetArrayViewFromImage(img_obj)

                        # check Practice - SimpleiTK.ipynb notebook for more info on why this swapaxes operation is req
                        pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                        pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)

                        seg_masks[patID, :, :, :] = pix_data_swapped
                    else:
                        continue
                else:
                    # this is to ensure that each channel stays at the same place
                    if 'isles' in dataset.lower():
                        if 'T1.' in imagefile:
                            i = 0
                            seq_name = 't1'
                        elif 'T2.' in imagefile:
                            i = 1
                            seq_name = 't2'
                        elif 'DWI.' in imagefile:
                            i = 2
                            seq_name = 'dwi'
                        elif 'Flair.' in imagefile:
                            i = 3
                            seq_name = 'flair'
                            vsd_id.append(os.path.basename(imagefile))
                    else:
                        if 'T1.' in imagefile:
                            i = 0
                            seq_name = 't1'
                        elif 'T2.' in imagefile:
                            i = 1
                            seq_name = 't2'
                        elif 'T1c.' in imagefile:
                            i = 2
                            seq_name = 't1c'
                        elif 'Flair.' in imagefile:
                            i = 3
                            seq_name = 'flair'
                            vsd_id.append(os.path.basename(imagefile))

                    img_obj = sitk.ReadImage(imagefile)
                    if preprocess == True:
                        logger.debug('performing N4ITK Bias Field Correction on {} modality'.format(seq_name))
                        img_obj = preprocessData(img_obj, process=preprocess)

                    img_obj = resize_mha_volume(img_obj, spacing=[1, 1, 1], size=[out_shape[0], out_shape[1], num_slices])

                    pix_data = sitk.GetArrayViewFromImage(img_obj)

                    pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                    pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)

                    images[patID, i, :, :, :] = pix_data_swapped

            patID += 1

            if batch_id % batch_size == 0:
                patID = 0
                if loadSeg == True:
                    yield images, seg_masks, vsd_id
                elif loadSeg == False:
                    yield images, vsd_id

            vsd_id = []

            batch_id += 1



def apply_mean_std(im, mean_var):
    """
    Supercedes the standardize function. Takes the mean/var  file generated during preprocessed data generation and
    applies the normalization step to the patch.
    :param im: patch of size  (num_egs, channels, x, y, z) or (channels, x, y, z)
    :param mean_var: dictionary containing mean/var value calculated in preprocess.py
    :return: normalized patch
    """

    # expects a dictionary of means and VARIANCES, NOT STD
    for m in range(0, 4):
        if len(np.shape(im)) > 4:
            im[:, m, ...] = (im[:, m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])
        else:
            im[m, ...] = (im[m, ...] - mean_var['mn'][m]) / np.sqrt(mean_var['var'][m])

    return im


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
