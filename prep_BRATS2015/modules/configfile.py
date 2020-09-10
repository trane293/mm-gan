"""
==========================================================
               Config File to set Parameters
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
DESCRIPTION: This file is solely created for the purpose of
             managing parameters in a global setting. All the
             database loading and generation parameters reside
             here, and are inherited by create_hdf5_file.py
             to generate the HDF5 data store.

             The parameters are also used in the test_database.py
             script to test the created database.
LICENCE: Proprietary for now.
"""
import os
import platform

# WE CAN USE THIS TO CHANGE IMAGE_DATA_FORMAT on the fly
# keras.backend.common._IMAGE_DATA_FORMAT='channels_first'

# to make the code portable even on cedar,you need to add conditions here
node_name = platform.node()
if node_name == 'XPS15':
    # this is my laptop, so the cedar-rm directory is at a different place
    mount_path_prefix = '/home/anmol/mounts/cedar-rm/'
elif 'computecanada' in node_name: # we're in compute canada, maybe in an interactive node, or a scheduler node.
    # mount_path_prefix = '/home/asa224/' # home directory
    mount_path_prefix = ''  # home directory
else:
    # this is probably my workstation or TS server
    # mount_path_prefix = '/local-scratch/asa224_new/mounts/cedar-rm/'
    mount_path_prefix = ''

config = {}
# set the data directory and output hdf5 file path.
# data_dir is the top level path containing both training and validation sets of the brats dataset.
# config['data_dir_prefix'] = os.path.join(mount_path_prefix, 'rrg_proj_dir/scratch_files_globus/Datasets/BRATS2015/') # this should
config['data_dir_prefix'] = os.path.join(mount_path_prefix, '/local-scratch/anmol/data/BRATS2015') # this should#  be top level path
config['hdf5_filepath_prefix'] = os.path.join(mount_path_prefix, '/local-scratch/anmol/data/BRATS2015/HDF5_Datasets/BRATS2015.h5') # top level path

config['spatial_size_for_training'] = (240, 240) # If any preprocessing is done, then this needs to change. This is the shape of data that you want to train with. If you are changing this that means you did some preprocessing.
config['num_slices'] = 155 # number of slices in input data. THIS SHOULD CHANGE AS WELL WHEN PERFORMING PREPROCESSING
config['volume_size'] = list(config['spatial_size_for_training']) + [config['num_slices']]
config['seed'] = 1338
config['data_order'] = 'th' # what order should the indices be to store in hdf5 file
config['train_hgg_patients'] = 220 # number of HGG patients in training
config['train_lgg_patients'] = 54 # number of LGG patients in training
config['testing_hgglgg_patients'] = 110 # number of HGG patients in training

config['cropping_coords'] = [29, 223, 41, 196, 0, 148] # coordinates used to crop the volumes, this is generated using the notebook checkLargestCropSize.ipynb
config['size_after_cropping'] = [194, 155, 148] # set this if you set the above variable. Calculate this using the notebook again.

config['batch_size'] = 1 # how many images to load at once in the generator

# check the order of data and chose proper data shape to save images
if config['data_order'] == 'th':


    config['train_shape_hgg'] = (
    config['train_hgg_patients'], 4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][1],
    config['num_slices'])
    config['train_shape_lgg'] = (
    config['train_lgg_patients'], 4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][1],
    config['num_slices'])
    config['train_segmasks_shape_hgg'] = (
    config['train_hgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1],
    config['num_slices'])
    config['train_segmasks_shape_lgg'] = (
    config['train_lgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1],
    config['num_slices'])
    config['testing_hgglgg_patients_shape'] = (config['testing_hgglgg_patients'], 4,
                                               config['spatial_size_for_training'][0],
                                               config['spatial_size_for_training'][1],
                                               config['num_slices'])


    config['train_shape_hgg_crop'] = (
    config['train_hgg_patients'], 4, config['size_after_cropping'][0], config['size_after_cropping'][1],
    config['size_after_cropping'][2])
    config['train_shape_lgg_crop'] = (
    config['train_lgg_patients'], 4, config['size_after_cropping'][0], config['size_after_cropping'][1],
    config['size_after_cropping'][2])
    config['train_segmasks_shape_hgg_crop'] = (
    config['train_hgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1],
    config['size_after_cropping'][2])
    config['train_segmasks_shape_lgg_crop'] = (
    config['train_lgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1],
    config['size_after_cropping'][2])
    config['testing_hgglgg_patients_shape_crop'] = (config['testing_hgglgg_patients'], 4,
                                                    config['size_after_cropping'][0],
                                                    config['size_after_cropping'][1],
                                                    config['size_after_cropping'][2])

