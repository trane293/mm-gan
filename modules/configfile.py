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
    mount_path_prefix = '/home/asa224/' # home directory
else:
    # this is probably my workstation or TS server
    mount_path_prefix = '/local-scratch/anmol/data'

config = {}
# set the data directory and output hdf5 file path.
# data_dir is the top level path containing both training and validation sets of the brats dataset.
config['data_dir_prefix'] = os.path.join(mount_path_prefix, 'BRATS2018_Full/') # this should be top level path
config['hdf5_filepath_prefix'] = os.path.join(mount_path_prefix, 'BRATS2018/HDF5_Datasets/BRATS2018_Unprocessed.h5') # top level path
config['hdf5_filepath_prefix_2017'] = os.path.join(mount_path_prefix, 'scratch/asa224/Datasets/BRATS2017/HDF5_Datasets/BRATS.h5') # top level path
config['hdf5_combined'] = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS_Combined_Unprocessed.h5')
config['hdf5_filepath_cropped'] = os.path.join(mount_path_prefix, 'BRATS2018/HDF5_Datasets/BRATS2018_Cropped_Normalized_Unprocessed.h5') # top level path
config['saveMeanVarFilepathHGG'] = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS2018_HDF5_Datasetstraining_data_hgg_mean_var.p')
config['saveMeanVarFilepathLGG'] = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS2018_HDF5_Datasetstraining_data_lgg_mean_var.p')
config['saveMeanVarCombinedData'] = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'combined_data_mean_var.p')

config['model_snapshot_location'] = os.path.join(mount_path_prefix, 'scratch/asa224/model-snapshots/')
config['model_checkpoint_location'] = os.path.join(mount_path_prefix, 'scratch/asa224/model-checkpoints/')
config['model_prediction_location'] = os.path.join(mount_path_prefix, 'scratch/asa224/model-predictions/')
# # IF YOU PERFORM PREPROCESSING, THESE VARIABLES ARE TO BE CHANGED. DEFAULT VALUES ARE:
# config['spatial_size_for_training'] = (240, 240) # If any preprocessing is done, then this needs to change. This is the shape of data that you want to train with. If you are changing this that means you did some preprocessing.
# config['num_slices'] = 155 # number of slices in input data. THIS SHOULD CHANGE AS WELL WHEN PERFORMING PREPROCESSING

# IF YOU PERFORM PREPROCESSING, THESE VARIABLES ARE TO BE CHANGED.
config['spatial_size_for_training'] = (240, 240) # If any preprocessing is done, then this needs to change. This is the shape of data that you want to train with. If you are changing this that means you did some preprocessing.
config['num_slices'] = 155 # number of slices in input data. THIS SHOULD CHANGE AS WELL WHEN PERFORMING PREPROCESSING
config['volume_size'] = list(config['spatial_size_for_training']) + [config['num_slices']]
config['seed'] = 1338
config['data_order'] = 'th' # what order should the indices be to store in hdf5 file
config['train_hgg_patients'] = 210 # number of HGG patients in training
config['train_lgg_patients'] = 75 # number of LGG patients in training
config['validation_patients'] = 66 # number of patients in validation

config['batch_size'] = 1 # how many images to load at once in the generator

config['cropping_coords'] = [29, 223, 41, 196, 0, 148] # coordinates used to crop the volumes, this is generated using the notebook checkLargestCropSize.ipynb
config['size_after_cropping'] = [194, 155, 148] # set this if you set the above variable. Calculate this using the notebook again.

config['data_split'] = {'train': 98,
                        'test': 2}

config['std_scale_range'] = [6] # [4,6,8,10] scale the standard deviation for path generation process to allow patches from far off regions
config['num_patches_per_patient'] = 50 # number of patches to generate for a single patient
config['patch_size'] = [64, 64, 64] # size of patch to extract
config['patch_input_shape'] = [4] + config['patch_size']
config['gen_patches_from'] = 'original' # generate patches from the cropped version of the database or original.
config['validate_on'] = 'original' # Perform validation on original images or cropped images
config['num_labels'] = 3 # number of labels in the segmentation mask, except background
config['max_label_val'] = 4

config['val_shape_after_prediction'] = []

# check the order of data and chose proper data shape to save images
if config['data_order'] == 'th':
    config['train_shape_hgg'] = (config['train_hgg_patients'], 4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['train_shape_lgg'] = (config['train_lgg_patients'], 4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['train_segmasks_shape_hgg'] = (config['train_hgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['train_segmasks_shape_lgg'] = (config['train_lgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['val_shape'] = (config['validation_patients'], 4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])

    config['train_shape_hgg_crop'] = (config['train_hgg_patients'], 4, config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['train_shape_lgg_crop'] = (config['train_lgg_patients'], 4, config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['train_segmasks_shape_hgg_crop'] = (config['train_hgg_patients'], config['size_after_cropping'][0],config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['train_segmasks_shape_lgg_crop'] = (config['train_lgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['val_shape_crop'] = (config['validation_patients'], 4, config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['numpy_patch_size'] = (config['num_patches_per_patient'], 4, config['patch_size'][0], config['patch_size'][1],
                           config['patch_size'][2])
elif config['data_order'] == 'tf':
    config['train_shape_hgg'] = (config['train_hgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'], 4)
    config['train_shape_lgg'] = (config['train_lgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'], 4)
    config['train_segmasks_shape_hgg'] = (config['train_hgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['train_segmasks_shape_lgg'] = (config['train_lgg_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'])
    config['val_shape'] = (config['validation_patients'], config['spatial_size_for_training'][0], config['spatial_size_for_training'][1], config['num_slices'], 4)

    config['train_shape_hgg_crop'] = (config['train_hgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2], 4)
    config['train_shape_lgg_crop'] = (config['train_lgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2], 4)
    config['train_segmasks_shape_hgg_crop'] = (config['train_hgg_patients'], config['size_after_cropping'][0],config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['train_segmasks_shape_lgg_crop'] = (config['train_lgg_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2])
    config['val_shape_crop'] = (config['validation_patients'], config['size_after_cropping'][0], config['size_after_cropping'][1], config['size_after_cropping'][2], 4)
    config['numpy_patch_size'] = (config['num_patches_per_patient'], config['patch_size'][0], config['patch_size'][1], config['patch_size'][2], 4)

tmp = list(config['val_shape'])
tmp[1] = config['num_labels']
config['val_shape_after_prediction'] = tuple(tmp)