# MM-GAN: Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network 

This repository contains source code for MM-GAN: [Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network](https://ieeexplore.ieee.org/document/8859286). MM-GAN is a novel GAN-based approach that allows synthesizing missing pulse sequences (modalities) for an MRI scan. For more details, please refer to the paper. 

To cite the paper (IEEE TMI):
```
@ARTICLE{Sharma20,
  author={A. {Sharma} and G. {Hamarneh}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Missing MRI Pulse Sequence Synthesis Using Multi-Modal Generative Adversarial Network}, 
  year={2020},
  volume={39},
  number={4},
  pages={1170-1183},}
```

To cite this repository:
```
@misc{Sharma20Code,
  author = {A. {Sharma}},
  title = {MM-GAN: Missing MRI Pulse Sequence Synthesis using Multi-Modal Generative Adversarial Network},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/trane293/mm-gan}}
}
```

## How to Run
In order to run the code, we recommend that you use [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution. We provivde an environment.yml file that can be used to recreate the exact same Python environment that can be used to run this code.

Once Anaconda is installed, simply do:
```sh
conda env create -f environment.yml
```

Download BRATS2018 dataset from the official website https://www.med.upenn.edu/sbia/brats2018/data.html and extract both the Training and Validation zip files to a folder, say:

```
/local-scratch/data/BRATS2018/Training/
    /local-scratch/data/BRATS2018/Training/HGG
    /local-scratch/data/BRATS2018/Training/LGG
```
Validation patients do not have HGG/LGG labels. 
```
/local-scratch/data/BRATS2018/Validation/
```

Open `modules/configfile.py` and change the following paths to match your folder structure:

```
config['data_dir_prefix'] = os.path.join(mount_path_prefix, 'BRATS2018_Full/') # this should be top level path
config['hdf5_filepath_prefix'] = os.path.join(mount_path_prefix, 'BRATS2018/HDF5_Datasets/BRATS2018_Unprocessed.h5') # top level path
config['hdf5_filepath_prefix_2017'] = os.path.join(mount_path_prefix, 'scratch/asa224/Datasets/BRATS2017/HDF5_Datasets/BRATS.h5') # top level path
config['hdf5_combined'] = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS_Combined_Unprocessed.h5')
config['hdf5_filepath_cropped'] = os.path.join(mount_path_prefix, 'BRATS2018/HDF5_Datasets/BRATS2018_Cropped_Normalized_Unprocessed.h5') # top level path
```

Once done, switch to the conda virtualenv, and run:

`python modules/create_hdf5_file.py`

Once it's finished, run:

`python preprocess.py`

You should have two HDF5 datasets now, one of which is raw BRATS2018 dataset, and one that is cropped to coordinates. 

In order to train MM-GAN on HGG data do:
```sh
./mmgan_hgg.sh
```

Similarly for training on LGG data:
```sh
./mmgan_lgg.sh
```
And it should start the network training process. 

## Directory Structure
| Folder Name | Purpose |
| ------ | ------ |
| modules | contains modules and helper functions, along with network architectures|
| modules/advanced_gans | contains the main Pix2Pix implementation|
| notebooks | contains rough notebooks for quick prototyping|
|prep_BRATS2015| contains code to prepare BRATS2015 data for training/evaluation|
|prep_ISLES2015| contains code to prepare ISLES2015 data for training/validation|