"""
==========================================================
                Misc Helper Classes/Functions
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
DESCRIPTION: The module has various helper classes/functions
             that can be used throughout the pipeline, and
             don't fit exactly in either data loading or
             visualization operations.
LICENCE: Proprietary for now.
"""

import numpy as np
from configfile import config
import h5py
from nilearn._utils import check_niimg
from nilearn.image import new_img_like
from nilearn.image import reorder_img, resample_img



class Rect3D:
    """
    Class to encapsulate the Rectangle coordinates. This prevents future
    issues when the coordinates need to be standardized.
    """
    def __init__(self, coord_list):
        if len(coord_list) < 6:
            print('Coordinate list shape is incorrect, creating empty object!')
            coord_list = [0, 0, 0, 0, 0, 0]
            self.empty = True
        else:
            self.empty = False

        self.rmin = coord_list[0]
        self.rmax = coord_list[1]
        self.cmin = coord_list[2]
        self.cmax = coord_list[3]
        self.zmin = coord_list[4]
        self.zmax = coord_list[5]
        self.list_view = coord_list

    def show(self):
        return self.list_view

class Rect2D:
    """
    Class to encapsulate the Rectangle coordinates. This prevents future
    issues when the coordinates need to be standardized.
    """
    def __init__(self, coord_list):
        if len(coord_list) < 4:
            print('Coordinate list shape is incorrect, creating empty object!')
            coord_list = [0, 0, 0, 0]
            self.empty = True
        else:
            self.empty = False

        self.rmin = coord_list[0]
        self.rmax = coord_list[1]
        self.cmin = coord_list[2]
        self.cmax = coord_list[3]
        self.list_view = coord_list

    def show(self):
        return self.list_view

def bbox_3D(img, tol=0.5):
    """
    TOL = argument used when dark regions are >0
          (usually after some preprocessing, like
          rescaling).
    """
    r, c, z = np.where(img > tol)
    rmin, rmax, cmin, cmax, zmin, zmax = np.min(r), np.max(r), np.min(c), np.max(c), np.min(z), np.max(z)
    rect_obj = Rect3D([rmin, rmax, cmin, cmax, zmin, zmax])
    return rect_obj

def bbox_2D(img, tol=0.5):
    """
    TOL = argument used when dark regions are >0
          (usually after some preprocessing, like
          rescaling).
    """
    r, c = np.where(img > tol)
    if r.size == 0 or c.size == 0:
        return Rect2D([-1, -1, -1, -1])
    else:
        rmin, rmax, cmin, cmax = np.min(r), np.max(r), np.min(c), np.max(c)
        rect_obj = Rect2D([rmin, rmax, cmin, cmax])
        return rect_obj

def open_hdf5(filepath=None, mode='r'):
    if filepath == None:
        filepath = config['hdf5_filepath_prefix']

    return h5py.File(filepath, mode=mode)

def get_data_splits_bbox(hdf5_filepath, train_start=0, train_end=190, test_start=190, test_end=None):
    """

    :param hdf5_filepath:
    :param train_start: Start index to slice to get the training data. For 10 instances starting from 0, choose 0.
    :param train_end: End index for training. Remember this index is 'exclusive', so if you want 10 instances, choose this as 10
    :param test_start: Start index to slice to get the testing data. Same comment as above.
    :param test_end: End index for testing.
    :return: Keras instances to slice into x_train, y_train, x_test, y_test.
    """
    import keras
    filepath = config['hdf5_filepath_prefix'] if hdf5_filepath is None else hdf5_filepath

    x_train = keras.utils.io_utils.HDF5Matrix(filepath, "training_data_hgg", start=train_start, end=train_end,
                                              normalizer=None)
    y_train = keras.utils.io_utils.HDF5Matrix(filepath, "bounding_box_hgg", start=train_start, end=train_end,
                                              normalizer=None)

    x_test = keras.utils.io_utils.HDF5Matrix(filepath, "training_data_hgg", start=test_start, end=test_end,
                                             normalizer=None)
    y_test = keras.utils.io_utils.HDF5Matrix(filepath, "bounding_box_hgg", start=test_start, end=test_end,
                                             normalizer=None)

    return x_train, y_train, x_test, y_test

def createDense(bbox, im):
    box = np.zeros(im.shape)
    box[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
    return box


def _crop_img_to(img, slices, copy=True):
    """Crops image to a smaller size
    Crop img to size indicated by slices and adjust affine
    accordingly
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Img to be cropped. If slices has less entries than img
        has dimensions, the slices will be applied to the first len(slices)
        dimensions
    slices: list of slices
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
        defines a 3D cube
    copy: boolean
        Specifies whether cropped data is to be copied or not.
        Default: True
    Returns
    -------
    cropped_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Cropped version of the input image
    """

    img = check_niimg(img)

    data = img.get_data()
    affine = img.affine

    cropped_data = data[slices]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    return new_img_like(img, cropped_data, new_affine)


def crop_img_custom(img, slices=None, rtol=1e-8, copy=True):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_data()

    if slices is not None:
        return _crop_img_to(img, slices, copy=copy), slices
    else:
        infinity_norm = max(-data.min(), data.max())
        passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                         data > rtol * infinity_norm)

        if data.ndim == 4:
            passes_threshold = np.any(passes_threshold, axis=-1)
        coords = np.array(np.where(passes_threshold))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

        # pad with one voxel to avoid resampling problems
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, data.shape[:3])

        slices = [slice(s, e) for s, e in zip(start, end)]

        return _crop_img_to(img, slices, copy=copy), slices


def resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation, clip=True)