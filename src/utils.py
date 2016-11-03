import cv
import cv2
import h5py
import os
import PIL
import sys


def read_data(filename='../data/fonts.hdf5', verbose=True):
    """
    Read the fonts file.

    :param filename: path to the fonts data set
    :param verbose: sets verbosity
    :return numpy ndarray containing the fonts data set
    """
    if not os.path.exists(filename):
        print 'File', filename, 'does not exists'
        sys.exit(1)

    if verbose:
        print 'Reading file', filename

    fd = h5py.File(filename, 'r')
    return fd['fonts']


def grayscale_to_rgb(grayscale_array):
    """
    Convert two-dimensional grayscale image to rgb by cloning single
    channel input to the remaining two channels.

    :param grayscale_array: grayscale image matrix
    :return: 3-channel RGB image corresponding to input grayscale image
    """
    if len(grayscale_array.shape) != 2:
        print 'Input grayscale image should be two-dimensional'
        return

    rgb_image = cv2.cvtColor(grayscale_array, cv.CV_GRAY2RGB)
    return PIL.Image.fromarray(255 - rgb_image)


if __name__ == '__main__':
    print 'Hello world. Welcome to utilities.'
