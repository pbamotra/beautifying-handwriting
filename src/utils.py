import cv
import cv2
import h5py
import os
from PIL import Image
import sys
import string


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
    return Image.fromarray(255 - rgb_image)


def save_font_image_by_char(font_data, char='all', n_subset=0, out='../data/images/'):
    charset = list(string.ascii_uppercase) + list(string.ascii_lowercase) + map(str, range(10))
    mapping = {ch: ch_id for (ch_id, ch) in enumerate(charset)}

    if char == 'all':
        for font_id, font in enumerate(font_data):
            if (font_id + 1) % 10 == 0:
                print 'Processed', font_id + 1, 'fonts'
            for ch_id, ch in enumerate(font):
                img = grayscale_to_rgb(ch)
                img.save(out + str(font_id) + '_' + str(ch_id) + '.png')
            if n_subset != 0 and (font_id + 1) == n_subset:
                break
    else:
        for font_id, font in enumerate(font_data):
            if (font_id + 1) % 10 == 0:
                print 'Processed', font_id + 1, 'fonts'
            for ch_id, cha in enumerate(char):
                ch = font[mapping[cha]]
                img = grayscale_to_rgb(ch)
                img.save(out + str(font_id) + '_' + cha + '.png')
            if n_subset != 0 and (font_id + 1) == n_subset:
                break


def get_font_image_by_char(font_data, char='all', n_subset=0):

    charset = list(string.ascii_uppercase) + list(string.ascii_lowercase) + map(str, range(10))
    mapping = {ch: ch_id for (ch_id, ch) in enumerate(charset)}

    result = list()

    if char == 'all':
        for font_id, font in enumerate(font_data, start=1):
            partial_list = list()
            for ch_id, ch in enumerate(font):
                partial_list.append(grayscale_to_rgb(ch))
            result.append(partial_list)
            if n_subset and (font_id == n_subset):
                break
    else:
        for font_id, font in enumerate(font_data, start=1):
            partial_list = list()
            for ch_id, cha in enumerate(char):
                ch = font[mapping[cha]]
                partial_list.append(grayscale_to_rgb(ch))
            result.append(partial_list)
            if n_subset and (font_id == n_subset):
                break

    return result


def image_resize(input_image, output_filename='out.png', resize=(64,64)):
    """
    Convert input image from NIST SD 19 to 64x64 RGB image

    :param input_image: image filename
    :param output_filename: output image filename
    :param resize: output image size
    :return: 3-channel RGB image corresponding to input image
    """
    image = Image.open(input_image).convert('RGB')
    image = image.resize(resize)
    image.save(output_filename)

if __name__ == '__main__':
    print 'Hello world. Welcome to utilities.'
