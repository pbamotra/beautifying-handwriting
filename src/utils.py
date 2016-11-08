import cv
import cv2
import h5py
import os
import PIL
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
    return PIL.Image.fromarray(255 - rgb_image)


def get_font_image_by_char(font_data, char='all', n_subset=0, out='../data/images/'):
    charset = list(string.ascii_uppercase) + list(string.ascii_lowercase) + map(str, range(10))
    mapping = {ch: ch_id for (ch_id, ch) in enumerate(charset)}

    if char == 'all':
        for font_id, font in enumerate(font_data):
            for ch_id, ch in enumerate(font):
                img = grayscale_to_rgb(ch)
                img.save(out + str(font_id) + '_' + str(ch_id) + '.png')
            if n_subset != 0 and font_id + 1 == n_subset:
                break
    else:
        for font_id, font in enumerate(font_data):
            ch = font[mapping[char]]
            img = grayscale_to_rgb(ch)
            img.save(out + str(font_id) + '_' + ch + '.png')
            if n_subset != 0 and font_id + 1 == n_subset:
                break


def image_to_64x64(input_image, output_filename='out.png'):
    """
    Convert input image from NIST SD 19 to 64x64 RGB image

    :param input_image: image filename
    :param output_filename: output image filename
    :return: 3-channel RGB image corresponding to input image
    """
    image = PIL.Image.open(input_image).convert('RGB')
    image.save(output_filename)


if __name__ == '__main__':
    print 'Hello world. Welcome to utilities.'
