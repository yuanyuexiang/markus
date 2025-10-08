""" This module contains functions for preprocessing signatures

"""
import numpy as np
from scipy.ndimage import interpolation
from PIL import Image

# scipy.misc已废弃,使用PIL替代
def imread(path):
    return np.array(Image.open(path))

def imsave(path, img):
    Image.fromarray(img.astype(np.uint8)).save(path)

def imresize(img, size):
    return np.array(Image.fromarray(img.astype(np.uint8)).resize(size[::-1]))


def normalize_image(img, size=(952, 1360)):
    """ Size-normalizes a signature to a given canvas size

    Centers the signature in a canvas of size (size_r x size_c), and resizes
    the signature if necessary to ensure it fits the canvas.

    Parameters:
        img (numpy.ndarray): The input image
        size (tuple): The canvas size (height, width)

    Returns:
        numpy.ndarray: The normalized image
    """

    # Crop the image to the size of the signature
    binarized_image = img < 50
    r, c = np.where(binarized_image)  # 修复：找前景(True=1)而非背景
    
    # If image is completely white/blank
    if len(r) == 0:
        return np.ones(size, dtype=np.uint8) * 255

    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
    cropped = img[r.min(): r.max(), c.min(): c.max()]

    # Center the image
    img_r, img_c = cropped.shape
    max_r, max_c = size

    r_start = max_r // 2 - r_center
    c_start = max_c // 2 - c_center

    # Make sure the new image does not go off bounds
    # Case 1: image larger than required: Crop.
    if img_r > max_r:
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_r - max_r
        crop_start = difference // 2
        cropped = cropped[crop_start: crop_start + max_r, :]

    elif img_r > max_r - r_start:
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        difference = img_r - (max_r - r_start)
        cropped = cropped[:-difference, :]

    if img_c > max_c:
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_c - max_c
        crop_start = difference // 2
        cropped = cropped[:, crop_start: crop_start + max_c]

    elif img_c > max_c - c_start:
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        difference = img_c - (max_c - c_start)
        cropped = cropped[:, :-difference]

    normalized = np.ones(size, dtype=np.uint8) * 255
    # Add the signature to the blank canvas
    r_end = r_start + cropped.shape[0]
    c_end = c_start + cropped.shape[1]

    normalized[r_start:r_end, c_start:c_end] = cropped
    return normalized


def crop_center(img, size=(952, 1360)):
    """ Crops a center of size (size_r x size_c) from an image

    Parameters:
        img (numpy.ndarray): The input image
        size (tuple): The size to crop (height, width)

    Returns:
        numpy.ndarray: The cropped image

    """
    img_shape = img.shape
    max_r, max_c = size

    # Case 1: Image is larger than the expected size, crop it
    r_start = (img_shape[0] - max_r) // 2
    r_end = r_start + max_r

    c_start = (img_shape[1] - max_c) // 2
    c_end = c_start + max_c

    cropped = img[r_start:r_end, c_start:c_end]
    return cropped


def resize_image(img, size=(170, 242)):
    """ Resizes an image to a given size

    Parameters:
        img (numpy.ndarray): The input image
        size (tuple): The desired size (height, width)

    Returns:
        numpy.ndarray: The resized image

    """
    resized = np.array(Image.fromarray(img).resize((size[1], size[0])))
    return resized


def preprocess_signature(img, canvas_size=(952, 1360)):
    """ Preprocesses a signature image

    1) Normalize the image to a given canvas size
    2) Crop the center of the image to the expected size for the CNN
    3) Resize to the input size of the network

    Parameters:
        img (numpy.ndarray): The input image
        canvas_size (tuple): The canvas size (height, width)

    Returns:
        numpy.ndarray: The preprocessed image

    """
    img = img.astype(np.uint8)
    centered = normalize_image(img, size=canvas_size)
    cropped = crop_center(centered, size=canvas_size)

    # Resize to the expected input size of the network
    resized = resize_image(cropped, size=(150, 220))

    return resized


def remove_background(img):
    """ Removes the background of the signature (sets it to 255)

    Parameters:
        img (numpy.ndarray): The input image

    Returns:
        numpy.ndarray: The image with the background removed

    """
    binarized = (img < 50).astype(np.uint8)
    return binarized * img + (1 - binarized) * 255
