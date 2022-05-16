import numpy as np
import PIL
import cv2


def resize_clip(clip, size, interpolation='bilinear'):
    """
    Resize the frames of an input clip
    :param clip: List of images (each image is either a numpy.ndarray or a PIL.Image.Image)
    :param size: Target size (either int/float or a (height, width) tuple)
    :param interpolation: Interpolation algorithm, either bilinear or nearest (neighbor)
    :return: Resized clip
    """

    if isinstance(clip[0], np.ndarray):
        if isinstance(size, (int, float)):
            img_h, img_w, _ = clip[0].shape  # shape: (H x W x C)
            if min(img_h, img_w) == size:
                # min spatial dimension already matches min image size
                return clip
            size = get_sizes(img_h, img_w, size)
        size = size[::-1]  # cv2.resize expects a (width, height) tuple
        if interpolation == 'bilinear':
            interp = cv2.INTER_LINEAR
        else:
            interp = cv2.INTER_NEAREST
        scaled_clip = [cv2.resize(img, size, interpolation=interp) for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, (int, float)):
            img_w, img_h = clip[0].size  # PIL.Image shape (W x H)
            if min(img_h, img_w) == size:
                return clip
            size = get_sizes(img_h, img_w, size)
        size = size[::-1]
        if interpolation == 'bilinear':
            interp = PIL.Image.BILINEAR
        else:
            interp = PIL.Image.NEAREST
        scaled_clip = [img.resize(size, interp) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image.Image' +
                        'but got list of {}'.format(type(clip[0])))

    return scaled_clip


def crop_clip(clip, min_h, min_w, h, w):
    """
    Crop a list of frames
    :param clip: List of images (each image is either a numpy.ndarray or a PIL.Image.Image)
    :param min_h: Minimum height (i.e. the row where the crop starts)
    :param min_w: Minimum width (i.e. the column where the crop starts)
    :param h: Total height of the crop
    :param w: Total width of the crop
    :return: List of cropped frames
    """
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        # box: (left, upper, right, lower)
        cropped = [img.crop(box=(min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image.Image' +
                        'but got list of {}'.format(type(clip[0])))

    return cropped


def get_sizes(img_height, img_width, size):
    """
    Get height and width of an image after resizing
    :param img_height: Original image height
    :param img_width: Original image width
    :param size: Target size
    :return: Target image height and width
    """

    if img_width < img_height:
        output_width = size
        output_height = int(size * img_height / img_width)
    else:
        output_width = int(size * img_width / img_height)
        output_height = size

    return output_height, output_width



