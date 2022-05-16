import random
import numpy as np
import PIL
import torchvision.transforms.functional as vF
from utils.datatransforms import functional as F
import math


class Compose(object):
    """
    Compose a chain of transforms
    Args:
        transforms: list of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)

        return clip


class RandomDrop(object):
    """
    Randomly drop frames from input video to match a predefined number of frames.
    """
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, clip):
        assert len(clip) >= self.num_frames
        if len(clip) == self.num_frames:
            return clip
        indices = sorted(random.sample(range(len(clip)), k=self.num_frames))
        return [clip[i] for i in indices]


class RandomHorizontalFlip(object):
    """
    Randomly flip a list of images horizontally (with probability 0.5)
    """

    def __call__(self, clip):
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
            else:
                raise TypeError('Expected either numpy.ndarray or PIL.Image.Image' +
                                'but got list of {}'.format(type(clip[0])))
        return clip


class RandomGray(object):
    """
    Randomly transform an RGB Image into Grayscale (with probability 0.3)
    """

    def __call__(self, clip):
        return [vF.to_grayscale(img) if random.random() < 0.3 else img for img in clip]


class Resize(object):
    """
    Resize a list of images to a target size
    Args:
        size: Target size, either a (height, width) tuple or an int/float (in this case the
                resized image shape is inferred from the original image shape)
        interpolation: Interpolation algorithm, either bilinear or nearest (neighbor)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        return F.resize_clip(clip, self.size, self.interpolation)


class RandomResize(object):
    """
    Resize a list of images based on a randomly chosen aspect ratio
    Args:
        ratio: Aspect ratio lower/upper bound (tuple)
        interpolation: Interpolation algorithm, either bilinear or nearest (neighbor)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(*self.ratio)
        if isinstance(clip[0], np.ndarray):
            img_h, img_w, _ = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            img_w, img_h = clip[0].size
        # get new height, width
        new_h = int(scaling_factor * img_h)
        new_w = int(scaling_factor * img_w)

        return F.resize_clip(clip, (new_h, new_w), self.interpolation)


class CenterCrop(object):
    """
    Extract center crop from a list of images
    Args:
        size: Crop size (either int or (height, width) tuple)
    """

    def __init__(self, size):
        if isinstance(size, (int, float)):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            img_h, img_w, _ = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            img_w, img_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image.Image' +
                            'but got list of {}'.format(type(clip[0])))
        if crop_h > img_h or crop_w > img_w:
            raise ValueError('Image size should be larger than crop size.' +
                             'Got image shape {} and crop shape {}'.format((img_h, img_w), (crop_h, crop_w)))
        # center crop's start height and width
        min_h = img_h // 2 - crop_h // 2
        min_w = img_w // 2 - crop_w // 2

        return F.crop_clip(clip, min_h, min_w, crop_h, crop_w)


class RandomCrop(object):
    """
    Extract random crop from a list of images
    Args:
        size: Crop size (either int or (height, width) tuple)
    """

    def __init__(self, size):
        if isinstance(size, (int, float)):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            img_h, img_w, _ = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            img_w, img_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image.Image' +
                            'but got list of {}'.format(type(clip[0])))
        if crop_h > img_h or crop_w > img_w:
            raise ValueError('Image size should be larger than crop size.' +
                             'Got image shape {} and crop shape {}'.format((img_h, img_w), (crop_h, crop_w)))
        # crop's start height and width (randomly chosen)
        min_h = random.randint(0, img_h - crop_h)
        min_w = random.randint(0, img_w - crop_w)

        return F.crop_clip(clip, min_h, min_w, crop_h, crop_w)


class TenCrop(object):
    """
    Generate 10 crops for each frame in the clip
    Args:
        size: Crop size (either int or (height, width) tuple)
    """

    def __init__(self, size):
        if isinstance(size, (int, float)):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        cropped = []
        for img in clip:
            cropped += list(vF.ten_crop(img, self.size))

        return cropped


class RandomResizedCrop(object):
    """
    Crop a list of images to a random size and aspect ratio
    Args:
        size: Crop size
        scale: Scale range, i.e. the size of the crop wrt the original image (tuple)
        ratio: Aspect ratio range (tuple)
        interpolation: Interpolation algorithm, either bilinear or nearest (neighbor)

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, (int, float)):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = PIL.Image.BILINEAR if interpolation == 'bilinear' else PIL.Image.NEAREST

    @staticmethod
    def get_params(img, scale, ratio):
        """
        Create a crop box
        :param img: Input image (PIL.Image.Image)
        :param scale: Scale range, i.e. the size of the crop wrt the original image (tuple)
        :param ratio: Aspect ratio range (tuple)
        :return: height of top left corner, width of top left corner, height, width of the crop box (tuple)
        """

        # area of the image
        area = np.prod(img.size)
        # randomly sample a valid crop width/height
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = [math.log(r) for r in ratio]
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            crop_w = round(math.sqrt(target_area * aspect_ratio))  # aspect ratio is width / height
            crop_h = round(math.sqrt(target_area / aspect_ratio))

            if crop_w <= img.size[0] and crop_h <= img.size[1]:
                top = random.randint(0, img.size[1] - crop_h)
                left = random.randint(0, img.size[0] - crop_w)
                return top, left, crop_h, crop_w

        # if it gets here, extract a center crop
        img_ratio = np.divide(*img.size)
        if img_ratio < min(ratio):
            # image width is small --> use min aspect ratio
            crop_w = img.size[0]
            crop_h = round(crop_w / min(ratio))
        elif img_ratio > max(ratio):
            # image height is small --> use max aspect ratio
            crop_h = img.size[1]
            crop_w = round(crop_h * max(ratio))
        else:
            # get the entire image
            crop_w, crop_h = img.size
        top = img.size[1] // 2 - crop_h // 2
        left = img.size[0] // 2 - crop_w // 2

        return top, left, crop_h, crop_w

    def __call__(self, clip):
        box = self.get_params(clip[0], self.scale, self.ratio)
        return [vF.resized_crop(img, *box, self.size, self.interpolation) for img in clip]


class ColorJitter(object):
    """
    Randomly change the brightness, contrast, saturation and hue of a clip
    Args:
        brightness: Amount of brightness jitter (float), chosen randomly from
            [max(0, 1 - brightness), 1 + brightness]
        contrast: Amount of contrast jitter (float), chosen randomly from
            [max(0, 1 - contrast), 1 + contrast]
        saturation: Amount of saturation jitter (float), chosen randomly from
            [max(0, 1 - saturation), 1 + saturation]
        hue: Amount of hue jitter (float), chosen randomly from [-hue, hue].
            Input value should be in [0, 0.5] range.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness = None

        if contrast > 0:
            contrast = random.uniform(max(0, 1 - contrast), 1 + contrast)
        else:
            contrast = None

        if saturation > 0:
            saturation = random.uniform(max(0, 1 - saturation), 1 + saturation)
        else:
            saturation = None

        if 0 <= hue <= 0.5:
            hue = random.uniform(-hue, hue)
        else:
            hue = None

        return brightness, contrast, saturation, hue

    def __call__(self, clip):
        """
        Color jitter an entire clip
        :param clip: List of frames (PIL.Image.Image)
        :return: Color jittered clip
        """

        assert type(clip[0]) == PIL.Image.Image, "Expected frames of type PIL.Image.Image"
        brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast,
                                                                self.saturation, self.hue)
        # create augmentation sequence
        img_transforms = []
        if brightness:
            img_transforms.append(lambda img: vF.adjust_brightness(img, brightness))
        if contrast:
            img_transforms.append(lambda img: vF.adjust_contrast(img, contrast))
        if saturation:
            img_transforms.append(lambda img: vF.adjust_saturation(img, saturation))
        if hue:
            img_transforms.append(lambda img: vF.adjust_hue(img, hue))
        # shuffle transforms
        random.shuffle(img_transforms)

        # apply transforms to frames
        jittered_clip = []
        for img in clip:
            for t in img_transforms:
                img = t(img)
            jittered_clip.append(img)

        return jittered_clip


class TemporalJitter(object):
    """
    Temporal jittering, i.e. sample multiple clips from each video with random start times
    Args:
        n_frames: Number of output frames (int)
        time_scale: Range of subsampling rate (tuple)
        crop_type: Which segment of the subsampled video should be returned (either center
            or random)
    """

    def __init__(self, n_frames, time_scale=(1., 1.), crop_type='center'):
        self.n_frames = n_frames
        self.time_scale = time_scale
        assert crop_type in ('random', 'center'), "Accepted crop types: 'random' or 'center'"
        self.crop_type = crop_type

    def __call__(self, clip):
        if len(set(self.time_scale)) == 1:
            rate = self.time_scale[0]
        else:
            rate = random.uniform(self.time_scale[0], min(self.time_scale[1], len(clip) / self.n_frames))
        # subsampled clip
        clip_ss = [clip[int(t)] for t in np.arange(0, len(clip), rate)]
        if len(clip_ss) == self.n_frames:
            clip_out = clip_ss
        elif len(clip_ss) < self.n_frames:
            # fill the subsampled clip with more frames
            clip_out = [clip_ss[t % len(clip_ss)] for t in range(self.n_frames)]
        else:
            # extract crop from the subsampled clip
            if self.crop_type == 'random':
                t_start = random.randint(0, len(clip_ss) - self.n_frames)
            else:
                t_start = len(clip_ss) // 2 - self.n_frames // 2
            clip_out = clip_ss[t_start:t_start + self.n_frames]

        return clip_out
