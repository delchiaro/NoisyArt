from ._cfg import *

import torch
import torchvision

from torchvision import transforms



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np
def assign_if_bool(var, case_true, case_false=None):
    if isinstance(var, bool):
        if var:
            var = case_true
        else:
            var = case_false
    return var



from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None



class ToTensorWithRange(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    @staticmethod
    def _is_pil_image(img):
        if accimage is not None:
            return isinstance(img, (Image.Image, accimage.Image))
        else:
            return isinstance(img, Image.Image)

    @staticmethod
    def _is_tensor_image(img):
        return torch.is_tensor(img) and img.ndimension() == 3

    @staticmethod
    def _is_numpy_image(img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

    def __init__(self, use_range_255=True):
        self.use_range_255 = use_range_255

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. See ``ToTensor`` for more details.
            Args:
                pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            Returns:
                Tensor: Converted image.
            """
        if not (self._is_pil_image(pic) or self._is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            if isinstance(img, torch.ByteTensor):
                img = img.float()
                if not self.use_range_255:
                    img = img.div(255)
            return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            img = img.float()
            if self.use_range_255 == False:
                img=img.div(255)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '()'



def get_loader(batch_size=32,
                 resize=True,
                 crop=True,
                 mean=True,
                 std=True,
                 range_255=False,
                 shuffle=False,
                 dataset_imgs_path=None,
                 workers=0,
                 verbose=0):
    dataset = get_dataset(resize=resize, crop=crop, mean=mean, std=std, range_255=range_255, dataset_imgs_path=dataset_imgs_path, verbose=verbose)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle)
    if verbose:
        print('BS:      {}'.format(batch_size))
        print('shuffle: {}'.format(shuffle))
        print('workers: {}'.format(workers))
    return loader



def get_dataset(resize=True,
                crop=True,
                mean=True,
                std=True,
                range_255=False,
                dataset_imgs_path=None,
                verbose=0):
    data_path = DEFAULT_IMGS_PATH if dataset_imgs_path is None else dataset_imgs_path
    transformations = []

    resize = assign_if_bool(resize, DEFAULT_IM_RESIZE)
    crop = assign_if_bool(crop, DEFAULT_IM_CROP)
    use_mean = use_std = False
    if not((mean is None or mean is False) and (std is None and std is False)):
        if mean is not None and (mean is True or not isinstance(mean, bool)):
            use_mean = True
        if std is not None and (std is True or not isinstance(std, bool)):
            use_std = True
        mean = assign_if_bool(mean, DEFAULT_IM_CHANNEL_MEAN, [0, 0, 0])
        std = assign_if_bool(std, DEFAULT_IM_CHANNEL_STD, [1, 1, 1])


    # class ToTensorWithoutScaling(object):
    #     """H x W x C -> C x H x W"""
    #     def __call__(self, picture):
    #         return torch.FloatTensor(np.array(picture)).permute(2, 0, 1)

    if resize is not None:
        transformations.append(transforms.Resize(resize, interpolation=DEFAULT_IM_INTERPOLATION))
        # interpolation=Image. # LANCZOS (best, slowest), BILINEAR (torch default), NEAREST (keras default?)
    if crop is not None:
        transformations.append(transforms.CenterCrop(crop))
    if range_255:
        print('USING CUSTOM TO-TENSOR: ToTensorWithRange !!')
        transformations.append(ToTensorWithRange(use_range_255=True))
        if use_mean:
            mean=[int(m*255.) for m in mean]
        if use_std:
            std=[int(s*255) for s in std]
    else:
        transformations.append(transforms.ToTensor())
    if use_mean or use_std:
        transformations.append(transforms.Normalize(mean=mean, std=std))

    if verbose:
        print('Imgs dir:  {}'.format(data_path))
        print('Resize:    {}'.format(resize))
        print('Crop:      {}'.format(crop))
        print('Range:     {}'.format("[0-255]" if range_255 else "[0-1]"))
        print('mean:      {}'.format("not-used ({})".format(mean) if use_mean is False else mean))
        print('std:       {}'.format("not-used ({})".format(std) if use_std is False else std))

    transform = transforms.Compose(transformations)
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    return dataset


