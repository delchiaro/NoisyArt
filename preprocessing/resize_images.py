
from ._cfg import *
from .utils import assign_if_bool, get_loader

import os

def resize_noisyart_images(dataset_imgs_path, im_resize=True, crop=False, out_path=None, workers=10, img_format='png'):
    img_format = img_format.lower()
    img_format = 'jpg' if img_format == 'jpeg' else img_format
    assert img_format in ['png', 'jpg']
    dataset_imgs_path = dataset_imgs_path[:-1] if dataset_imgs_path[-1] is '/' else dataset_imgs_path

    im_resize = assign_if_bool(im_resize, DEFAULT_IM_RESIZE)
    crop = assign_if_bool(crop, DEFAULT_IM_CROP)
    assert im_resize is not None or crop is not None  # This function have to apply at least one of the two operations (resize/crop)

    if out_path is None:
        out_path = dataset_imgs_path
        if im_resize is not None:
            out_path += f'_r{im_resize}'
        if crop is not None:
            out_path += f"_c{crop}"
        out_path += f'_{img_format}'

    os.makedirs(out_path, exist_ok=False)

    dataset_imgs_path = DEFAULT_IMGS_PATH if dataset_imgs_path is None else dataset_imgs_path

    loader = get_loader(1, im_resize, crop, False, False, False, shuffle=False,
                        dataset_imgs_path=dataset_imgs_path,
                        verbose=True, workers=workers)

    filenames = [f.split(dataset_imgs_path + '/')[1] for f, l in loader.dataset.samples]

    from scipy.misc import imsave
    from progressbar import progressbar  # use progressbar2 from pip
    for (X, Y), fname in progressbar(zip(loader, filenames)):
        # for img, fname in progressbar(zip(X, filenames)):
        # matplotlib.image.imsave(join(out_dir_path, fname), img.permute(1,2,0))
        path = os.path.join(out_path, fname)
        try:
            os.makedirs('/'.join(path.split('/')[0:-1]), exist_ok=True)
        except FileExistsError:
            pass
        path = '.'.join(path.split('.')[:-1]) + f'.{img_format}'

        format = 'jpeg' if img_format == 'jpg' else img_format  # imsave wants jpeg not jpg
        imsave(path, X[0].permute(1, 2, 0), format=format)


