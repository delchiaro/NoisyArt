from ._cfg import *
from .utils import get_loader
import torch, torch.cuda, torch.utils, torch.backends, torch.utils.data, torch.multiprocessing, torch.backends.cudnn
import numpy as np
from torch.nn import Module





class FakeLayer(Module):
    def __init__(self,):
        super(FakeLayer, self).__init__()

    def forward(self, input):
        return input

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'\
            .format(self.in_features, self.out_features, self.bias is not None)



def preprocess_noisyart_images(net_selector, im_resize, crop,  #divisor,
                               mean_sub, std_norm, range_255, batch_size,
                               dataset_imgs_path=None, feats_out_dir_path=None, feats_out_append=None, workers=0,
                               feature_layer=None, verbose=False, device=None, seed=DEFAULT_SEED):
    if seed is not None:
        print(f"Using fixed seed = {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    from torchvision.models.alexnet import alexnet
    from torchvision.models.vgg import vgg16
    from torchvision.models.vgg import vgg19
    from torchvision.models.resnet import resnet50
    from torchvision.models.resnet import resnet101
    from torchvision.models.resnet import resnet152

    dataset_imgs_path = DEFAULT_IMGS_PATH if dataset_imgs_path is None else dataset_imgs_path


    feats_out_dir_path = str(DEFAULT_FEATS_PATH) if feats_out_dir_path is None else feats_out_dir_path
    feats_out_append = "" if feats_out_append is None else '_' + feats_out_append

    # divisor = assign_if_bool(divisor, case_true=255)
    # rescale=1./divisor if divisor is not None else None
    #
    # print("divisor: {}".format(divisor))
    # print("rescale: {}".format(rescale))
    print('\n\n')
    loader = get_loader(batch_size, im_resize, crop, mean_sub, std_norm, range_255, shuffle=False,
                        dataset_imgs_path=dataset_imgs_path,
                        verbose=True, workers=workers)



    #%%
    #labels = [int(l) for f, l in loader.dataset.samples]
    dir_path = dataset_imgs_path if dataset_imgs_path.endswith('/') else dataset_imgs_path + '/'
    filenames = [f.split(dir_path)[1] for f, l in loader.dataset.samples]
    class_indices = loader.dataset.class_to_idx

    resize_str = '_r{}'.format(im_resize) if im_resize else ''
    crop_str = '_c{}'.format(crop) if crop else ''
    mean_sub_str = '_meansub' if mean_sub else ''
    std_norm_str = '_stdnorm' if std_norm else ''
    range_255_str = '_range255' if range_255 else '' 
    #divisor_str = '_div{}'.format(divisor) if divisor is not None else ''
    #duplicate_seed_std = '_dseed{}'.format(duplicate_seeds) if duplicate_seeds else ''


    def get_save_path(feat_net_name):
        from os.path import join
        return join(feats_out_dir_path, '{}{}{}{}{}{}{}'.\
            format(feat_net_name, resize_str, crop_str, mean_sub_str, std_norm_str, range_255_str, feats_out_append))



    savepath = get_save_path(net_selector)
    print('After prediction, features will be saved in: ' + savepath)

    #%%

    # As an alternative to get layer outputs look here: https://www.stacc.ee/extract-feature-vector-image-pytorch/

    if net_selector.startswith('resnet'):
        if net_selector == 'resnet50':
            net = resnet50(pretrained=True)
        elif net_selector == 'resnet101':
            net = resnet101(pretrained=True)
        elif net_selector == 'resnet152':
            net = resnet152(pretrained=True)

        if feature_layer is None or feature_layer in ['pool', 'avgpool']:
            net.fc = FakeLayer() # remove useless layers
            #net.fc = nn.AdaptiveAvgPool2d(1024) # remove useless layers
        else:
            raise RuntimeError("resnet feature_layer can only be 'avgpool' ('pool' or None for short)")


    elif net_selector.startswith('vgg'):
        if net_selector == 'vgg16':
            net = vgg16(pretrained=True)
        elif net_selector == 'vgg19':
            net = vgg19(pretrained=True)

        default_feature_layer = 'fc7'
        feature_layer = default_feature_layer if feature_layer is None else feature_layer # default layer is fc7


        if feature_layer == 'fc6':
            l_index = 0 # layer 0 is FC6, we wont layer 0 output -> remove the next layers (1 to last)
        elif feature_layer == 'fc7':
            l_index = 3 # layer 3 is FC7, we wont layer 3 output -> remove the next layers (4 to last)
        else:
            raise RuntimeError("vgg feature_layer can only be 'fc6' or 'fc7' (None for {})".format(default_feature_layer))
        for i in range(l_index+1, len(net.classifier)):
            net.classifier[i] = FakeLayer()

    elif net_selector == 'alexnet':
        net = alexnet(pretrained=True)
        net.classifier = FakeLayer()  # remove useless layers



    print('Start prediction')
    from progressbar import progressbar

    preds = []
    labels = []

    # Fix for:  RuntimeError: received 0 items of ancdata
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    from PIL import ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True # Allow read truncated files. Otherwise truncated file will except and kill!
    Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS * 4 # Change the max pixel for 'decompressionBomb' warning


    if device is not None and device is not 'cpu':
        print("Using CUDA")
        net.to(device)
        net.eval()
        suffix = '\n' if verbose else ''
        for X, Y in progressbar(loader, suffix=suffix):
            if verbose:
                print("\nMax-Val: {}".format(X.max()))
                print("Min-Val: {}".format(X.min()))
                print("Mean:    {}".format(X.mean()))
                print("STD:     {}\n".format(X.std()))
            preds.append(net(X.to(device)).detach().cpu().numpy())
            labels.append(Y)
    else:
        print("Using CPU")
        for X, Y in progressbar(loader):
            preds.append(net(X).detach().numpy())
            labels.append(Y)

    preds = np.vstack(preds)
    labels = np.concatenate(labels)
    #labels = np.array(labels)

    print('Saving preds to: ' + savepath)
    saved = False

    while not saved:
        try:
            np.save(savepath, {'feats': preds, 'labels': labels, 'filenames': filenames, 'class_indices': class_indices})
            saved = True
        except MemoryError as E:
            import traceback
            from time import sleep
            traceback.print_exc()
            print('\n\nMemory Error')
            print("Waiting 30 seconds and will try again..")
            import gc
            gc.collect()
            sleep(60.0)

