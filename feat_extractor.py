from preprocessing import preprocess_noisyart_images
import torch, torch.cuda
from _cfg import SEED

#%%
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute image features of NoisyArt dataset using different parameters.')
    parser.add_argument('net', type=str, action="store", choices=['alexnet', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152'],
                        help='Network to use for feature computing')
    parser.add_argument('-d', '--data-path', type=str, action="store", default=None,
                        help="The path of the data folder, in which images are stored.")
    parser.add_argument('-o', '--out-path', type=str, action="store", default=None,
                        help="The path of the output feature folder.")
    parser.add_argument('-a', '--append', type=str, action="store", default=None,
                        help="Append .")

    parser.add_argument('-s', '--size', type=int, action="store", default=255,
                        help="The resolution at which the image will be resized. Default=255.")
    parser.add_argument('-c', '--crop', type=int, action="store", default=224,
                        help="The resolution at which images will be center-cropped.  Default=224.")

    parser.add_argument('-mean', '--mean', action="store_true", default=False,
                        help="Use ImageNet channel-mean subtraction over input images.")
    parser.add_argument('-std', '--std', action="store_true", default=False,
                       help="Use ImageNet channel-std normalization over input images.")

    parser.add_argument('-b', '--batch-size', type=int, action="store", default=32,
                        help="Specify how many images to process in parallel with the CNN.")

    parser.add_argument('-g', '--gpu', type=int, action="store", default=-1,
                        help="Chose the GPU (cuda device) on which run the script. Default: NO GPU (run on CPU).")

    parser.add_argument('-w', '--workers', type=int, action="store", default=0,
                        help="How many CPU workers to read images and preprocess.")

    parser.add_argument('-l', '--layer', type=str, action="store", default=None,
                        help="Layer in the CNN from which extract the features.")

    parser.add_argument('-r255', '--range-255', action="store_true",
                        help="This flag force to read the image with values from 0 to 255, instead of 0 to 1."
                             "This option is experimental, not tested.")

    parser.add_argument('--seed', type=int, action="store", default=SEED,
                        help="Specify the fixed seed to use for computation.")


    parser.add_argument('-v', '--verbose', action='store_true')

    # parser.add_argument('-d', '--divisor', type=int, action="store", default=1,
    #                     help="Divisor for image pixel colors, use 1 to use original input range (usually 0-255), "
    #                          "use 255 if you want to have network input to be in range 0-1 (for tipical 0-255 inputs).")

    # parser.add_argument('-ds', '--duplicate-seeds', type=int, action="store", default=0,
    #                     help="Specify how many times the seed images should be duplicated.")
    # parser.add_argument('-seed', '--seed-only', type=int, action="store_true",
    #                     help="Process only seed images")
    # parser.add_argument('-google', '--google-only', type=int, action="store_true",
    #                     help="Process only google images")
    # parser.add_argument('-flickr', '--flickr-only', type=int, action="store_true",
    #                     help="Process only flickr images")

    args = parser.parse_args()

    # INIT GPU AND SEED SETTINGS
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
    print("Torch is using device: {}".format(device))

    # divisor = False if args.divisor == 1 else args.divisor
    preprocess_noisyart_images(args.net, args.size, args.crop,  #divisor,
                               args.mean, args.std, args.range_255,
                               args.batch_size, args.data_path, args.out_path, args.append,
                               args.workers,
                               device=device,
                               verbose=args.verbose,
                               feature_layer=args.layer,
                               seed=args.seed)



