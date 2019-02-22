from preprocessing import resize_noisyart_images
from preprocessing._cfg import DEFAULT_IMGS_PATH
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and resize/crop images of NoisyArt images')

    parser.add_argument('-d', '--data-path', type=str, action="store", default=DEFAULT_IMGS_PATH,
                        help="The path of the data folder, in which images are stored.")
    parser.add_argument('-o', '--out-path', type=str, action="store", default=None,
                        help="The path of the output feature folder.")

    parser.add_argument('-s', '--size', type=int, action="store", default=255,
                        help="The image will be resized such that the short side will be at the selected size. "
                             "Standard size is 255. "
                             "Use False to disable resizing.")
    parser.add_argument('-c', '--crop', type=int, action="store", default=False,
                        help="The resolution at which images will be center-cropped. "
                             "You could crop during training keeping this value to False. "
                             "Use True for standard center-crop (224). "
                             "Use an int value for a custom center-crop.")


    parser.add_argument('-f', '--format', type=str, action="store", default='png', choices=['png','jpeg'],
                        help="Image files format to use for resized images.")

    parser.add_argument('-w', '--workers', type=int, action="store", default=0,
                        help="How many CPU workers to read images and preprocess.")

    # read_dbp3120_dataset(args.net, args.size, False,  # args.crop, #divisor,
    #                      args.data_path, args.workers)

    # divisor = False if args.divisor == 1 else args.divisor
    args = parser.parse_args()
    resize_noisyart_images(args.data_path, im_resize=args.size, crop=args.crop,
                           out_path=args.out_path, workers=args.workers, img_format=args.format)
