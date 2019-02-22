import argparse
from preprocessing import split_data_dict


# NB: The use of this procedure is not necessary: NoisyArtFeats objects will apply split at run-time.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply train/valid/bootstrap split to features from trainval image folder.')

    parser.add_argument('filename', type=str, action="store",
                        help='Filename or path of the features that will be splitted using the selected split.')

    parser.add_argument('split_dir', type=str, action="store",
                        help='Path of the directory that contains the split files '
                             '(train.txt, val.txt, trainval.txt, boot.txt, train_wo_boot.txt).')

    parser.add_argument('-fd', '--feats-dir', type=str, action="store", default='./',
                        help='Path of the directory that contains the features. '
                             'Files containing the splitted features will be placed here.')

    args = parser.parse_args()

    split_data_dict(args.filename, args.split_dir, args.feats_dir,
                    save_splitted_dicts=True, ignore_im_extensions=True, fix_data=False)

