from os.path import join
import os

_DEFAULT_SEED=41


def create_filenames_list_file(filenames_list, fname):
    f = open(fname, 'w')
    for name in filenames_list:
        f.write(name + '\n')

    ## Remove last \n
    f.seek(0, os.SEEK_END) # go end
    f.seek(f.tell() - 1, os.SEEK_SET) # go back 1
    f.truncate()
    f.close()


def open_img_list_file(fname):
    list_of_files = open(fname, 'r').read().split('\n')
    list_of_files.remove('')
    return list_of_files


def generate_split(feats, out_path='./', seed=_DEFAULT_SEED):
    ######## INIT  -- Open the original data dict
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    if isinstance(feats, str):
        data_dict = np.load(feats).item()
    elif isinstance(feats, dict):
        data_dict = feats

    labels = np.squeeze(np.array(data_dict['labels']))
    filenames = data_dict['filenames']
    filenames_np = np.array(filenames)

    os.makedirs(out_path, exist_ok=True)
    boot_file_path = join(out_path, 'boot.txt')
    train_wo_boot_file_path = join(out_path, 'train_wo_boot.txt')
    train_file_path = join(out_path, 'train.txt')
    valid_file_path =join(out_path, 'val.txt')
    trainval_file_path =join(out_path, 'trainval.txt')
    all_file_path =join(out_path, 'all.txt')


    # The difference of trainval from all is just the file-ordering/indices
    create_filenames_list_file(filenames, all_file_path)

    #########################################
    boot_indices = []
    for cls in sorted(set(labels)):
        index_for_cls = np.squeeze(np.argwhere(labels == cls))
        fnames_for_cls = filenames_np[index_for_cls]

        boot_found = False

        for i, f in enumerate(fnames_for_cls):
            if f.split('/')[1].startswith('seed_000'):
                boot_indices.append(index_for_cls[i])
                boot_found = True
                break

        if not boot_found:
            for i, f in enumerate(fnames_for_cls):
                if f.split('/')[1].startswith('google_000'):
                    boot_indices.append(index_for_cls[i])
                    boot_found = True
                    break

        if boot_found is False:
            raise RuntimeError("Can't find a bootstrap element for class {}".format(cls))

    create_filenames_list_file(filenames_np[boot_indices], boot_file_path)



    ########################### TRAIN-VALID SPLIT CLASS-WISE (taking into account unbalanced classes)


    VALID_SPLIT=0.20
    train_wo_boot_indices = []
    valid_indices = []
    for l in set(labels):
        indices = list(np.squeeze(np.argwhere(labels == l)))
        indices.remove(boot_indices[l]) # REMOVING BOOSTRAP ELEMENTS!!

        ## TODO: Take into account that flickr and google images should be splitted separately!!
        indices = np.array(indices)[np.random.permutation(len(indices))]
        separator_elem = int(len(indices)*(1 - VALID_SPLIT))
        twb_indices = np.squeeze(indices[:separator_elem])
        v_indices = np.squeeze(indices[separator_elem:])
        train_wo_boot_indices.extend(list(twb_indices))
        valid_indices.extend(list(v_indices))


    create_filenames_list_file(filenames_np[train_wo_boot_indices], train_wo_boot_file_path)
    create_filenames_list_file(np.concatenate([filenames_np[boot_indices],
                                               filenames_np[train_wo_boot_indices]]), train_file_path)
    create_filenames_list_file(filenames_np[valid_indices], valid_file_path)
    create_filenames_list_file(np.concatenate([filenames_np[boot_indices],
                                               filenames_np[train_wo_boot_indices],
                                               filenames_np[valid_indices]]) , trainval_file_path)






import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply train/valid/bootstrap split to features from trainval image folder.')

    parser.add_argument('filename', type=str, action="store",
                        help='Filename or path of the features to use for generating the split.')

    parser.add_argument('-o', '--output-path', type=str, action="store",
                        help='Path of the directory in which the generated split files will be written into.')

    parser.add_argument('-s', '--seed', type=int, action="store", default=None,
                        help='Seed to use for split generation.')


    args = parser.parse_args()
    generate_split(args.filename, args.output_path, args.seed)


# Example of usage from shell:
# python generate_split.py data/features/resnet50.npy -o data/splits/new_split
