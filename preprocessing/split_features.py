from ._cfg import *
import numpy as np
from os.path import join



def open_img_list_file(fname):
    list_of_files = open(fname, 'r').read().split('\n')
    list_of_files.remove('')
    return list_of_files


def fix_data_dict(data_dict:dict):
    return {'feats': np.array(data_dict['feats']), 'labels': np.array(data_dict['labels']),
            'filenames': data_dict['filenames'], 'class_indices': data_dict['class_indices']}

def sub_data_dict(data_dict: dict, indices: list):
    return {'feats': data_dict['feats'][indices],
            'labels': data_dict['labels'][indices],
            'filenames': list(np.array(data_dict['filenames'])[indices]),
            'class_indices': data_dict['class_indices']}


def split_data_dict(feats_fname:str,
                    split_dir_path=DEFAULT_SPLIT_PATH,
                    feats_dir_path=DEFAULT_FEATS_PATH,
                    save_splitted_dicts=False,
                    ignore_im_extensions=True,
                    fix_data=False):
    save_path = '.'.join(feats_fname.split('.')[:-1])
    save_path = join(feats_dir_path, save_path)
    if feats_fname.startswith('/'):
        feats_dir_path = '' # using absolute path


    data_dict: dict = np.load(join(feats_dir_path, feats_fname)).item()

    train_files = open_img_list_file(join(split_dir_path, 'train.txt'))
    valid_files = open_img_list_file(join(split_dir_path, 'val.txt'))
    boot_files = open_img_list_file(join(split_dir_path, 'boot.txt'))
    train_wo_boot_files = open_img_list_file(join(split_dir_path, 'train_wo_boot.txt'))
    split_file_lists = [train_files, valid_files, boot_files, train_wo_boot_files]

    if ignore_im_extensions:
        for i in range(len(split_file_lists)):
            split_file_lists[i] = ['.'.join(f.split('.')[:-1]) for f in split_file_lists[i]]
        name_to_index = {'.'.join(k.split('.')[:-1]) : val for val, k in enumerate(data_dict['filenames'])}
    else:
        name_to_index = {k : val for val, k in enumerate(data_dict['filenames'])}

    for i in range(len(split_file_lists)):
        split_file_lists[i] = set(name_to_index.keys()).intersection(set(split_file_lists[i]))

    indices_list = [[name_to_index[name] for name in sfl ] for sfl in split_file_lists]
    out_dicts = [sub_data_dict(data_dict, indices) for indices in indices_list]

    if fix_data:
        out_dicts = [fix_data_dict(d) for d in out_dicts]

    train_dict, valid_dict, boot_dict, train_wo_boot_dict = out_dicts
    trainval_dict = data_dict
    if save_splitted_dicts:
        np.save(save_path + '_train', train_dict)
        np.save(save_path + '_val', valid_dict)
        np.save(save_path + '_trainval', trainval_dict)
        np.save(save_path + '_boot', boot_dict)
        np.save(save_path + '_train_wo_boot', train_wo_boot_dict)

    return train_dict, valid_dict, trainval_dict, boot_dict, train_wo_boot_dict




#%%


