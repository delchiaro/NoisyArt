import numpy as np
import torch
from pathlib import Path
from exp_tools.utils import unison_shuffled_copies
from typing import Tuple
import torch.utils.data
from preprocessing import split_data_dict
from ._cfg import *


class NoisyArtFeats:
    def __init__(s,
                 trainval_feats_name=None,
                 test_feats_name=None,
                 noisyart_path=None,
                 feat_folder=None,
                 split_folder=None,
                 split_name=None,
                 bootstrap=False):
        trainval_feats_name = DEFAULT_TRAINVAL_FEATS_NAME if trainval_feats_name is None else trainval_feats_name
        test_feats_name = DEFAULT_TEST_FEATS_NAME if test_feats_name is None else test_feats_name
        noisyart_path = DEFAULT_NOISYART_PATH if noisyart_path is None else noisyart_path
        split_name = DEFAULT_SPLIT_NAME if split_name is None else split_name
        feat_folder = DEFAULT_FEATS_FOLDER if feat_folder is None else feat_folder
        split_folder = DEFAULT_SPLITS_FOLDER if split_folder is None else split_folder

        s.noisyart_path = Path(noisyart_path)
        s.noisyart_feats_path = s.noisyart_path / feat_folder
        s.noisyart_splits_path = s.noisyart_path / split_folder
        s.use_bootstrap = bootstrap

        if not test_feats_name.endswith('.npy'):
            test_feats_name = test_feats_name + '.npy'
        if not trainval_feats_name.endswith('.npy'):
            trainval_feats_name = trainval_feats_name + '.npy'

        s.test_dict = np.load(s.noisyart_feats_path / test_feats_name).item()
        if split_name is not None:
            _dicts = split_data_dict(str(s.noisyart_feats_path / trainval_feats_name),
                                     str(s.noisyart_splits_path / split_name),
                                     save_splitted_dicts=False)
            s.train_dict, s.val_dict, s.trainval_dict, s.boot_dict, s.train_wo_boot_dict = _dicts

        else:
            s.train_dict = np.load(s.noisyart_feats_path / (trainval_feats_name + "_train.npy")).item()
            s.val_dict = np.load(s.noisyart_feats_path / (trainval_feats_name + "_val.npy")).item()
            s.trainval_dict = np.load(s.noisyart_feats_path / (trainval_feats_name + "_trainval.npy")).item()
            if s.use_bootstrap:
                s.boot_dict = np.load(s.noisyart_feats_path / (trainval_feats_name + "_boot.npy")).item()
                s.train_wo_boot_dict = np.load(s.noisyart_feats_path / (trainval_feats_name + "train_wo_boot.npy")).item()
            else:
                s.boot_dict = None
                s.train_wo_boot_dict = None

        # Converting locals labels of test-set to global labels of trainval-set
        s.class_mapping = {}
        for cls, lbl in s.test_dict['class_indices'].items():
            global_lbl = s.val_dict['class_indices'][cls]
            s.class_mapping[lbl] = global_lbl
            s.test_dict['labels_local'] = s.test_dict['labels']
        s.test_dict['labels'] = np.array([s.class_mapping[lbl] for lbl in s.test_dict['labels']])

        class ClassFilters:
            def __init__(self, waf):
                self.revised_class_only = set(waf.test_dict['labels'])
                self.revised_class_only = sorted(self.revised_class_only)

        s.filters = ClassFilters(s)

        s.class_mapping = {cls: index for index, cls in enumerate(s.filters.revised_class_only)}
        s.class_mapping_inv = {small: full for full, small in s.class_mapping.items()}
        s.class_names = list(s.test_dict['class_indices'].keys())

    def class_name_by_full_label(self, label):
        return self.class_names[label]

    def class_name_by_revised_label(self, test_label):
        return self.class_names[self.class_mapping_inv[test_label]]

    def get_data_dict(self, dset='train'):
        if dset in ['train']:
            return self.train_dict
        elif dset in ['valid', 'val']:
            return self.val_dict
        elif dset in ['test']:
            return self.test_dict
        elif dset in ['boot', 'bootstrap']:
            return self.boot_dict
        else:
            raise ValueError("dset should be 'train`, 'val`, 'test`")

    def get_data_loader(self,
                        dset: str = 'train',
                        BS: int = 32,
                        shuffle: bool = False,
                        class_filter: Tuple[int] = None,
                        device=None,
                        additional_np_data: Tuple[np.ndarray] = ()):

        data_dict = self.get_data_dict(dset)
        return self.__prepare_data(data_dict, BS, shuffle, class_filter, device, additional_np_data)

    def __prepare_data(self, data_dict, batch_size=32, shuffle=True, class_filter=None, device=None,
                       additional_np_data: Tuple[np.ndarray] = ()):
        if data_dict is None:
            return None
        Y = data_dict['labels'].astype(np.long)
        X = data_dict['feats']

        if class_filter is not None:
            class_filter = sorted(class_filter)
            class_mapping = {cls: index for index, cls in enumerate(class_filter)}
            args = np.argwhere([y in class_filter for y in Y])
            args = np.squeeze(args)
            Y = np.array([class_mapping[y] for y in Y[args]])
            X = X[args]

        arrays = [X, Y]
        if len(additional_np_data) > 0:
            arrays = arrays + additional_np_data
        for A in arrays:
            assert len(A) == len(X)
        arrays = [torch.from_numpy(A) for A in arrays]

        if shuffle:
            arrays = unison_shuffled_copies(arrays)

        if device is not None:
            arrays = [A.to(device) for A in arrays]

        dataset = torch.utils.data.TensorDataset(*arrays)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader
