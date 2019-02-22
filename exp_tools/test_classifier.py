import numpy as np
import torch, torch.cuda, torch.backends, torch.optim
from noisyart import NoisyArtFeats
from exp_tools.classifier import Classifier






def predict(net, data_loader, T=None):
    if T is not None:
        old_temp = net.temperature
        net.temperature = T
    preds = []
    y_trues = []
    for i, data in enumerate(data_loader, 0):
        x, y = data
        pred = net(x)
        preds.append(pred.cpu().detach().numpy())
        y_trues.append(y.cpu().detach().numpy())
    preds = np.concatenate(preds)
    y_trues = np.concatenate(y_trues)
    if T is not None:
        net.temperature = old_temp
    return preds, y_trues

def compute_entropy(preds, y_trues):
    from scipy.stats import entropy
    entropy = [entropy(prob) for prob in preds]
    classes = set(y_trues)
    entropy_per_class = {cls: [] for cls in classes}
    for h, cls in zip(entropy, y_trues):
        entropy_per_class[cls].append(h)

    #entropy_per_class_mean = {cls: np.mean(entropy_per_class[cls]) for cls in classes}
    #entropy_per_class_std = {cls: np.std(entropy_per_class[cls]) for cls in classes}
    entropy_per_class_mean = np.array([np.mean(entropy_per_class[cls]) for cls in classes])
    entropy_per_class_std = np.array([np.std(entropy_per_class[cls]) for cls in classes])

    return entropy, entropy_per_class, entropy_per_class_mean, entropy_per_class_std, y_trues

def compute_entropy_loader(net, data_loader, T=None):
    y_preds, y_trues = predict(net, data_loader, T)
    return compute_entropy(y_preds, y_trues)


def compute_entropy_dset(net, dset, BS=32, T=None, device=None):
    dataset = NoisyArtFeats()
    data_loader = dataset.get_data_loader(dset, BS, shuffle=False, device=device)
    return compute_entropy_loader(net, data_loader, T)

def compute_perclass_accuracy(softmax_pred, y_trues, ret_class_support=False):
    acc = np.where(np.argmax(softmax_pred, axis=1)==y_trues, 1, 0)
    classes = set(y_trues)
    acc_per_class = {cls: [] for cls in classes}
    class_support = {cls: 0 for cls in classes}
    for a, cls in zip(acc, y_trues):
        acc_per_class[cls].append(a)
        class_support[cls] += 1
    for cls in classes:
        acc_per_class[cls] = np.mean(acc_per_class[cls])

    if ret_class_support:
        return acc_per_class, class_support
    else:
        return acc_per_class


def get_acc(y_trues, pred):
    pred = np.argmax(pred, axis=1) if len(pred.shape) == 2 else pred
    errors = (y_trues - pred)
    return np.sum(errors == 0) / float(len(pred))

def entropy_filtering(y_trues, softmax_pred):
    h, h_cls, h_cls_mean, h_cls_std, labels = compute_entropy(softmax_pred, y_trues)
    argsort_h_cls_mean = np.argsort(h_cls_mean)[::-1]  # from the highest to the lowest entropy

    y_preds = np.argmax(softmax_pred, axis=1)


    acc_removing_most_entrpy_cls = [get_acc(y_trues, y_preds)]
    nb_classes = 200
    filtered_y_trues = np.array(y_trues)
    filtered_y_preds = np.array(y_preds)
    for filterd_out in range(0, nb_classes):
        label_to_remove = argsort_h_cls_mean[filterd_out]
        indices_to_remove = np.argwhere(filtered_y_trues==label_to_remove)
        filtered_y_trues = np.delete(filtered_y_trues, indices_to_remove)
        filtered_y_preds= np.delete(filtered_y_preds, indices_to_remove)
        acc_removing_most_entrpy_cls.append(get_acc(filtered_y_trues, filtered_y_preds))
    return np.array(acc_removing_most_entrpy_cls)


def run_entropy_filtering(stats_dir: str,  # dir from which load states
                          net_input_feats: int,
                          net_hidden_units: list = (4096,),
                          BS: int = 32,
                          model_dirs: str=None,
                          feats_name='resnet50_r255_c224_meansub_stdnorm',
                          trainval_feats='trainval_200',
                          test_feats='test_200',
                          states: list = None,  # [1500, 1200, 1000, 'bestval.loss', 'bestval.acc', 'besttest.acc' , ..]
                          revisedclasses_only=True,
                          device=None):
    from os import listdir
    from os.path import join

    nb_classes = 200 if revisedclasses_only else 3120

    dataset = NoisyArtFeats(trainval_feats_name=feats_name + '_' + trainval_feats,
                            test_feats_name=feats_name + '_' + test_feats,
                            bootstrap=False)

    class_filter = dataset.filters.revised_class_only if revisedclasses_only else None
    # val_loader = dataset.get_data_loader('valid', BS, class_filter=class_filter, device=device if use_cuda else None)
    test_loader = dataset.get_data_loader('test', BS, class_filter=class_filter, device=device)

    if model_dirs is None:
        model_dirs = sorted(listdir(stats_dir))
        import os
        model_dirs = [m for m in model_dirs if not os.path.isfile(join(stats_dir, m))]
    elif isinstance(model_dirs, str):
        model_dirs = [model_dirs]


    net = Classifier(net_input_feats, net_hidden_units, nb_classes=nb_classes)
    # load_state_path=join(stats_dir, m, state_files_to_use[1])

    for m in model_dirs:
        ls_stats_dir = sorted(listdir(join(stats_dir, m)))
        state_files = [sf for sf in ls_stats_dir if sf.endswith('.state')]

        state_names = []
        for s in state_files:
            if s.endswith('.state'):
                dots = s.split('.')
                try:
                    state_names.append(str(int(dots[-2])))
                except:
                    state_names.append('.'.join(dots[-3:-1]))

        state_files_to_use = []
        for s in states:
            if s in state_names:
                state_files_to_use.append(state_files[state_names.index(s)])
            else:
                raise Warning("State {} can't be found for model_dir {}".format(s, join(stats_dir, m)))


        for state_file in state_files_to_use:
            state_file_path = join(stats_dir, m, state_file)
            print("Loading state file: {}".format(state_file_path))

            state_dict = torch.load(state_file_path)
            net.load_state_dict(state_dict, strict=False)

            net.eval()
            print(net)
            if device is not None:
                net.to(device)
            net.labelflip_disable()

            ts_y_trues, ts_y_preds, ts_softmax, ts_acc, ts_loss = run_epoch(net, data_loader=test_loader, epoch_index=-1)
            test_acc_w_entropy_filtering = entropy_filtering(ts_y_trues, ts_softmax)

            return test_acc_w_entropy_filtering



def run_epoch(net: Classifier,
              data_loader,
              optimizer: torch.optim.Optimizer=None,
              entropy_scaling: bool=False,
              loss_log: int=128,
              epoch_index: int=-1,
              device=None):

    preds = []
    y_trues = []
    losses = []

    if optimizer:
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            x, y, mul = data
            x.to(device)
            y.to(device)
            optimizer.zero_grad()            # zero the parameter gradients
            net.zero_grad()
            pred = net(x)
            if entropy_scaling is False:
                mul=None
            loss = net.loss(y_true=y, y_pred=pred, per_img_loss_mul=mul)
            loss.backward()
            net.labelflip.pre_opt()  # blank gradient if labelflip layer is frozen
            optimizer.step()
            net.labelflip.post_opt()  # Normalize labelflip matrix (stochastic matrix
            losses.append(loss.item())

            running_loss += loss.item()
            preds.append(pred.detach().cpu().numpy())
            y_trues.append(y.detach().cpu().numpy())

            # print statistics
            if i % loss_log == loss_log - 1:  # print every loss_log_intervals mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch_index, i + 1, running_loss / loss_log))
                running_loss = 0.0

    else:
        for i, data in enumerate(data_loader, 0):
            x, y = data
            pred = net(x)
            loss = net.loss(y_true=y, y_pred=pred)
            losses.append(loss.item())
            preds.append(pred.detach().cpu().numpy())
            y_trues.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds)
    y_preds = np.argmax(preds, axis=1)
    y_trues = np.concatenate(y_trues)
    errors = (y_preds - y_trues)
    accuracy = np.sum(errors == 0) / float(len(y_preds))
    return y_trues, y_preds, preds, accuracy, np.mean(losses)





from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


class Stat:
    def __init__(self, accuracy=None, loss=None,
                 mAP_dict:dict=None,
                 mAP_macro=None, mAP_micro=None, mAP_weighted=None, mAP_sample=None, mAP_classwise=None):
        self.accuracy = -1 if accuracy is None else accuracy
        self.loss = np.inf if loss is None else loss

        self.mAP = {'macro': -1, 'micro': -1, 'weighted': -1, 'sample': -1, 'classwise': None}
        if mAP_dict is not None:
            for k, v in mAP_dict.items():
                if k in self.mAP.keys():
                    self.mAP[k] = v
                else:
                    raise Warning("Key '{}' in mAP_dict is not recognised and will be ignored.".format(k))

        if mAP_macro is not None:
            self.mAP['macro'] = mAP_macro
        if mAP_micro is not None:
            self.mAP['micro'] = mAP_micro
        if mAP_weighted is not None:
            self.mAP['weighted'] = mAP_weighted
        if mAP_sample is not None:
            self.mAP['sample'] = mAP_sample
        if mAP_classwise is not None:
            self.mAP['classwise'] = mAP_classwise


class ModelStatistics:
    def __init__(self):
        self.stats_for_states = {}

    def add_stats(self, state_name, epoch, test_stat: Stat, valid_stat: Stat):
        self.stats_for_states[state_name] = {'epoch': epoch, 'test': test_stat, 'valid': valid_stat}


    @classmethod
    def write_stat_file_dict(cls, path, model_stats_dict:dict, dset='test'):
        stat_file = open(path, 'w')

        stat_file.write("Model State Name\tModel Name\tDataSet\tEpoch\t\t")
        stat_file.write("Loss\tAcc\tmAP-macro\tmAP-micro\tmAP-weighted\tmAP-sample\n")

        state_names = list(model_stats_dict.items())[0][1].stats_for_states.keys()

        for st_name in state_names:
            for model_name, model_stats in model_stats_dict.items():
                ts = model_stats_dict[model_name].stats_for_states[st_name][dset]
                stat_file.write('{}\t{}\t{}\t-\t\t'.format(st_name, model_name, dset))
                stat_file.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(ts.loss, ts.accuracy, ts.mAP['macro'], ts.mAP['micro'],
                                                      ts.mAP['weighted'], ts.mAP['sample']))
            stat_file.write('\n')
        stat_file.close()

    def write_stat_file(self, path):
        stat_file = open(path, 'w')
        stat_file.write("Model State Name\tEpoch\t\t")
        stat_file.write("Loss\tAcc\tmAP-macro\tmAP-micro\tmAP-weighted\tmAP-sample\n")


        for state_name, stats_for_state in self.stats_for_states.items():
            stat_file.write(state_name + ' - test\t-\t\t')
            ts = stats_for_state['test']
            stat_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(ts.loss, ts.accuracy, ts.mAP['macro'], ts.mAP['micro'], ts.mAP['weighted'], ts.mAP['sample']))

            stat_file.write(state_name + ' - valid\t-\t\t')
            ts = stats_for_state['valid']
            stat_file.write("{}\t{}\t{}\t{}\t{}\t{}\n"
                            .format(ts.loss, ts.accuracy, ts.mAP['macro'], ts.mAP['micro'], ts.mAP['weighted'],ts.mAP['sample']))

            stat_file.write('\n')

        stat_file.close()



def compute_mAP(y_trues, y_preds, average="macro", verbose=1):
    """
    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.
    :param y_trues:
    :param y_preds:
    :param average:
    :return:
    """
    y_trues_binary = label_binarize(y_trues, classes=range(1, 200))
    y_preds_binary = label_binarize(y_preds, classes=range(1, 200))
    if isinstance(average, list) or isinstance(average, dict) or isinstance(average, set) or isinstance(average, tuple):
        mAPs = {}
        for avg in average:
            if verbose:
                print("Computing mAP-{}".format(avg if avg is not None else "classwise"))
            if avg is None:
                 mAPs['classwise'] = average_precision_score(y_trues_binary, y_preds_binary, average=avg)
            elif avg is 'classwise':
                mAPs[avg] = average_precision_score(y_trues_binary, y_preds_binary, average=None)
            else:
                mAPs[avg] = average_precision_score(y_trues_binary, y_preds_binary, average=avg)
        return mAPs

    else:
        return average_precision_score(y_trues_binary, y_preds, average=average)

def run_test(stats_dir: str,  # dir from which load states

             net_input_feats: int,
             net_hidden_units: list = (4096,),
             BS: int = 32,
             feats_name='resnet50_r255_c224_meansub_stdnorm',
             trainval_feats='trainval_200',
             test_feats='test_200',
             states: list = None,  # [1500, 1200, 1000, 'bestval.loss', 'bestval.acc', 'besttest.acc' , ..]
             revisedclasses_only=True,
             device=None
             ):
    from os import listdir
    from os.path import join

    nb_classes = 200 if revisedclasses_only else 3120




    dataset = NoisyArtFeats(trainval_feats_name=feats_name + '_' + trainval_feats,
                            test_feats_name=feats_name+'_'+test_feats,
                            bootstrap=False)
    class_filter = dataset.filters.revised_class_only if revisedclasses_only else None
    val_loader = dataset.get_data_loader('valid', BS, class_filter=class_filter, device=device)
    test_loader = dataset.get_data_loader('test', BS, class_filter=class_filter, device=device)



    model_dirs = sorted(listdir(stats_dir))
    import os
    model_dirs = [m for m in model_dirs if not os.path.isfile(join(stats_dir, m))]

    net = Classifier(net_input_feats, net_hidden_units, nb_classes=nb_classes)
    # load_state_path=join(stats_dir, m, state_files_to_use[1])

    all_model_stats = {}
    for m in model_dirs:
        ls_stats_dir = sorted(listdir(join(stats_dir, m)))
        state_files=[sf for sf in ls_stats_dir if sf.endswith('.state')]

        state_names = []
        for s in state_files:
            if s.endswith('.state'):
                dots = s.split('.')
                try:
                    state_names.append(str(int(dots[-2])))
                except:
                    state_names.append('.'.join(dots[-3:-1]))

        state_files_to_use = []
        for s in states:
            if s in state_names:
                state_files_to_use.append(state_files[state_names.index(s)])
            else:
                raise Warning("State {} can't be found for model_dir {}".format(s, join(stats_dir, m)))

        model_stats = ModelStatistics()



        for state_file in state_files_to_use:

            dots = state_file.split('.')
            try:
                state_name = str(int(dots[-2]))
            except:
                state_name = '.'.join(dots[-3:-1])

            state_file_path = join(stats_dir, m, state_file)
            print("Loading state file: {}".format(state_file_path))


            #net = Classifier(2048, net_hidden_units, nb_classes=nb_classes, load_state_path=state_file_path)
            state_dict = torch.load(state_file_path)
            net.load_state_dict(state_dict, strict=False)

            net.eval()
            print(net)
            if device is not None:
                net.to(device)
            net.labelflip_disable()


            vl_y_trues, vl_y_preds, vl_preds, vl_acc, vl_loss = run_epoch(net, data_loader=val_loader, epoch_index=-1)
            mAPs = compute_mAP(vl_y_trues, vl_y_preds, ("macro", "weighted"))
            vl_stats = Stat(vl_acc, vl_loss, mAPs)

            ts_y_trues, ts_y_preds, ts_preds, ts_acc, ts_loss = run_epoch(net, data_loader=test_loader, epoch_index=-1)
            mAPs = compute_mAP(ts_y_trues, ts_y_preds, ("macro", "weighted"))
            ts_stats = Stat(ts_acc, ts_loss, mAPs)

            net.train()

            model_stats.add_stats(state_name, -1, ts_stats, vl_stats)

        all_model_stats[m] = model_stats
        model_stats.write_stat_file(join(stats_dir, '../' + m + '.statsfile.csv'))

    ModelStatistics.write_stat_file_dict(join(stats_dir,'../../all_stats_{}_test.csv'.format(feats_name.split('_')[0])),
                                         all_model_stats, dset='test')
    ModelStatistics.write_stat_file_dict(join(stats_dir,'../../all_stats_{}_valid.csv'.format(feats_name.split('_')[0])),
                                              all_model_stats, dset='valid')







