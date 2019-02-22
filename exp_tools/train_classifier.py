# SETTINGS:
from exp_tools.utils import NP
import numpy as np
import torch, torch.cuda, torch.backends, torch.optim, torch.utils, torch.utils.data


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

    entropy_per_class_mean = {cls: np.mean(entropy_per_class[cls]) for cls in classes}
    entropy_per_class_std = {cls: np.std(entropy_per_class[cls]) for cls in classes}

    return entropy, entropy_per_class, entropy_per_class_mean, entropy_per_class_std, y_trues

def compute_entropy_loader(net, data_loader, T=None):
    y_preds, y_trues = predict(net, data_loader, T)
    return compute_entropy(y_preds, y_trues)


def compute_entropy_dset(net, dset, BS=32, T=None, device=None):
    dataset = NoisyArtFeats()
    data_loader = dataset.get_data_loader(dset, BS, shuffle=False, device=device)
    return compute_entropy_loader(net, data_loader, T)

def compute_perclass_accuracy(preds, y_trues, ret_class_support=False):
    acc = np.where(np.argmax(preds, axis=1)==y_trues, 1, 0)
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




def run_epoch(net: Classifier,
              data_loader: torch.utils.data.DataLoader,
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





from tensorboardX import SummaryWriter
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

import datetime
import time

from typing import Callable, Tuple, Any


def get_logistic_entropy_score_fn(m=20., b=0.8, v0=1., force_v1_eq_0=False):
    """
    https://www.desmos.com/calculator/3exuemp2sj
    https://www.desmos.com/calculator/34jplmbri3s
    :param m: Slope of the function in the linear zone
    :param b: point b for which holds f(b)=0.5
    :param v0: maximum value, i.e. f(0)=v
    :param force_v1_eq_0: if True, normalization will force f(1)  to always be 0
    :return: the function fn
    """
    if force_v1_eq_0:
        # f =  lambda x: v0/(1+np.exp(m*x - m*b))
        # return lambda x: (f(1)+1)*f(x) - f(1)
        return lambda x: (v0/(1+np.exp(m*1 - m*b)) + 1)  * ( v0/(1+np.exp(m*x - m*b)) ) -  (v0/(1+np.exp(m*1 - m*b)))
                       # (        f(1)             + 1)  *            f(x)              -           f(1)        
    else:
        return lambda x: v0/(1+np.exp(m*x - m*b))



class Statistics:
    def __init__(s):
        LOSS_INF = 99999999
        s.best_test_by_acc = {'test': {'acc': 0.0, 'loss': LOSS_INF}, 'valid': {'acc': 0.0, 'loss': LOSS_INF}, 'epoch': -1}
        s.best_test_by_loss = {'test': {'acc': 0.0, 'loss': LOSS_INF}, 'valid': {'acc': 0.0, 'loss': LOSS_INF}, 'epoch': -1}
        s.best_valid_by_acc = {'test': {'acc': 0.0, 'loss': LOSS_INF}, 'valid': {'acc': 0.0, 'loss': LOSS_INF}, 'epoch': -1}
        s.best_valid_by_loss = {'test': {'acc': 0.0, 'loss': LOSS_INF}, 'valid': {'acc': 0.0, 'loss': LOSS_INF}, 'epoch': -1}

    def update_check_test_acc(self, epoch, valid_loss, test_loss, valid_acc, test_acc):
        if test_acc > self.best_test_by_acc['test']['acc']:
            self.update(self.best_test_by_acc, epoch, valid_loss, test_loss, valid_acc, test_acc)
            return True
        else:
            return False

    def update_check_test_loss(self, epoch, valid_loss, test_loss, valid_acc, test_acc):
        if test_loss < self.best_test_by_loss['test']['loss']:
            self.update(self.best_test_by_loss, epoch, valid_loss, test_loss, valid_acc, test_acc)
            return True
        else:
            return False

    def update_check_valid_acc(self, epoch, valid_loss, test_loss, valid_acc, test_acc):
        if valid_acc > self.best_valid_by_acc['valid']['acc']:
            self.update(self.best_valid_by_acc, epoch, valid_loss, test_loss, valid_acc, test_acc)
            return True
        else:
            return False

    def update_check_valid_loss(self, epoch, valid_loss, test_loss, valid_acc, test_acc):
        if valid_loss < self.best_valid_by_loss['valid']['loss']:
            self.update(self.best_valid_by_loss, epoch, valid_loss, test_loss, valid_acc, test_acc)
            return True
        else:
            return False

    @staticmethod
    def update(stat_dict, epoch, valid_loss, test_loss, valid_acc, test_acc):
        stat_dict['epoch'] = epoch
        stat_dict['valid']['loss'] = valid_loss
        stat_dict['valid']['acc'] = valid_acc
        stat_dict['test']['loss'] = test_loss
        stat_dict['test']['acc'] = test_acc

    def write_stat_file(self, path):
        stat_file = open(path, 'w')
        D = {   'best-test-acc': self.best_test_by_acc,
                'best-test-loss': self.best_test_by_loss,
                'best-valid-acc': self.best_valid_by_acc,
                'best-valid-loss': self.best_valid_by_loss     }
        stat_file.write('Criterion\tEpoch\tTest-Acc\tTest-Loss\tValid-Acc\tValid-Loss\n')
        for k, v in D.items():
            stat_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(k, v['epoch'], v['test']['acc'],  v['test']['loss'],
                                                                             v['valid']['acc'], v['valid']['loss']))
        stat_file.close()



def run_train(net: Classifier,
              opt_fn: Callable[[Any], torch.optim.Optimizer],
              BS: int,
              nb_epochs: int,

              boot_epochs: int=-1,

              feats_name='resnet50_r255_c224_meansub_stdnorm',
              trainval_feats='trainval_200',
              test_feats='test_200',

              labelflip_epoch: int=-1,
              weight_decay: float=0.,
              lf_alpha: float=0.1,
              lf_reg: str='trace_l1',
              lf_opt_fn: Callable[[], torch.optim.Optimizer]=None,
              entropy_scaling_epoch: int=-1,
              lf_disable_entropy: bool=True,
              entropy_scaling_type: str='class',
              es_update_interval=1,
              es_w_lf=False,
              es_v0=1, es_m=20, es_b=.8, es_v1_0=False,

              checkpoint_interval: int=None,
              earlystopping_epochs: int=-1,
              virtual_early_stopping: bool=False,
              revisedclasses_only: bool=False,
              temp_scaling: float=1.,
              temp_scaling_during_opt: bool=False,
              temp_scaling_epoch: int=-1,
              epsilon: float=1e-20,
              append_name: str=None,
              prepend_name: str=None,
              custom_name: str=None,
              disable_timestamp: bool=False,
              log_dir: str='runs',
              first_epoch: int=1,
              opt_replacement: Tuple[Callable[[], torch.optim.Optimizer], int]=None,
              save_last_state_every_epoch=False,
              skip_save_model_ep=0,
              test_map_epoch_interval=-1,

              device=None
              ):
    from shutil import copyfile
    from torch import nn

    if first_epoch < 0:
        raise ValueError("First epoch should be >= 0.")
    if labelflip_epoch < 0:
        print("Labelflip Disabled")
    # if entropy_scaling_epoch < 0:
    #     print("")

    bootstrap = True if boot_epochs > 0 else False
    dataset = NoisyArtFeats(trainval_feats_name=feats_name + '_' + trainval_feats,
                            test_feats_name=feats_name+'_'+test_feats,
                            bootstrap=bootstrap)

    lf_alpha = None if labelflip_epoch <= 0 else lf_alpha
    old_alpha = net.labelflip.alpha
    if lf_alpha is not None:
        net.labelflip.alpha=lf_alpha

    print(net)
    if device is not None:
        net.to(device)
    criterion = nn.CrossEntropyLoss()

    def get_params(network):
        if weight_decay <= 0:
            prms = list(network.parameters())
        else:
            prms = network.weight_decay_parameters(l2_value=weight_decay, skip_list=('labelflip.weight'))
        return prms
    params = get_params(net)
    opt = opt_fn(params)
    if lf_opt_fn is None:
        lf_opt = opt_fn(params)
    else:
        lf_opt = lf_opt_fn(params)



    summary_str = ""
    if prepend_name is not None:
        summary_str += prepend_name

    if custom_name:
        summary_str += custom_name

    else:
        if revisedclasses_only:
            summary_str += 'REV-ONLY '

        summary_str += "net={} opt={} lr={} bs={}".format(net.get_description(), type(opt).__name__, opt.defaults['lr'], BS)
        if labelflip_epoch > 0:
            summary_str += " lf@{} lf_alpha={}".format(str(labelflip_epoch), lf_alpha)
            if lf_opt_fn is not None:
                summary_str += " lf_opt={} lf_lr={}".format(type(lf_opt).__name__, lf_opt.defaults['lr'])

        if temp_scaling_epoch is not None and entropy_scaling_epoch > 0:
            summary_str += " TS@{}".format(temp_scaling_epoch)
            summary_str += " T={}".format(temp_scaling)

        if entropy_scaling_epoch is not None and entropy_scaling_epoch > 0:
            summary_str += " ES@{}".format(entropy_scaling_epoch)
            #summary_str += " ES_l={}".format(es_l)


        if append_name is not None:
            summary_str += " " + str(append_name)

    class_filter = dataset.filters.revised_class_only if revisedclasses_only else None
    train_loader = dataset.get_data_loader('train', BS, class_filter=class_filter, device=device)
    boot_loader = dataset.get_data_loader('boot', BS, class_filter=class_filter, device=device)
    val_loader = dataset.get_data_loader('valid', BS, class_filter=class_filter, device=device)
    test_loader = dataset.get_data_loader('test', BS, class_filter=class_filter, device=device)


    if not disable_timestamp:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H:%M:%S')
        summary_str = timestamp + " " + summary_str

    from os.path import join
    writer = SummaryWriter(log_dir=join(log_dir, summary_str))

    import os
    states_dir = join(log_dir, "states", summary_str)
    os.makedirs(states_dir, exist_ok=True)

    nb_virtual_early_stop = 0
    use_entropy = False
    entropy_last_update=0


    T = 1
    dT = None

    if opt_replacement is not None:
        opt_replacement_epoch = opt_replacement[1]
    else:
        opt_replacement_epoch = -1

    epoch = first_epoch  # default = 1
    train_dset = 'train'
    if bootstrap:
        last_boot_epoch = epoch + boot_epochs
        if nb_epochs < boot_epochs:
            raise ValueError("nb_epochs should be greater than boot_epochs!")
        train_dset = 'boot'
        actual_train_loader = boot_loader


    else:
        last_boot_epoch = epoch
        actual_train_loader = train_loader


    stats = Statistics()
    es_score_fn = get_logistic_entropy_score_fn(es_m, es_b, es_v0, es_v1_0)

    nb_train_examples = len(actual_train_loader.dataset)
    per_img_loss_multiplier = np.ones(nb_train_examples).astype(np.float32)

    last_epoch = first_epoch + nb_epochs - 1
    while epoch <= last_epoch:
    #for epoch in range(nb_epochs+labelflip_epochs):
        print("")
        print("Epoch {}".format(epoch))

        if epoch == last_boot_epoch:
            train_dset = 'train'
            actual_train_loader = train_loader
            nb_train_examples = len(actual_train_loader.dataset)
            per_img_loss_multiplier = np.ones(nb_train_examples).astype(np.float32)
            print("\nSTOP BOOT - START NORMAL TRAINING")


        if epoch == labelflip_epoch:
            print("\nLABELFLIP UNLOCK!!\n")
            net.labelflip_unlock()  # unlock parameters of labelflip: now are trainable
            opt = lf_opt
            nb_virtual_early_stop = 0
            net.labelflip_enable()  # enable labelflip during training: it should absorb noise
            if lf_disable_entropy:
                use_entropy = False
        elif labelflip_epoch > 0 and epoch >= labelflip_epoch:
            net.labelflip_enable()  # enable labelflip during training: it should absorb noise

        if epoch == opt_replacement_epoch:
            print("\nOPT REPLACEMENT!!\n")
            params = get_params(net)
            opt = opt_replacement[0](params)

        if dT is not None:
            T+= dT

        if temp_scaling_during_opt:
            net.temperature = T


        train_loader_shuffeled = dataset.get_data_loader(train_dset, BS, class_filter=class_filter, device=device,
                                                         shuffle=True, additional_np_data=[per_img_loss_multiplier])

        y_trues, y_preds, preds, acc, loss = run_epoch(net, data_loader=train_loader_shuffeled, optimizer=opt,
                                                       entropy_scaling=use_entropy, loss_log=300, epoch_index=epoch,
                                                       device=device)

        if net.labelflip_enabled:
            _lf = net.labelflip.weight.detach().cpu().numpy()
            #png = mat2png(_lf, 'labelflip-matrix', 'Figure/labelflip')
            #png = mat2png_v2(_lf)
            writer.add_image('Figure/labelflip', np.expand_dims(_lf, 0) , global_step=epoch)

        #entropy_summaries = write_entropy_summary(net, train_loader, writer, epoch)

        print("Training Accuracy: {}".format(acc))
        writer.add_scalar('loss/train', loss, epoch)
        writer.add_scalar('acc/train', acc, epoch)


        if labelflip_epoch > 0  and epoch >= labelflip_epoch:
            net.labelflip_disable()  # disable labelflip during prediction



        if epoch == entropy_scaling_epoch:
            use_entropy = True
            entropy_last_update = es_update_interval 

        if epoch == temp_scaling_epoch:

            if temp_scaling is not None:
                if isinstance(temp_scaling, float) or isinstance(temp_scaling, int):
                    T = temp_scaling
                else:
                    T = temp_scaling[0]
                    remaining_epochs = last_epoch - epoch
                    dT = (temp_scaling[1] - temp_scaling[0]) / remaining_epochs


        entropy_last_update +=1
        if use_entropy and entropy_last_update>=es_update_interval:
            print("Computing Entropy...")
            entropy_last_update = 0
            h, h_cls, h_cls_mean, h_cls_std, labels = compute_entropy_loader(net, data_loader=actual_train_loader, T=T)
            scaled_entropy = np.ones(len(h))
            if entropy_scaling_type is 'class':
                h_cls_max = {cls: np.max(h_cls[cls]) for cls in h_cls.keys()}
                h_cls_min = {cls: np.min(h_cls[cls]) for cls in h_cls.keys()}
                for i, (e, l) in enumerate(zip(h, labels)):
                    scaled_entropy[i] = (e - h_cls_min[l]) / (h_cls_max[l]-h_cls_min[l]+epsilon)  # scaling in 0 - 1 linearly

            elif entropy_scaling_type is 'global':
                h_max = np.max(h)
                h_min = np.min(h)
                for i, e in enumerate(h):
                    scaled_entropy[i] = (e-h_min) / (h_max-h_min + epsilon)

            scaled_entropy = scaled_entropy.astype(np.float32)
            print("scaled_entropy = {}  +/-{} ".format(np.mean(scaled_entropy), np.std(scaled_entropy)))
            
            
            per_img_loss_multiplier = es_score_fn(scaled_entropy)
            print("per_img_loss_multiplier = {}  +/-{} ".format(np.mean(per_img_loss_multiplier), np.std(per_img_loss_multiplier)))
            
            if es_w_lf:
                labelflip_class_prob = torch.ones(net.labelflip.weight.shape[0]) - net.labelflip.weight.diag()
                lf_prob = labelflip_class_prob[labels] # or labelflip_class_prob[labels]
                per_img_loss_multiplier = 1 - NP(lf_prob) * (1 - per_img_loss_multiplier)




        y_trues, y_preds, preds, val_acc, val_loss = run_epoch(net, data_loader=val_loader, epoch_index=epoch, device=device)
        print("Validation accuracy: {}".format(val_acc))
        writer.add_scalar('loss/valid', val_loss, epoch)
        writer.add_scalar('acc/valid', val_acc, epoch)
        #y_trues_binary = label_binarize(y_trues, classes=range(preds.shape[-1]))


        y_trues, y_preds, preds, test_acc, test_loss = run_epoch(net, data_loader=test_loader, epoch_index=epoch, device=device)
        print("Test accuracy: {}".format(test_acc))
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', test_acc, epoch)

        if test_map_epoch_interval>0 and epoch%test_map_epoch_interval ==0:
            y_trues_binary = label_binarize(y_trues, classes=range(preds.shape[-1]))
            map_samples = average_precision_score(y_trues_binary, preds, average='samples')
            writer.add_scalar('mAP-samples/test', map_samples, epoch)





        if save_last_state_every_epoch:
            if epoch >= skip_save_model_ep:
                torch.save(net.state_dict(), join(states_dir, summary_str + ".last.state"))

        if checkpoint_interval is not None and epoch%checkpoint_interval==0:
            torch.save(net.state_dict(),  join(states_dir, summary_str + ".{}.state".format(epoch)))




        if stats.update_check_test_acc(epoch, val_loss, test_loss, val_acc, test_acc):
            if epoch >= skip_save_model_ep:
                torch.save(net.state_dict(), join(states_dir, summary_str + ".besttest.acc.state"))

        if stats.update_check_test_loss(epoch, val_loss, test_loss, val_acc, test_acc):
            if epoch >= skip_save_model_ep:
                torch.save(net.state_dict(), join(states_dir, summary_str + ".besttest.loss.state"))

        if stats.update_check_valid_acc(epoch, val_loss, test_loss, val_acc, test_acc):
            if epoch >= skip_save_model_ep:
                torch.save(net.state_dict(), join(states_dir, summary_str + ".bestval.acc.state"))

        if stats.update_check_valid_loss(epoch, val_loss, test_loss, val_acc, test_acc):
            if epoch >= skip_save_model_ep:
                torch.save(net.state_dict(), join(states_dir, summary_str + ".bestval.loss.state"))
        else:
           if epoch - stats.best_valid_by_loss['epoch'] == earlystopping_epochs:
                nb_virtual_early_stop+=1
                if epoch-1 < last_epoch:
                    name = ".valloss-earlystop-{}.ep-{}.state".format(nb_virtual_early_stop, epoch)
                else:
                    name = ".valloss-earlystop-LF-{}.ep-{}.state".format(nb_virtual_early_stop, epoch)
                copyfile( join(states_dir, summary_str + ".bestval.state"),
                          join(states_dir, summary_str + name))
                if not virtual_early_stopping:
                    last_epoch = epoch + 1
                    print("Early Stopping!")

        epoch += 1

    if lf_alpha is not None:
        net.labelflip.alpha=old_alpha
    
    torch.save(net.state_dict(),  join(states_dir, summary_str + ".last.{}.state".format(epoch)))

    writer.close()
    stats.write_stat_file(join(states_dir, summary_str + ".stats"))
    return locals()




def run_exp_entropy(hiddens, BS, dropout=None, revisedclasses_only=False, state_dict_path=None, T=None, dset='train', prepend_name='',
                    device=None):


    dataset = NoisyArtFeats()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H:%M:%S')
    #nb_classes = 200 if testclass_only else 3120
    nb_classes = 3120
    net = Classifier(2048, hiddens, nb_classes, dropout=dropout)
    if state_dict_path is not None:
        print('loading weights: {}'.format(state_dict_path))
        state_dict = torch.load(state_dict_path)
        net.load_state_dict(state_dict, strict=False)
    print(net)
    if device is not None:
        net.to(device)

    class_filter = dataset.filters.revised_class_only if revisedclasses_only else None
    data_loader = dataset.get_data_loader(dset, BS, class_filter=class_filter, device=device)

    net.labelflip_disable()
    h, h_cls, h_cls_mean, h_cls_std,  labels = compute_entropy_loader(net, data_loader=data_loader, T=T)

    d = {'entropy': h,
         'class_entropy': h_cls,
         'class_entropy_mean': h_cls_mean,
         'class_entropy_std': h_cls_std,
         'labels': labels}
    if prepend_name is not None:
        prepend_name = "_" + prepend_name
    fname = 'entropy_dicts/{}{}_entropy_dict'.format(timestamp,prepend_name)
    if T is not None:
        fname += 'T={}'.format(T)
    np.save(fname, d)


    return d, net, data_loader


#%%










