import torch
from torch import nn
import torch.nn.functional as F

EPSILON=1e-20


class Labelflip(nn.Linear):
    def __init__(self,  in_features, alpha=.1, regularizer='trace_l1'):
        super(Labelflip, self).__init__( in_features, in_features, False)
        assert regularizer in ['trace_l1', 'trace_l2', 'ridge_l1', 'ridge_l2']
        self.weight.data = torch.diag(torch.ones([in_features]), 0)
        self.alpha = alpha
        self.regularizer = regularizer
        self.trainable = False


    @property
    def trainable(self):
        return self.weight.requires_grad
    @trainable.setter
    def trainable(self, value: bool):
        self.weight.requires_grad = value


    def post_opt(self):
        """ Project each column of the labelflip matrix such that it is a probability vector """
        if self.trainable:
            self.weight.data = self.unit_norm_non_neg_contraint(self.weight.data, dim=0)
            #self.weight.data = self.stochastic_vector_contraint(self.weight.data, dim=0)
            # dim=0 --> sum of elments in each column forced to be 1

    @staticmethod
    def unit_norm_non_neg_contraint(w, dim=0, epsilon=EPSILON):
        w = w * (w >=.0).float()
        #return w / (epsilon + torch.sum(w, dim=dim, keepdim=True))
        return w / (torch.sum(w, dim=dim, keepdim=True))

    @staticmethod
    def stochastic_vector_contraint(w: torch.Tensor, dim=0, epsilon=EPSILON):
        min, _ = torch.min(w, dim=dim, keepdim=True)
        w = w - min * (min<.0).float()
        max, _ = torch.max(w, dim=dim, keepdim=True)
        w = w/max
        #return w / (epsilon + torch.sum(w, dim=axis, keepdim=True))
        return w / (torch.sum(w, dim=dim, keepdim=True))

    def loss_regularizer(self):
        weights = self.weight
        if self.regularizer.startswith('trace'):
            weights = torch.diag(self.weight, 0)
        if self.regularizer.endswith('l1'):
            loss = torch.sum(torch.abs(weights))
        if self.regularizer.endswith('l2'):
            loss = torch.sum(weights ** 2)
        return self.alpha * loss



class Classifier(nn.Module):

    def __init__(self, in_feats, hiddens=([4096],), nb_classes=3120, dropout=None, lf_alpha=0.1, lf_reg='trace_l1',
                 temperature=None, load_state_path=None):
        super(Classifier, self).__init__()
        self.out_size = nb_classes
        self.hidden_layers = []
        prev_size = in_feats
        if isinstance(hiddens, int):
            hiddens = [hiddens]
        self.dropout = dropout
        for l in hiddens:
            if isinstance(l, int):
                self.hidden_layers.append(nn.Linear(prev_size, l))
                prev_size = l

        self.out_layer = nn.Linear(prev_size, self.out_size)
        self.labelflip_enabled = False
        self.labelflip = Labelflip(self.out_size, lf_alpha, regularizer=lf_reg)

        if temperature is not None:
            self.temperature = torch.Tensor([temperature])
        else:
            self.temperature = torch.Tensor([1])

        if load_state_path is not None:
            print('loading weights: {}'.format(load_state_path))
            state_dict = torch.load(load_state_path)
            self.load_state_dict(state_dict, strict=False)

    def get_description(self):
        s = "["
        for h in self.hidden_layers:
            s += str(h.out_features) + '-'
        return s[:-1] + ']'



    def weight_decay_parameters(self, l2_value, skip_list=()):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


    def forward(self, x):
        # Max pooling over a (2, 2) window
        for l in self.hidden_layers:
            x = l(x)
            x = F.relu(x)
            if self.dropout is not None:
                x = nn.Dropout(self.dropout)(x)
        x = F.softmax(self.out_layer(x)/self.temperature)
        if self.labelflip_enabled:
            x = self.labelflip(x)
        return x

    def loss(self, y_true, y_pred=None, per_img_loss_mul=None):
        '''

        :param y_true:
        :param y_pred:
        :param per_img_loss_mul:
        '''
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(y_pred, y_true)
        if per_img_loss_mul is not None:
            loss *= per_img_loss_mul
        if self.labelflip.trainable:
            loss += self.labelflip.loss_regularizer()
        loss = torch.Tensor.mean(loss)
        return loss

    def labelflip_unlock(self):
        self.labelflip.trainable = True
        # optimizer is able to change the weights

    def labelflip_lock(self):
        self.labelflip.trainable = False
        # optimizer can't change the weights (frozen)

    def labelflip_disable(self):
        self.labelflip_enabled = False

    def labelflip_enable(self):
        self.labelflip_enabled = True




