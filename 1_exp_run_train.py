from exp_tools.train_classifier import run_train
from exp_tools.classifier import Classifier
from torch import optim

GPU = 0
bs=32

opt_fn = lambda p: optim.Adam(p, lr=lr)

checkpoint_interval=50
skip_save_model_ep=150

first_ep = 1
epochs = 1500
lr = 1e-4
wd = 1e-7

# LF common:
lf_a = 1e-3

# ES common:
es_u = 40
es_type = 'class' #global
es_m = 20
es_b = .8
es_v0 = 1
es_v1_0 = True

# TS common:
t = 4
ts_o = False


# Import configuration variables
from _cfg import *

#%%
for feats_name in FEATS_NAMES:
    configurations = list(zip(LF, ES, TS, BOOT_EP))

    for lf, es, ts, boot_ep in configurations:
        last_ep = first_ep + epochs - 1

        logdir = LOGDIR + '/' + feats_name.split('_')[0]
        custom_name='{}-[lr={}]'.format(first_ep, lr)
        custom_name += '-[WD={}]'.format(wd) if wd>0 else ''
        custom_name += '-[LF@{}_a={}]'.format(lf, lf_a) if lf > 0 else ''
        custom_name += '-[ES@{}_upd={}_t={}_m={}_b={}_v0={}]'\
            .format(es, es_u, es_type, es_m, es_b, es_v0, '_v1:=0' if es_v1_0 else '') if es > 0 else ''
        custom_name += '-[TS@{}_T={}{}]'.format(ts, t, '_opt' if ts_o else '') if ts > 0 else ''
        custom_name += '-[BOOT={}]'.format(boot_ep) if boot_ep > 0 else ''
        custom_name += '-{}'.format(last_ep)

        device = init(gpu_index=GPU, seed=SEED)



        input_size = 2048
        if feats_name.startswith('vgg'):
            input_size = 4096

        net = Classifier(input_size, HIDDEN_UNITS, nb_classes=200)


        exp_locals = run_train(net, opt_fn, nb_epochs=epochs,

                               feats_name=feats_name,
                               trainval_feats=TRAINVAL_FEATS,
                               test_feats=TEST_FEATS,

                               BS=bs, revisedclasses_only=True,
                               disable_timestamp=True,

                               boot_epochs=boot_ep,

                               # LF:
                               labelflip_epoch=lf,
                               lf_alpha=lf_a,

                               # ES:
                               entropy_scaling_epoch=es,
                               es_update_interval=es_u,
                               entropy_scaling_type=es_type,

                               es_m=es_m, es_b=es_b, es_v0=es_v0, es_v1_0=True,

                               # TS:
                               temp_scaling_epoch=ts,
                               temp_scaling=t,
                               temp_scaling_during_opt=ts_o,

                               # WD:
                               weight_decay=wd,

                               custom_name=custom_name,
                               first_epoch=first_ep,
                               log_dir=logdir,

                               checkpoint_interval=checkpoint_interval,
                               # opt_replacement=(wd_opt_fn,501),

                               skip_save_model_ep=skip_save_model_ep,
                               device=device
                               )

