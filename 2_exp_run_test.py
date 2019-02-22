from exp_tools.test_classifier import run_test

GPU = 0
bs=64

from _cfg import *

#%%
for feats_name in FEATS_NAMES:
    print("Starting {}".format(feats_name))
    logdir = LOGDIR + '/' + feats_name.split('_')[0]

    device = init(gpu_index=GPU, seed=SEED)

    input_size=2048
    if feats_name.startswith('vgg'):
        input_size=4096

    from os.path import join

    run_test(join(logdir, 'states'), input_size, HIDDEN_UNITS, BS=bs,

             feats_name=feats_name,
             trainval_feats=TRAINVAL_FEATS,
             test_feats=TEST_FEATS,

             states=['1500', 'bestval.loss', 'bestval.acc', 'besttest.loss', 'besttest.acc'],
             device=device)