from exp_tools.test_classifier import run_entropy_filtering

GPU = 0
bs=64

labels = ['ResNet-50 BS', 'ResNet-101 BS', 'ResNet-151 BS', 'VGG16 BS', 'VGG19 BS']

from _cfg import *

#%%

accuracies = []
for feats_name, label in zip(FEATS_NAMES, labels):
    print("Starting {}".format(feats_name))
    logdir = LOGDIR + '/' + feats_name.split('_')[0]

    device = init(gpu_index=GPU, seed=SEED)


    input_size=2048
    if feats_name.startswith('vgg'):
        input_size=4096

        import numpy as np

    from os.path import join
    from matplotlib import  pyplot as plt
    model = "1-[lr=0.0001]-[WD=1e-07]-[ES@80_upd=40_t=class_m=20_b=0.8_v0=1]-[TS@80_T=4]-[BOOT=80]-1500"

    acc_w_entropy_filtering = run_entropy_filtering(join(logdir, 'states'), input_size, HIDDEN_UNITS, model_dirs=model,
                                                    BS=bs,
                                                    feats_name=feats_name,
                                                    trainval_feats=TRAINVAL_FEATS,
                                                    test_feats=TEST_FEATS,

                                                    device=device,
                                                    states=['bestval.acc'],
                                                    )
    accuracies.append(acc_w_entropy_filtering)


#%%

import numpy as np
for stuff , label in zip(accuracies, labels):
    plt.plot(np.arange(0, 201) / 2, stuff, label=label)
    plt.xlabel('% removed classes')
    plt.ylabel('Accuracy on test set')
    plt.legend()

plt.savefig('entropy_filtering.pdf')
plt.show()


