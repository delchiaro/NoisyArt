#!/usr/bin/env bash

NOISYART="/mnt/2tb/datasets/noisyart/"
FEATS="${NOISYART}features/"


GPU=0
W=0
BS=32


# Processing Training Set 200-classes
IMGS="${NOISYART}imgs/trainval_200_r255_png/"

python feat_extractor.py resnet50  -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a trainval_200
python feat_extractor.py resnet101 -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a trainval_200
python feat_extractor.py resnet152 -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a trainval_200
python feat_extractor.py vgg16     -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a trainval_200
python feat_extractor.py vgg19     -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a trainval_200


# Processing Test Set 200-classes
IMGS="${NOISYART}imgs/test_200/"
python feat_extractor.py resnet50  -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a test_200
python feat_extractor.py resnet101 -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a test_200
python feat_extractor.py resnet152 -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a test_200
python feat_extractor.py vgg16     -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a test_200
python feat_extractor.py vgg19     -b ${BS} --mean --std --gpu=${GPU} --workers=${W} -d ${IMGS} -o ${FEATS} -a test_200


