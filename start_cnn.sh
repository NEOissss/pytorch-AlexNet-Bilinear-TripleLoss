#!/bin/sh

net='--net Triplet'
param=''
# param='--param Triplet-param-467382378'
lr='--lr 0.001'
decay='--decay 0'
margin='--margin 5'
epoch='--epoch 10'
batch='--batch 256'
verbose='--verbose 1'
freeze='--freeze'
#freeze='--no-freeze'
valid='--valid'
#valid='--no-valid'

python3 pytorch/cnn.py $net $param $lr $decay $margin $epoch $batch $version $verbose $freeze $valid