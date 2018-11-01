#!/bin/sh

net='--net Metric'
param=''
# param='--param Triplet-param-467382378'
lr='--lr 0.001'
decay='--decay 0'
margin='--margin 5'
epoch='--epoch 10'
batch='--batch 256'
verbose='--verbose 1'
valid='--valid'
#valid='--no-valid'

python3 network/ml.py $net $param $lr $decay $margin $epoch $batch $version $verbose $valid