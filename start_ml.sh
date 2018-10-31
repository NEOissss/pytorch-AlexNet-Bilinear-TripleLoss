#!/bin/sh

net='--net Metric'
param=''
# param='--param Triplet-param-467382378'
lr='--lr 0.001'
margin='--margin 5'
epoch='--epoch 10'
batch='--batch 256'
version='--version 0'
verbose='--verbose 1'
valid='--valid'
#valid='--no-valid'

python3 pytroch/ml.py $net $param $lr $margin $epoch $batch $version $verbose $valid