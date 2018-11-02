#!/bin/sh

net='--net Triplet'
#net='--net Bilinear'

param=''
# param='--param Triplet-param-467382378'

version='--version 0'
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

python3 network/cnn.py ${net} ${param} ${version} ${lr} ${decay} ${margin} ${epoch} ${batch} ${version} ${verbose} ${freeze} ${valid}
