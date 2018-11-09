#!/bin/sh

net='--net Triplet'
#net='--net TripletConv5'
#net='--net Bilinear'
#net='--net BilinearConv5'

dim='--dim 1'

weight='--weight official'
#weight='--weight places365'

param=''
# param='--param Triplet-param-467382378'

version='--version 0'
lr='--lr 0.0001'
decay='--decay 0.01'
margin='--margin 5'
epoch='--epoch 10'
batch='--batch 256'
verbose='--verbose 1'

freeze='--no-freeze'
#freeze='--freeze'

valid='--valid'
#valid='--no-valid'

python3 models/cnn.py ${net} ${dim} ${weight} ${param} ${version} ${lr} ${decay} ${margin} ${epoch} ${batch} ${version} ${verbose} ${freeze} ${valid}
