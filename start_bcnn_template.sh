#!/bin/sh

net='--net AlexFC7'
#net='--net AlexConv5'

metric='--metric None'
#metric='--metric Diagonal'
#metric='--metric Symmetric'
#metric='--metric Bilinear'

dim='--dim 1'

weight='--weight official'
#weight='--weight places365'

n_param=''
#n_param='--n_param Triplet-param-467382378'

m_param=''
#m_param='--m_param Triplet-param-467382378'

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

python3 models/bcnn.py ${net} ${metric} ${dim} ${weight} ${n_param} ${m_param} ${lr} ${decay} ${margin} ${epoch} ${batch} ${version} ${verbose} ${freeze} ${valid}
