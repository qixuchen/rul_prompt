#!/bin/bash

# Execute some commands or script

python train_fed.py --cfg exps/clip_pe_net/fed_non_iid.yaml > ./exp_results/non_iid.txt

python train_fed.py --cfg exps/clip_pe_net/fed_FD004.yaml  > ./exp_results/FD004.txt