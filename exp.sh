#!/bin/bash

# Execute some commands or script

# python train_fed.py --cfg exps/clip_pe_net/fed_non_iid.yaml > ./exp_results/penet_fed_non_iid.txt

# python train_fed.py --cfg exps/clip_bilstm/fed_non_iid.yaml > ./exp_results/bilstm_fed_non_iid.txt

# python train_fed.py --cfg exps/clip_pe_net/fed_non_iid_v2.yaml > ./exp_results/penet_fed_non_iid_v2.txt

python train_fed.py --cfg exps/clip_bilstm/fed_non_iid_v2.yaml > ./exp_results/bilstm_fed_non_iid_v2.txt

# python train_fed.py --cfg exps/clip_bilstm/fed_FD003.yaml > ./exp_results/bilstm_fed_FD003.txt

# python train_fed.py --cfg exps/clip_pe_net/fed_FD003.yaml > ./exp_results/penet_fed_FD003.txt

# python train_fed.py --cfg exps/clip_bilstm/fed_FD001.yaml > ./exp_results/bilstm_fed_FD001.txt

# python train_fed.py --cfg exps/clip_pe_net/fed_FD001.yaml > ./exp_results/penet_fed_FD001.txt