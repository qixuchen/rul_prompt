# Code for Machine Remaining Useful Life Prediction task 


## Prerequisites
This code is based on Pytorch. 

It has been tested on Pytorch v1.7.1 and CUDA v10.1.243 under Ubuntu 18.04. 

You can also try to run it on other versons. 

## Training
All your configurations are included in ./exps/*.yaml and the default value are defined in ./config.py. 

For Bi-LSTM network without CLIP, please run:
```
python train.py --cfg exps/basic.yaml
```
For Bi-LSTM with CLIP, please run:
```
python train_clip.py --cfg exps/clip.yaml
```
## Visualization
Please indicate the path for the .log file in ./plot.py, then run:
```
python plot.py
```
A loss curve will be saved at the root path.
