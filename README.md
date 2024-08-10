# One-step Spiking Transformer with a Linear Complexity, IJCAI 2024.

This is the code implementation for the paper "One-step Spiking Transformer with a Linear Complexity".

<p align="center">
<img src="https://github.com/songxt3/songxt3.github.io/blob/main/images/OST_framework.jpg">
</p>

### Requirements
timm==0.5.4

cupy==10.6.0

pytorch==1.13.1

spikingjelly==0.0.0.0.12

pyyaml==6.0

data prepare: your should download corresponding dataset and modify the dateset path 'data_dir:' at each dataset '.yml' file.


### Training on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Training on cifar10/cifar100
Setting hyper-parameters in cifar10.yml/cifar100.yml

```
cd cifar
python train.py
```

### Training on cifar10DVS
Setting hyper-parameters in train.py

```
cd dvs_cifar10
python train.py
```

### Training on DVS128Gesture
Setting hyper-parameters in train.py

```
cd dvs_128_gesture
python train.py
```

## Reference
```
@inproceedings{ijcai2024p348,
  title     = {One-step Spiking Transformer with a Linear Complexity},
  author    = {Song, Xiaotian and Song, Andy and Xiao, Rong and Sun, Yanan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {3142--3150},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/348},
  url       = {https://doi.org/10.24963/ijcai.2024/348},
}
```
