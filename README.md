# 'Almost' PyTorch in NumPy for Computer Vision
### (a from scratch implementation of PyTorch's modules for Computer Vision using NumPy)

## Implemented Layers
- [x] Conv2d
- [x] Linear
- [x] BatchNorm2d
- [x] BatchNorm1d
- [x] Dropout
- [x] MaxPool2d
#### todo:
- [ ] ConvTranspose2d

## Implemented Activation Functions
- [x] ReLU
- [x] Sigmoid
- [x] Softmax
- [x] Flatten
#### todo:
- [ ] Tanh
- [ ] LeakyReLU

## Implemented Loss Functions
- [x] CrossEntropyLoss
- [x] MSELoss
#### todo:
- [ ] L1Loss
- [ ] BCELoss

## Implemented Optimizers
- [x] SGD
- [x] Adam

## Implemented Utils
- [x] train_one_epoch
- [x] train (train_one_epoch + validation)
- [x] Model wrapper (combination of Sequential and Module) 
#### todo:
- [ ] DataLoader
- [ ] Dataset
- [ ] Model evaluation mode
- [ ] Training loop: validation loss 