# LeNet-5 for Digit Classification(mnist)
Assignment 3 for CS698: Topics in Computer Vision
Using Only Numpy and Scipy

# Data Setup
```
cd data/ && python make2D.py 
```

# Configuration
All the config(network, training, logging) parameters: config.py

# Training
```
cd src/ && python train.py
```

# Source Files

## src/convnet.py
Contains the Structure of LeNet-5 along with forward pass, backward pass

## src/fwd.py
Forward Pass Functions

## src/back.py
Backward Pass Functions

## src/activations.py
Activation Functions

## src/train.py
Driver Training Script

### Supporting scripts in _scripts/_
### Models are saved in _models/_
### Logs are saved in _logs/_
