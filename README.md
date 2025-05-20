## About

This projet aims at exploring convolutional neural networks to classify MRI brain scans.

## Todos

- [ ] Try deeper networks > unable to reach more than 5 conv layers because the output size becomes too small > keep learning deepl learning to explore architecture possibilities
- [ ] (Add preprocessing: CLHE, canny edge detection, normalization, etc) &rarr; probably unecessary since most of them are convolutional approaches anyway &rarr; check if it improves performance and/or learning though
- [ ] Improve evaluation metrics (add confusion matrix, ROC curve, etc.) &rarr; use torchmetrics / lightningai
- [ ] Add dropouts
- [ ] Add early stopping
- [ ] Add model checkpointing
- [ ] Add learning rate scheduler
- [ ] Add hyperparameter tuning
- [ ] Add data augmentation
- [ ] Test transfer learning (commonly done; models are rarely trained from scratch)
- [ ] Add tensorboard logging

## Notes

data & output folders are omitted to save space

## Installation

install cuda !
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

## Models

CNNModel: 4 conv layers, 1 fc layer
CNNModelV2: 5 conv layers, 1 fc layer
CNNModel7: 7 conv layers, 1 fc layer
CNN_MRI: ...

## Training

Execute the following command in the terminal:
`python train.py --epochs 10 --tag v4`

Note: Around 50 epochs looks like a sweet spot for V1.
! Warning: Training takes very long (on CPU)!

## Learning

Mistakes I've made:
- No pooling layers kept the feature map size very large & an enormous number of parameters in the fully connected layers (64*128*128)
- Final fully connected layer was to big (64*128*128) which slowed down training & stalled learning &rarr; reducing it to 64*16*16 by using pooling layers improved it
- Learning rate was too high or too low ()
- Pytorch CUDA was not installed &rarr; it must be installed via a special install using pip (not conda) &rarr; training was using CPU instead of (Nvidia) GPU
- Forgot to add squeeze() to the final output of the model &rarr; it was returning a 2D tensor instead of a 1D tensor &rarr; this caused an error when calculating the loss

## Log

**note: 4 or 5 conv layers seem to improve learning curve compared to 3**

--tag 5cl+pool
Training loss: 0.000, training acc: 100.000
Validation loss: 0.458, validation acc: 89.848
--------------------------------------------------
Training complete.
Training time: 473.25 seconds

--tag 4cl+pool
Training loss: 0.000, training acc: 100.000
Validation loss: 0.502, validation acc: 89.594
--------------------------------------------------
Training complete.
Training time: 459.26 seconds

**note: best lr = e-3 or e-4**

--tag 3cl+pool e:30: -lr:e-2 
Training loss: 13.760, training acc: 86.237
Validation loss: 25.848, validation acc: 73.350
--------------------------------------------------
Training complete.
Training time: 470.63 seconds

--tag 3cl+pool e:30 lr:1e-4
Training loss: 0.009, training acc: 100.000
Validation loss: 0.340, validation acc: 89.848
--------------------------------------------------
Training complete.
Training time: 465.11 seconds

--tag 3cl+pool-lre5
Training loss: 0.135, training acc: 94.425
Validation loss: 0.552, validation acc: 70.558
--------------------------------------------------
Training complete.
Training time: 541.22 seconds

--tag 3cl+pool e:30 lr:1e-3
Training loss: 0.000, training acc: 100.000
Validation loss: 0.497, validation acc: 89.340
--------------------------------------------------
Training complete.
Training time: 502.65 seconds

## Resources

- https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/
