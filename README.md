## About

This projet aims at exploring convolutional neural networks to classify MRI brain scans.

## Todos

- [x] Try deeper networks &rarr; done up to 5 conv layers &rarr; unable add more because the output size becomes too small > keep learning deepl learning to explore architecture possibilities
- [x] Test different learning rates &rarr; done, 1e-3 or 1-e4 seem to be the best
- Add batch normalization
- [x] (Re)Add pooling layers
- [x] Add dropouts layers
- [x] (Add data augmentation)
- [ ] Fix load & save model
- [ ] Test transfer learning (commonly done; models are rarely trained from scratch)
- [ ] (Add preprocessing: CLHE, canny edge detection, normalization, etc) &rarr; probably unecessary since most of them are convolutional approaches anyway &rarr; check if it improves performance and/or learning though
- [ ] (Add early stopping)
- [ ] (Add model checkpointing)
- [ ] (Add learning rate scheduler)
- [ ] (Add hyperparameter tuning)
- [ ] (Improve evaluation metrics (add confusion matrix, ROC curve, etc.) &rarr; use torchmetrics / lightningai)
- [ ] (Add tensorboard logging)

## Notes

data & output folders are omitted to save space

## Installation

! install cuda
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

## Models



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
- Added too much dropout: too many (one per convolutional layers so 3-5) and too high (0.5) &rarr; it caused the model to stop learning (too much destruction)

## Log

### RN50()

python train.py --epochs 30 --tag rn50
Training loss: 0.000, training acc: 100.000
Validation loss: 0.708, validation acc: 91.371
--------------------------------------------------
Training complete.
Training time: 2140.50 seconds


python train.py --epochs 5 --tag rn50
Training loss: 0.023, training acc: 99.094
Validation loss: 2.234, validation acc: 37.817
--------------------------------------------------
Training complete.
Training time: 305.19 seconds


### CNN()

python train.py --epochs 50 --tag 4cl+pools-1drop02-notrfs
Training loss: 0.000, training acc: 100.000
Validation loss: 0.754, validation acc: 91.371
--------------------------------------------------
Training complete.
Training time: 766.23 seconds


train.py --epochs 100 --tag 4cl+pools-1drop02-trfsflipsrotate
Training loss: 0.010, training acc: 99.512
Validation loss: 0.731, validation acc: 88.579
--------------------------------------------------
Training complete.
Training time: 1840.31 seconds

python train.py --epochs 50 --tag 4cl+pools-1drop01-trfsflipsrotate
Training loss: 0.030, training acc: 98.676
Validation loss: 0.304, validation acc: 90.355
--------------------------------------------------
Training complete.
Training time: 809.16 seconds

python train.py --epochs 50 --tag 3cl+pools-1drop01-notrsf
Training loss: 0.000, training acc: 100.000
Validation loss: 0.499, validation acc: 91.624
--------------------------------------------------
Training complete.
Training time: 1241.13 seconds

python train.py --epochs 50 --tag 3cl+pools-1drop01-trfs-flip-rotate
?

note: maybe lower dropout rate or to 0.3 reduce dropout to a single layer after the final conv layer

--epochs 40 --tag 3cl+pool+drop
Training loss: 0.058, training acc: 98.014
Validation loss: 0.627, validation acc: 89.848
--------------------------------------------------
Training complete.
Training time: 607.92 seconds

--epochs 50 --tag 3cl+pool+drop
Training loss: 13.767, training acc: 86.237
Validation loss: 18.450, validation acc: 73.350
--------------------------------------------------
Training complete.
Training time: 789.68 seconds

--tag 3cl+pool+drop e:30
Training loss: 0.104, training acc: 96.307
Validation loss: 0.801, validation acc: 77.919
--------------------------------------------------
Training complete.
Training time: 454.29 seconds

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
- https://streamlit.io/
- https://www.gradio.app/
