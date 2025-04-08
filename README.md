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



## Models

CNNModel: 4 conv layers, 1 fc layer
CNNModelV2: 5 conv layers, 1 fc layer
CNNModel7: 7 conv layers, 1 fc layer
CNN_MRI: ...

## Training

Execute the following command in the terminal:
`python train.py --epochs 10 --tag v4`

Note: Around 50 epochs looks like a sweet spot for V1.
# Warning: Training takes very long (on CPU)!

## Resources

- https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/
