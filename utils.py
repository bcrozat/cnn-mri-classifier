# Import dependencies
import torch
import matplotlib.pyplot as plt

def save_acc_plot(train_acc, test_acc, model, tag, epochs):
    """
    Function to save the accuracy plot.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='Train accuracy')
    plt.plot(test_acc, color='blue', linestyle='-', label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'outputs/{model.__class__.__name__}-{tag}-{epochs}e-accuracy.png')

def save_loss_plot(train_loss, test_loss, model, tag, epochs):
    """
    Function to save the loss plot.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='Train loss')
    plt.plot(test_loss, color='red', linestyle='-', label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{model.__class__.__name__}-{tag}-{epochs}e-loss.png')