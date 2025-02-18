import torch
import matplotlib.pyplot as plt

def save_model(epochs, model, optimizer, loss_function):
    """
    Function to save the trained model.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_function,
                }, 'outputs/model.pth')

def save_acc_plot(train_acc, test_acc):
    """
    Function to save the accuracy plot.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='Train accuracy')
    plt.plot(test_acc, color='blue', linestyle='-', label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')

def save_loss_plot(train_loss, test_loss):
    """
    Function to save the loss plot.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='Train loss')
    plt.plot(test_loss, color='red', linestyle='-', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')