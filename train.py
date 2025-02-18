# Import dependencies
import argparse # Allows to provide input parameters (parse arguments) from the command line instead of hardcoding them
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom modules
from model import CNNModel
from datasets import train_loader, test_loader
from utils import save_model, save_acc_plot, save_loss_plot

# Set up argument parser
parser = argparse.ArgumentParser() # Initialize argument parser
parser.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs to train the model for')
args = vars(parser.parse_args())

# Parameters
learning_rate = 1e-3
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')
model = CNNModel().to(device)
print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function (criterion)
loss_function = nn.BCELoss() # nn.CrossEntropyLoss() for multi-class classification, nn.BCELoss() for binary classification

# Train function # TODO: to modify
#         _, preds = torch.max(outputs.data, 1)
#         train_running_correct += (preds == labels).sum().item()
#         # backpropagation
#         loss.backward()
#         # update the optimizer parameters
#         optimizer.step()
#     # loss and accuracy for the complete epoch
#     epoch_loss = train_running_loss / counter
#     epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
#     return epoch_loss, epoch_acc

# GPT train loop
# Train function # TODO: to modify
def train(model, train_loader, optimizer, loss_function):
    model.train()
    print('Training...')
    train_loss = 0.0
    train_correct = 0
    count = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        count += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device).float()  # Convert to float for BCELoss
        optimizer.zero_grad()
        # Forward pass
        outputs = model(image)
        # Compute loss
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        # Compute accuracy
        predictions = (outputs > 0.5).float() # Apply threshold to get binary predictions
        print(predictions)
        train_correct += (predictions == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update optimizer / parameters / weights
        optimizer.step()
    # Loss and accuracy for the complete epoch
    epoch_loss = train_loss / count
    epoch_acc = 100. * (train_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

# Test function
def test(model, test_loader, loss_function):
    model.eval()
    print('Testing')
    test_loss = 0.0
    test_correct = 0
    count = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            count += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device).float()  # Convert to float for BCELoss
            # Forward pass
            outputs = model(image)
            # Compute loss
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            # Compute accuracy
            predictions = (outputs > 0.5).float()  # Apply threshold to get binary predictions
            test_correct += (predictions == labels).sum().item()
    # Loss and accuracy for the complete epoch
    epoch_loss = test_loss / count
    epoch_acc = 100. * (test_correct / len(test_loader.dataset))
    return epoch_loss, epoch_acc

# Train loop
# Initialize lists to keep track of losses and accuracies
train_loss, test_loss = [], []
train_acc, test_acc = [], []
# Start training
print('Training started.')
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, loss_function)
    test_epoch_loss, test_epoch_acc = test(model, test_loader, loss_function)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    train_acc.append(train_epoch_acc)
    test_acc.append(test_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {test_epoch_loss:.3f}, validation acc: {test_epoch_acc:.3f}")
    print('-' * 50)
    # time.sleep(5)

# Save accuracy plot
save_acc_plot(train_acc, test_acc)

# Save loss plot
save_loss_plot(train_loss, test_loss)

# save the trained model weights
save_model(epochs, model, optimizer, loss_function)
print('Training complete.')

# Execute script
# Execute the following command in the terminal:
# python train.py --epochs 10

# Note: Around 50 epochs looks like a sweet spot for V1.
# Warning: Training takes very long (on CPU)!

# TODO: Try deeper networks
# TODO: Improve evaluation metrics (add confusion matrix, ROC curve, etc.)
# TODO: Add early stopping
# TODO: Add model checkpointing
# TODO: Add learning rate scheduler
# TODO: Add hyperparameter tuning
# TODO: Add data augmentation
# TODO: Test transfer learning
# TODO: Add tensorboard logging