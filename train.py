# Import dependencies
import argparse # Allows to provide input parameters (parse arguments) from the command line instead of hardcoding them
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom modules
from datasets import train_loader, test_loader
from model import CNN, RN50
from utils import save_acc_plot, save_loss_plot

# Start timer
start_time = time.time()

# Set up argument parser
parser = argparse.ArgumentParser() # Initialize argument parser
parser.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs to train the model for')
parser.add_argument('-t', '--tag', type=str, default=10,
    help='model tag to save')
args = vars(parser.parse_args())

# Parameters
learning_rate = 1e-3 # Best learning rate seems to be 1e-3 or 1e-4
epochs = args['epochs']
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')
model = RN50().to(device) # Use ResNet50 model
# model = CNN().to(device) # Use custom CNN model
# model.load_state_dict(torch.load('models/CNN-4cl+pools-1drop01-notrfs-5e-model.pth')) # Load model weights
tag = args['tag']
print(model)
print(f'Tag: {tag}')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function (criterion)
criterion = nn.BCELoss() # nn.CrossEntropyLoss() for multi-class classification, nn.BCELoss() for binary classification

# Train function
def train(model, train_loader, optimizer, criterion):
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
        loss = criterion(outputs, labels)
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
def test(model, test_loader, criterion):
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
            loss = criterion(outputs, labels)
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
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
    test_epoch_loss, test_epoch_acc = test(model, test_loader, criterion)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    train_acc.append(train_epoch_acc)
    test_acc.append(test_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {test_epoch_loss:.3f}, validation acc: {test_epoch_acc:.3f}")
    print('-' * 50)
    # time.sleep(5)

# Save accuracy plot
save_acc_plot(train_acc=train_acc, test_acc=test_acc, model=model, tag=tag, epochs=epochs)

# Save loss plot
save_loss_plot(train_loss=train_loss, test_loss=test_loss, model=model, tag=tag, epochs=epochs)

# Save the trained model weights
torch.save(model.state_dict(), f'models/{model.__class__.__name__}-{tag}-{epochs}e.pth')
print('Training complete.')

# Print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training time: {elapsed_time:.2f} seconds')