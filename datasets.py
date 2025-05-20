# Import dependencies
from datetime import datetime # Imports the datetime class from the datetime module
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
batch_size = 64

print(f'{datetime.today().strftime("%H:%M:%S")} Loading data.')

# Set directories
train_dir = Path(r'data\train') # D:\Data\mri-brain-scans\train on Framework
test_dir = Path(r'data\test') # D:\Data\mri-brain-scans\test on Framework

# Define transformation for data preparation & augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 224x224 (ResNet standard size)
    transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
    transforms.RandomVerticalFlip(), # Randomly flip images vertically
    # transforms.RandomRotation(10), # Randomly rotate images by 10 degrees
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Randomly change brightness, contrast, saturation and hue
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize (adjust based on dataset)
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images (ResNet standard size: 224x224)
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize (adjust based on dataset)
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform, # Transform performed on data (images)
    target_transform=None # Transform performed on labels (if necessary)
)
test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform, # Transform performed on data (images)
    target_transform=None # Transform performed on labels (if necessary)
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Print class-to-index mapping
print(train_dataset.class_to_idx) # Should print {'no_tumor': 0, 'tumor': 1}

# Get a batch of data
images, labels = next(iter(train_loader))
print(f'Batch shape: {images.shape}')

# Indicate end
print(f'Data loaded.')