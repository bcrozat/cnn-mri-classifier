# Import dependencies
from datetime import datetime # Imports the datetime class from the datetime module
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
batch_size = 64

print(f'{datetime.today().strftime("%H:%M:%S")} Loading data.')

# Set directories
train_dir = Path(r'D:\Data\mri-brain-scans\train')
test_dir = Path(r'D:\Data\mri-brain-scans\test')

# Define transformation for data preparation & augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 224x224 (ResNet standard size)
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize (adjust based on dataset)
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 224x224 (ResNet standard size)
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