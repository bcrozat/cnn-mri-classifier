from PIL import Image
import gradio as gr
import torch

from model import CNN
from torchvision import transforms

print(f'Launching app.')

# Load model
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')
model = CNN().to(device)
model.load_state_dict(torch.load('models/CNN-4cl+pools-1drop01-notrfs-5e-model.pth', map_location=torch.device(device)))  # Load model weights
model.eval()
print(f'Model loaded.')

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images (ResNet standard size: 224x224)
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize (adjust based on dataset)
])

# Load test image
# img_path = 'data/test/no_tumor/image(1).jpg'

def classify_mri(img_path):
    img = Image.open(img_path)
    tensor = transform(img).unsqueeze(0).to(device)
    prediction = model(tensor).item()
    return prediction

demo = gr.Interface(
    fn=classify_mri,
    inputs=gr.Image(type='filepath', label='Upload MRI Image'),
    outputs=gr.Number(label='Tumor probability', precision=4),
    title='Brain Tumor Detector',
    description='This application process a MRI to detect if it contains a tumor or not.')

demo.launch()

print(f'App launched. Open http://127.0.0.1:7860.')