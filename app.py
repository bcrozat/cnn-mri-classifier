from PIL import Image
import gradio as gr
import torch

from model import CNN
from torchvision import transforms

# Load model
torch.serialization.add_safe_globals([torch.nn.modules.loss.BCELoss])
model = CNN()
model.load_state_dict(torch.load('models/cnn-v1-5e-model.pth')) # TODO: fix load & save model
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images (ResNet standard size: 224x224)
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize (adjust based on dataset)
])

# Load test image
img_test = 'data/test/tumor/glioma_tumor/image(1).jpg'
#
#
# # Define the transforms - update according to your training transforms
# image = Image.open('test.png')
# image = test_transform(image)
# # image = image.unsqueeze(0)  # Add batch dimension
# #
# # def classify_mri(img):
# #     return model(img)
# #
# # demo = gr.Interface(
# #     fn=greet,
# #     inputs=["text", "slider"],
# #     outputs=["text"],
# # )
# #
# # demo.launch()
