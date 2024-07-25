import os
import torch
from torch import nn as nn
from PIL import Image
from torchvision import transforms
from my_net import mynet
import matplotlib.pyplot as plt

# Define the dataset class
class myDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_filenames = [f for f in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the model
def load_model(model_path):
    net = mynet(channels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(net).to(device)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dictionary directly if it was saved with nn.DataParallel
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# Add noise to the image
def add_noise(image, noise_level):
    noise = torch.FloatTensor(image.size()).normal_(mean=0, std=noise_level/255.).to(image.device)
    return image + noise

# Main function to process an image
def process_image(image_path, model_path, noise_level=25):
    # Load the model
    model, device = load_model(model_path)

    # Load the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Add noise to the image
    noisy_image_tensor = add_noise(image_tensor, noise_level)

    # Denoise the image
    with torch.no_grad():
        denoised_image_tensor = model(noisy_image_tensor)

    # Convert tensors to images
    image_tensor = image_tensor.squeeze(0).cpu()
    noisy_image_tensor = noisy_image_tensor.squeeze(0).cpu()
    denoised_image_tensor = denoised_image_tensor.squeeze(0).cpu()

    image_pil = transforms.ToPILImage()(image_tensor)
    noisy_image_pil = transforms.ToPILImage()(noisy_image_tensor)
    denoised_image_pil = transforms.ToPILImage()(denoised_image_tensor)

    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_pil)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_image_pil)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised_image_pil)
    plt.axis('off')

    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "D:\Dataset\DIV2K_train_HR\\0010.png"
    model_path = "mynet.pth"
    process_image(image_path, model_path)
