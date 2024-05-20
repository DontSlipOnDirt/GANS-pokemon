import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Generator, Discriminator

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset(folder, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
    ])
    
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    dataloader = get_dataset("pokemon_images/", 128, 128)
    latent_dim = 100
    # Create models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    print(generator)
    print(discriminator)