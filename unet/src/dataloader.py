import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split

def get_transforms():
    """Define data transformations for images and segmentation masks."""
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize for efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),  # Keeps integer class values
    transforms.PILToTensor(),  # Ensures mask stays in int format (not float)
    # transforms.Lambda(lambda x: (x.to(torch.long) - 1).clamp(min=0))  # Convert to long and shift {1,2,3} → {0,1,2}
    ])  
    # mask_transform = transforms.Compose([
    #     transforms.Resize((128, 128)),  # Resize masks to match images
    #     transforms.ToTensor()
    # ])

    return image_transform, mask_transform

class OxfordPetDataset(torch.utils.data.Dataset):
    """Custom dataset wrapper for Oxford-IIIT Pet Dataset with segmentation masks."""

    def __init__(self, root="./data", split="train", transform=None, target_transform=None):
        self.dataset = OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=True)
        self.transform = transform
        self.target_transform = target_transform

        # Train-test split (80-20)
        total_size = len(self.dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        self.train_data, self.test_data = random_split(self.dataset, [train_size, test_size])

        self.data = self.train_data if split == "train" else self.test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # ✅ Fix: Convert labels from {1,2,3} → {0,1,2} inside dataset
        mask = (mask.to(torch.long) - 1).clamp(min=0)
        
        return image, mask

def get_dataloaders(config):
    """Returns train and test DataLoaders, using config for hyperparameters."""
    
    # Unpack with default values
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 2)

    image_transform, mask_transform = get_transforms()

    train_dataset = OxfordPetDataset(split="train", transform=image_transform, target_transform=mask_transform)
    test_dataset = OxfordPetDataset(split="test", transform=image_transform, target_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, test_loader