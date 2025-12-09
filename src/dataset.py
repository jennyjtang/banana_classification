import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BananaDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.images = []

        for fname in os.listdir(images_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                # determine class from filename
                if "Ripe_Banana" in fname:
                    label = 1
                elif "Raw_Banana" in fname:
                    label = 0
                else:
                    continue
                self.images.append((os.path.join(images_dir, fname), label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(data_dir="data_banana", batch_size=32):
    """Create dataloaders for training, validation, and test sets"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Standard transform for val/test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = BananaDataset(os.path.join(data_dir, "train/images"), transform=train_transform)
    val_ds = BananaDataset(os.path.join(data_dir, "val/images"), transform=eval_transform)
    test_ds = BananaDataset(os.path.join(data_dir, "test/images"), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
