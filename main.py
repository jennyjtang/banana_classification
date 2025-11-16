from src.dataset import get_dataloaders
from src.train import train_model

DATA_DIR = "data_banana"

def main():
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)
    train_model(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()
