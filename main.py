import os
from src.dataset import get_dataloaders
from src.train import train_model

# Get absolute path to data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_banana")

def main():
    """Train the ResNet_GAP_RD model on banana dataset"""
    print("=" * 60)
    print("BANANA RIPENESS CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=32)
    
    # Train model
    train_model(
        train_loader, 
        val_loader, 
        test_loader, 
        epochs=15,  # Will early stop if validation loss doesn't improve
        lr=1e-3,
        save_path='banana_model.pth'
    )

if __name__ == "__main__":
    main()
