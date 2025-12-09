import torch
import torch.nn as nn
import torch.optim as optim
from src.cnn import ResNet_GAP_RD
from sklearn.metrics import classification_report

def get_accuracy_and_loss(model, data_loader, criterion, device):
    """Helper function to calculate accuracy and loss"""
    model.eval()
    correct = 0
    total_loss = 0
    total_count = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_count += data.size(0)
    
    accuracy = correct / total_count
    avg_loss = total_loss / total_count
    return accuracy, avg_loss

def train_model(train_loader, val_loader, test_loader, epochs=15, lr=1e-3, save_path='banana_model.pth'):
    """Train ResNet model with validation monitoring and early stopping"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("=" * 60)
    
    model = ResNet_GAP_RD().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float("inf")
    patience = 3
    epochs_without_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total_count = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_count += data.size(0)
            loss.backward()
            optimizer.step()

        # Calculate metrics
        train_accuracy = correct / total_count
        train_loss = train_loss / total_count
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        
        # Validation phase
        val_accuracy, val_loss = get_accuracy_and_loss(model, val_loader, criterion, device)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  \u2713 Best model saved (Val Loss: {val_loss:.4f})")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print("\n" + "=" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")

    # Evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nTEST SET RESULTS")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["Unripe", "Ripe"]))
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }