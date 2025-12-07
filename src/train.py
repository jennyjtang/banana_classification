import torch
import torch.nn as nn
import torch.optim as optim
from src.cnn import BananaCNN
from sklearn.metrics import classification_report

def train_model(train_loader, val_loader, test_loader, epochs=5, lr=1e-3, save_path='banana_model.pth'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = BananaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Evaluation on test set
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nTEST SET RESULTS")
    print(classification_report(y_true, y_pred, target_names=["unripe", "ripe"]))

    return model